# federated/client.py
from __future__ import annotations
import torch
import flwr as fl
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader
from advanced_engine.multimodal_fusion import MultimodalFusionClassifier


class NeuroClinicalClient(fl.client.NumPyClient):
    """
    Hospital-side Flower client.
    Raw MRI data NEVER leaves this node — only model weight deltas are sent.
    """

    def __init__(
        self,
        hospital_id:  str,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        device:       torch.device,
    ):
        self.hospital_id  = hospital_id
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.model        = MultimodalFusionClassifier().to(device)
        self.criterion    = torch.nn.CrossEntropyLoss()

    # Convert model weights ↔ NumPy arrays (Flower protocol)
    def get_parameters(self, config) -> list[np.ndarray]:
        return [p.cpu().detach().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters: list[np.ndarray]):
        state_dict = OrderedDict(
            (k, torch.tensor(v))
            for k, v in zip(self.model.state_dict().keys(), parameters)
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config) -> tuple:
        self.set_parameters(parameters)
        lr      = config.get("learning_rate", 1e-4)
        epochs  = config.get("local_epochs", 3)
        opt     = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        self.model.train()
        total_loss = 0.0
        n_samples  = 0

        for _ in range(epochs):
            for volume, metadata, labels in self.train_loader:
                volume   = volume.to(self.device)
                metadata = metadata.to(self.device)
                labels   = labels.to(self.device)

                opt.zero_grad()
                out  = self.model(volume, metadata)
                loss = self.criterion(out["logits"], labels)
                loss.backward()

                # Gradient clipping — first line of differential privacy defence
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                opt.step()

                total_loss += loss.item() * labels.size(0)
                n_samples  += labels.size(0)
            scheduler.step()

        avg_loss = total_loss / max(n_samples, 1)
        return self.get_parameters(config={}), n_samples, {"train_loss": avg_loss}

    def evaluate(self, parameters, config) -> tuple:
        self.set_parameters(parameters)
        self.model.eval()
        total_loss, correct, n_samples = 0.0, 0, 0

        with torch.no_grad():
            for volume, metadata, labels in self.val_loader:
                volume   = volume.to(self.device)
                metadata = metadata.to(self.device)
                labels   = labels.to(self.device)
                out      = self.model(volume, metadata)
                loss     = self.criterion(out["logits"], labels)
                preds    = out["logits"].argmax(dim=-1)
                total_loss += loss.item() * labels.size(0)
                correct    += (preds == labels).sum().item()
                n_samples  += labels.size(0)

        return (
            total_loss / max(n_samples, 1),
            n_samples,
            {
                "val_accuracy": correct / max(n_samples, 1),
                "val_loss":     total_loss / max(n_samples, 1),
            },
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hospital-id", required=True)
    parser.add_argument("--server",      default="localhost:8080")
    args = parser.parse_args()

    # Each hospital builds its own local DataLoaders from local encrypted storage
    # train_loader, val_loader = build_local_dataloaders(args.hospital_id)
    client = NeuroClinicalClient(
        hospital_id  = args.hospital_id,
        train_loader = None,   # substitute real loaders
        val_loader   = None,
        device       = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    fl.client.start_client(server_address=args.server, client=client.to_client())