# federated/server.py
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Metrics
from typing import List, Tuple, Optional


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate validation accuracy weighted by number of local samples."""
    total   = sum(n for n, _ in metrics)
    acc_agg = sum(n * m["val_accuracy"] for n, m in metrics) / total
    loss_agg= sum(n * m["val_loss"]     for n, m in metrics) / total
    return {"val_accuracy": acc_agg, "val_loss": loss_agg}


strategy = FedAvg(
    fraction_fit             = 0.8,    # sample 80 % of clients per round
    fraction_evaluate        = 1.0,
    min_fit_clients          = 3,
    min_evaluate_clients     = 3,
    min_available_clients    = 3,
    evaluate_metrics_aggregation_fn = weighted_average,
    # Differential privacy: clip gradients, add Gaussian noise
    # In production: wrap with fl.server.strategy.DifferentialPrivacyClientSideAdaptiveClipping
)

if __name__ == "__main__":
    fl.server.start_server(
        server_address = "0.0.0.0:8080",
        config         = fl.server.ServerConfig(num_rounds=20),
        strategy       = strategy,
    )