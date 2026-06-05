# reset_weights.py
import os
import torch
from advanced_engine.multimodal_fusion import MultimodalFusionClassifier

def reset_model_weights():
    os.makedirs("checkpoints", exist_ok=True)
    # Reinitialize the structure to map baseline 3D block values
    model = MultimodalFusionClassifier(num_classes=4, clinical_input_dim=5)
    torch.save(model.state_dict(), "checkpoints/fusion_model.pt")
    print("[🎉] Weights mapped and synced successfully for 3D ResNet layers!")

if __name__ == "__main__":
    reset_model_weights()