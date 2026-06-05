# advanced_engine/multimodal_fusion.py
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


class ClinicalMetadataMLP(nn.Module):
    """
    Encodes tabular clinical features into a 512-dim latent vector.
    Input features: [age_norm, gender_bin, mmse_norm, apoe_e4_bin, cdr_score_norm]
    """
    def __init__(self, input_dim: int = 5, hidden_dim: int = 256, output_dim: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)  # (B, 512)


class VolumericVisualEncoder(nn.Module):
    """
    3D ResNet-18 backbone (r3d_18) with a custom projection head.
    Input: (B, 3, D, H, W) — e.g. (B, 3, 32, 128, 128) volumetric MRI.
    Output: (B, 2048) latent visual embedding.
    """
    def __init__(self, freeze_backbone: bool = True):
        super().__init__()
        base = r3d_18(weights=R3D_18_Weights.DEFAULT)
        # Strip the classification head; keep spatial feature layers
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # → (B, 512, 1, 1, 1)
        self.project = nn.Sequential(
            nn.Flatten(),               # (B, 512)
            nn.Linear(512, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Dropout(p=0.4),
        )
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def unfreeze_top_layers(self, n_blocks: int = 2):
        children = list(self.backbone.children())
        for layer in children[-n_blocks:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)        # (B, 512, 1, 1, 1)
        return self.project(feats)      # (B, 2048)


class MultimodalFusionClassifier(nn.Module):
    """
    Late-fusion of visual (2048-d) and tabular (512-d) streams.
    Fusion: concat → gated attention → classification head.

    Tensor flow:
      visual_feat  (B, 2048)
      tabular_feat (B,  512)
          │               │
          └──── cat ───── ┘
               (B, 2560)
                  │
           Linear(2560→512)
            BatchNorm1d
              GELU
            Dropout(0.4)
                  │
           Linear(512→4)      ← 4 AD severity stages
    """
    def __init__(self, num_classes: int = 4, clinical_input_dim: int = 5):
        super().__init__()
        self.visual_encoder   = VolumericVisualEncoder(freeze_backbone=True)
        self.metadata_encoder = ClinicalMetadataMLP(input_dim=clinical_input_dim)

        # Gated cross-modal attention weight (scalar per modality)
        self.gate = nn.Sequential(
            nn.Linear(2048 + 512, 2),
            nn.Softmax(dim=-1),
        )

        self.fusion_head = nn.Sequential(
            nn.Linear(2048 + 512, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(p=0.4),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        volume:   torch.Tensor,   # (B, 3, D, H, W)
        metadata: torch.Tensor,   # (B, clinical_input_dim)
    ) -> dict:
        v_feat = self.visual_encoder(volume)     # (B, 2048)
        m_feat = self.metadata_encoder(metadata) # (B, 512)

        concat  = torch.cat([v_feat, m_feat], dim=-1)  # (B, 2560)
        gates   = self.gate(concat)                    # (B, 2)  — modality importance
        # Gate the contributions (for interpretability logging)
        gated   = torch.cat([
            v_feat * gates[:, 0:1],
            m_feat * gates[:, 1:2],
        ], dim=-1)                                     # (B, 2560)

        logits  = self.fusion_head(gated)              # (B, 4)
        return {
            "logits":       logits,
            "visual_feat":  v_feat,
            "tabular_feat": m_feat,
            "modal_gates":  gates,   # log these for explainability
        }