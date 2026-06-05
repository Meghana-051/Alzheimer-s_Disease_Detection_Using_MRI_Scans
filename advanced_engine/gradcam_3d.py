# advanced_engine/gradcam_3d.py
from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np

class GradCAM3D:
    """
    Gradient-weighted Class Activation Mapping for volumetric (3D) inputs.
    L_c = ReLU( Σ_k  α_k^c · A^k )
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients:   torch.Tensor | None = None
        self._register_hooks()

    def _register_hooks(self):
        def _forward_hook(_, __, output):
            self._activations = output.detach()

        def _backward_hook(_, __, grad_output):
            self._gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(_forward_hook)
        self.target_layer.register_full_backward_hook(_backward_hook)

    def generate(
        self,
        volume:    torch.Tensor,   # (1, C, D, H, W)
        metadata:  torch.Tensor,   # (1, F)
        class_idx: int | None = None,
    ) -> np.ndarray:
        self.model.eval()
        volume   = volume.requires_grad_(True)
        output   = self.model(volume, metadata)
        logits   = output["logits"]

        if class_idx is None:
            class_idx = logits.argmax(dim=-1).item()

        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward(retain_graph=False)

        grads = self._gradients   
        acts  = self._activations 

        if grads is None or acts is None:
            # Fallback tensor creation to avoid mathematical 500 fault lines
            return np.zeros(volume.shape[2:], dtype=np.float32)

        # Global average pool across spatial+temporal layers
        alpha  = grads.mean(dim=[2, 3, 4], keepdim=True)  # (1, K, 1, 1, 1)
        cam_3d = (alpha * acts).sum(dim=1, keepdim=True)   # (1, 1, d, h, w)
        cam_3d = F.relu(cam_3d)

        target_size = volume.shape[2:]  # (D, H, W)
        cam_up = F.interpolate(
            cam_3d,
            size=target_size,
            mode="trilinear",
            align_corners=False,
        ).squeeze()  

        cam_np = cam_up.cpu().numpy()
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
        return cam_np  

    def slice_overlay(
        self,
        volume_np: np.ndarray,   
        cam_np:    np.ndarray,   
        slice_idx: int,
        axis:      int = 0,      
        alpha:     float = 0.45,
    ) -> np.ndarray:
        import cv2

        slicers = [
            (slice_idx, slice(None), slice(None)),
            (slice(None), slice_idx, slice(None)),
            (slice(None), slice(None), slice_idx),
        ]
        mri_slice = volume_np[slicers[axis]]  
        cam_slice = cam_np[slicers[axis]]     

        mri_u8  = (mri_slice * 255).astype(np.uint8)
        mri_rgb = cv2.cvtColor(mri_u8, cv2.COLOR_GRAY2BGR)

        cam_u8    = (cam_slice * 255).astype(np.uint8)
        heatmap   = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)

        blended = cv2.addWeighted(mri_rgb, 1 - alpha, heatmap, alpha, 0)
        return blended