"""
gradcam.py
----------
Grad-CAM implementation for the CelebA attribute model (ResNet-18).
Generates class-discriminative heatmaps showing which face regions
most influenced a specific attribute prediction.
"""

import numpy as np
import torch
import torch.nn as nn
import cv2
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Generic Grad-CAM wrapper for any PyTorch model.

    Usage:
        cam = GradCAM(model, target_layer=model.layer4[-1])
        heatmap = cam.generate(input_tensor, class_idx=5)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._feature_maps: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self._feature_maps = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for the target class.

        Args:
            input_tensor: Preprocessed input tensor (1, C, H, W).
            class_idx: Target class index. Uses argmax if None.

        Returns:
            Normalized heatmap array of shape (H_in, W_in) in [0, 1].
        """
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)

        output = self.model(input_tensor)

        # Handle multi-output models (e.g., AgeGenderModel returns tuple)
        if isinstance(output, tuple):
            output = output[1]  # Use gender logits

        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        if self._gradients is None or self._feature_maps is None:
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._feature_maps).sum(dim=1).squeeze()  # (H, W)
        cam = torch.relu(cam).cpu().numpy()

        # Normalize
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam.astype(np.float32)


def apply_heatmap_overlay(
    face_bgr: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on a face image.

    Args:
        face_bgr: Original face image (BGR).
        cam: Normalized heatmap array in [0, 1].
        alpha: Heatmap opacity (0=invisible, 1=opaque).
        colormap: OpenCV colormap constant.

    Returns:
        BGR image with heatmap overlay.
    """
    h, w = face_bgr.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), colormap)
    overlay = cv2.addWeighted(face_bgr, 1.0 - alpha, heatmap, alpha, 0)
    return overlay


def generate_attribute_gradcam(
    face_bgr: np.ndarray,
    model: nn.Module,
    transform,
    device: torch.device,
    target_attribute_idx: int = 13,  # Default: "Smiling"
) -> Tuple[np.ndarray, float]:
    """
    High-level function to generate Grad-CAM for a CelebA attribute.

    Args:
        face_bgr: BGR face image.
        model: CelebAAttributeModel instance.
        transform: Preprocessing transform.
        device: Torch device.
        target_attribute_idx: Index in CELEBA_ATTRIBUTES list.

    Returns:
        (heatmap_overlay_bgr, attribute_confidence)
    """
    import cv2 as _cv2
    from PIL import Image

    rgb = _cv2.cvtColor(face_bgr, _cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    tensor = transform(pil_img).unsqueeze(0).to(device)

    # Target layer: last residual block in ResNet-18
    try:
        target_layer = model.backbone.layer4[-1]
    except AttributeError:
        target_layer = list(model.backbone.children())[-2]

    grad_cam = GradCAM(model, target_layer)
    cam = grad_cam.generate(tensor, class_idx=target_attribute_idx)

    # Get confidence for the target attribute
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
        confidence = float(probs[target_attribute_idx])

    heatmap_overlay = apply_heatmap_overlay(face_bgr, cam, alpha=0.45)
    return heatmap_overlay, confidence
