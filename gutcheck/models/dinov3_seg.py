"""DINOv3 + lightweight segmentation head.

The backbone (frozen) produces patch tokens at stride ~16. We upsample those
into a dense feature map and run a small MLP/conv decoder to predict a 1-channel
mask logit per pixel. This mirrors the "SegDINO" recipe: linear head over patch
features, trained only on the decoder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel


class DinoV3SegHead(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 256, num_classes: int = 1):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.classifier = nn.Conv2d(hidden, num_classes, kernel_size=1)

    def forward(self, tokens: torch.Tensor, h_feat: int, w_feat: int, out_hw: tuple[int, int]) -> torch.Tensor:
        # tokens: [B, N, C] (patch tokens only, CLS already removed)
        b, n, c = tokens.shape
        x = self.norm(tokens)
        x = x.transpose(1, 2).reshape(b, c, h_feat, w_feat)
        x = self.proj(x)
        logits = self.classifier(x)
        logits = F.interpolate(logits, size=out_hw, mode="bilinear", align_corners=False)
        return logits


class DinoV3Segmenter(nn.Module):
    """Frozen DINOv3 backbone + trainable DinoV3SegHead."""

    def __init__(self, hf_repo: str, local_dir: str | Path | None = None, input_size: int = 518):
        super().__init__()
        path = str(local_dir) if local_dir else hf_repo
        self.processor = AutoImageProcessor.from_pretrained(path)
        self.backbone = AutoModel.from_pretrained(path)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        hidden = getattr(self.backbone.config, "hidden_size", None)
        if hidden is None:
            raise RuntimeError("Could not infer hidden_size from backbone config")
        self.head = DinoV3SegHead(in_channels=hidden, hidden=256, num_classes=1)
        # input_size must be divisible by patch size for grid reshape
        patch = getattr(self.backbone.config, "patch_size", 16)
        self.input_size = (input_size // patch) * patch
        self.patch_size = patch

    def backbone_features(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """Return patch tokens [B, H*W, C] and spatial grid (H, W)."""
        with torch.no_grad():
            out = self.backbone(pixel_values=pixel_values)
        hidden = out.last_hidden_state  # [B, 1 + R + H*W, C]
        b, n, c = hidden.shape
        h = w = self.input_size // self.patch_size
        # Drop CLS and any register tokens. DINOv3 ViT-L/16 default uses 4 register tokens.
        n_reg = getattr(self.backbone.config, "num_register_tokens", 0)
        patch_tokens = hidden[:, 1 + n_reg :, :]
        # Some configs may report 0 register tokens but actually have them. Safety check:
        expected = h * w
        if patch_tokens.shape[1] != expected:
            # fallback: assume no extras, keep last expected tokens
            patch_tokens = hidden[:, -expected:, :]
        return patch_tokens, h, w

    def forward(self, pixel_values: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
        patch_tokens, h, w = self.backbone_features(pixel_values)
        return self.head(patch_tokens, h, w, out_hw)


def preprocess_batch(processor, images_rgb_list, target_size: int) -> torch.Tensor:
    inputs = processor(
        images=images_rgb_list,
        return_tensors="pt",
        size={"height": target_size, "width": target_size},
    )
    return inputs["pixel_values"]


@torch.inference_mode()
def predict(model: DinoV3Segmenter, image_rgb: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    h0, w0 = image_rgb.shape[:2]
    px = preprocess_batch(model.processor, [image_rgb], model.input_size).to(device)
    logits = model(px, out_hw=(h0, w0))
    return (torch.sigmoid(logits)[0, 0].cpu().numpy() > 0.5)
