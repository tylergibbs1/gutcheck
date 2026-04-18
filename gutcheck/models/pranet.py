"""PraNet (Fan et al., MICCAI 2020) wrapper.

Uses the published `DengPingFan/PraNet` repo cloned to checkpoints/PraNet.
We import directly from it instead of reimplementing Res2Net.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms


def _ensure_on_path(pranet_repo: Path) -> None:
    p = str(pranet_repo.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


class PraNetWrapper:
    """Run inference with the official PraNet-19 checkpoint.

    Official training resolution is 352x352. We upsample the prediction back to
    the original image size and threshold at 0.5 on the sigmoided output.
    """

    INPUT_SIZE = 352

    def __init__(self, pranet_repo: str | Path, checkpoint: str | Path, device: str = "cuda"):
        pranet_repo = Path(pranet_repo)
        checkpoint = Path(checkpoint)
        _ensure_on_path(pranet_repo)

        from lib.PraNet_Res2Net import PraNet  # type: ignore

        self.device = torch.device(device)
        self.model = PraNet()
        # Work around Res2Net backbone URL being fetched at __init__ on PyTorch 2.x:
        # PraNet's Res2Net implementation tries to load the pretrained backbone by
        # default. If the user has the standalone backbone file we point to it; else
        # we rely on the checkpoint fully overwriting the state dict below.
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        # Some PraNet checkpoints are saved as raw state_dicts, some as dicts with "model".
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"[pranet] missing keys: {len(missing)} (e.g. {missing[:3]})")
        if unexpected:
            print(f"[pranet] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
        self.model.to(self.device).eval()

        # ImageNet normalization per PraNet/eval_polyp.py
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    @torch.inference_mode()
    def predict(self, image_rgb: np.ndarray) -> np.ndarray:
        """Return binary mask in the input image's original H x W."""
        h0, w0 = image_rgb.shape[:2]
        img = cv2.resize(image_rgb, (self.INPUT_SIZE, self.INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        t = self.normalize(t).unsqueeze(0).to(self.device)

        outputs = self.model(t)
        # PraNet forward returns (lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2)
        # final prediction is lateral_map_2.
        if isinstance(outputs, (list, tuple)):
            logits = outputs[-1]
        else:
            logits = outputs
        logits = F.interpolate(logits, size=(h0, w0), mode="bilinear", align_corners=False)
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        return prob > 0.5
