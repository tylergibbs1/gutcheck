"""SAM 3 text-promptable segmentation wrapper.

Uses `transformers.Sam3Model` / `Sam3Processor` with the `facebook/sam3` checkpoint.
(SAM 3.1 itself only ships as a raw Meta-repo checkpoint and is API-compatible
with SAM 3 in terms of paradigm.)

Two modes:
    - zero-shot: just `predict("polyp", image)`
    - LoRA-adapted: `attach_lora(model, target_modules=...)` + fine-tune mask decoder
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Sam3Model, Sam3Processor

DEFAULT_PROMPT = "polyp"


class Sam3Wrapper:
    def __init__(
        self,
        model_dir: str | Path,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.processor = Sam3Processor.from_pretrained(str(model_dir))
        self.model = Sam3Model.from_pretrained(str(model_dir), torch_dtype=dtype).to(device).eval()
        self.device = torch.device(device)
        self.dtype = dtype

    @torch.inference_mode()
    def predict(
        self,
        image_rgb: np.ndarray,
        prompt: str = DEFAULT_PROMPT,
    ) -> np.ndarray:
        """Return binary HxW mask at the input image's resolution."""
        h0, w0 = image_rgb.shape[:2]
        inputs = self.processor(
            images=image_rgb, text=prompt, return_tensors="pt"
        ).to(self.device)
        out = self.model(**inputs, multimask_output=False)
        sem = self.processor.post_process_semantic_segmentation(
            out, target_sizes=inputs["original_sizes"].tolist()
        )[0]
        return sem.cpu().numpy().astype(bool)

    @torch.inference_mode()
    def predict_batch(
        self, images_rgb: list[np.ndarray], prompt: str = DEFAULT_PROMPT
    ) -> list[np.ndarray]:
        inputs = self.processor(
            images=images_rgb,
            text=[prompt] * len(images_rgb),
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        out = self.model(**inputs, multimask_output=False)
        sem_list = self.processor.post_process_semantic_segmentation(
            out, target_sizes=inputs["original_sizes"].tolist()
        )
        return [s.cpu().numpy().astype(bool) for s in sem_list]


# ---------------------------------------------------------------------------
# LoRA adaptation of the mask decoder
# ---------------------------------------------------------------------------


def find_lora_target_modules(model: nn.Module, scopes: Iterable[str] = ("mask_decoder",)) -> list[str]:
    """Collect fully-qualified nn.Linear module names under `scopes` for LoRA.
    Returns absolute dotted paths (peft treats full-path targets as exact matches)."""
    hits: list[str] = []
    scopes = tuple(scopes)
    for name, module in model.named_modules():
        if not any(s in name for s in scopes):
            continue
        if isinstance(module, nn.Linear):
            hits.append(name)
    return hits


def attach_lora(model: Sam3Model, r: int = 16, alpha: int = 32, dropout: float = 0.05, scopes: Iterable[str] = ("mask_decoder",)):
    """Wrap the mask decoder with LoRA adapters. Everything else stays frozen."""
    from peft import LoraConfig, get_peft_model

    targets = find_lora_target_modules(model, scopes=scopes)
    if not targets:
        raise RuntimeError("No linear layers found under scopes; inspect model.named_modules().")

    # Freeze everything first so peft only trains the LoRA params.
    for p in model.parameters():
        p.requires_grad = False

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=targets,
        task_type=None,
    )
    peft_model = get_peft_model(model, cfg)
    peft_model.print_trainable_parameters()
    return peft_model, targets


def sam3_mask_loss(out, gt_masks: torch.Tensor) -> torch.Tensor:
    """Dice + BCE loss against SAM 3's `semantic_seg` (1 logit per pixel, low-res)."""
    sem = out.semantic_seg  # [B, 1, H, W]
    b = sem.shape[0]
    gt = F.interpolate(gt_masks.float().unsqueeze(1), size=sem.shape[-2:], mode="nearest")
    gt = gt.to(sem.dtype)
    bce = F.binary_cross_entropy_with_logits(sem, gt, reduction="mean")

    prob = torch.sigmoid(sem).float()
    gt_f = gt.float()
    dims = (2, 3)
    inter = (prob * gt_f).sum(dim=dims)
    denom = prob.sum(dim=dims) + gt_f.sum(dim=dims) + 1e-6
    dice = 1 - (2 * inter / denom).mean()
    return bce + dice
