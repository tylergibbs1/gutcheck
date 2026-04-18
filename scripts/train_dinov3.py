"""Train the DinoV3SegHead on 900 Kvasir-SEG training images.

Backbone frozen. Head is MLP + Conv + classifier on top of DINOv3 ViT-L/16
patch tokens, upsampled bilinearly to the ground-truth resolution.
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gutcheck.data import KvasirSEG
from gutcheck.models.dinov3_seg import DinoV3Segmenter
from scripts._common import (
    DINOV3_DIR,
    DINOV3_HEAD_PATH,
    KVASIR_ROOT,
    KVASIR_TEST_SPLIT_DIR,
    LOGS,
)


class DinoTrainSet(Dataset):
    def __init__(self, dataset, segmenter: DinoV3Segmenter, mask_size: int = 224):
        self.ds = dataset
        self.proc = segmenter.processor
        self.input_size = segmenter.input_size
        self.mask_size = mask_size

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx):
        s = self.ds.get(idx)
        enc = self.proc(
            images=s.image,
            return_tensors="pt",
            size={"height": self.input_size, "width": self.input_size},
        )
        gt = torch.from_numpy(s.mask.astype(np.uint8)).float()
        gt = F.interpolate(gt[None, None], size=(self.mask_size, self.mask_size), mode="nearest")[0, 0]
        return {"pixel_values": enc["pixel_values"][0], "gt": gt}


def dice_bce_loss(logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, gt, reduction="mean")
    prob = torch.sigmoid(logits)
    inter = (prob * gt).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + gt.sum(dim=(1, 2, 3)) + 1e-6
    dice = 1 - (2 * inter / denom).mean()
    return bce + dice


def main():
    seed = 1234
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda")

    seg = DinoV3Segmenter(hf_repo=str(DINOV3_DIR), local_dir=DINOV3_DIR, input_size=448).to(device)
    seg.backbone.to(torch.bfloat16)

    kvasir = KvasirSEG(KVASIR_ROOT, split="train", held_out_ids_path=KVASIR_TEST_SPLIT_DIR)
    print(f"train n={len(kvasir)}")
    mask_size = 224
    train_ds = DinoTrainSet(kvasir, seg, mask_size=mask_size)
    loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)

    opt = AdamW(seg.head.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(loader) * 40)

    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / "train_dinov3.log"
    log_f = log_path.open("a", buffering=1)

    epochs = 40
    for ep in range(epochs):
        t0 = time.time()
        running = 0.0
        for batch in tqdm(loader, desc=f"epoch {ep}"):
            pv = batch["pixel_values"].to(device, dtype=torch.bfloat16)
            gt = batch["gt"].to(device).unsqueeze(1)
            patch_tokens, h, w = seg.backbone_features(pv)
            logits = seg.head(patch_tokens.float(), h, w, (mask_size, mask_size))
            loss = dice_bce_loss(logits, gt)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(seg.head.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()
        avg = running / max(1, len(loader))
        msg = f"ep={ep} loss={avg:.4f} dt={time.time()-t0:.1f}s"
        print(msg); log_f.write(msg + "\n")

    DINOV3_HEAD_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head": seg.head.state_dict(), "input_size": seg.input_size}, DINOV3_HEAD_PATH)
    print(f"saved head to {DINOV3_HEAD_PATH}")
    log_f.close()


if __name__ == "__main__":
    main()
