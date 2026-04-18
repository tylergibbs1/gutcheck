"""Evaluate DINOv3 + trained decoder head."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gutcheck.data import KvasirSEG, CVCClinicDB
from gutcheck.metrics import MetricAccumulator, append_to_master
from gutcheck.models.dinov3_seg import DinoV3Segmenter, predict
from scripts._common import (
    DINOV3_DIR,
    DINOV3_HEAD_PATH,
    KVASIR_ROOT,
    KVASIR_TEST_SPLIT_DIR,
    CVC_TEST_ROOT,
    RESULTS,
    METRICS_MASTER,
)

APPROACH = "dinov3"


def load() -> DinoV3Segmenter:
    device = torch.device("cuda")
    ckpt = torch.load(str(DINOV3_HEAD_PATH), map_location="cpu", weights_only=False)
    seg = DinoV3Segmenter(hf_repo=str(DINOV3_DIR), local_dir=DINOV3_DIR, input_size=ckpt.get("input_size", 448))
    seg.head.load_state_dict(ckpt["head"])
    seg.to(device)
    seg.backbone.to(torch.bfloat16)
    seg.eval()
    return seg


def save_pred(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask.astype(np.uint8)) * 255)


@torch.inference_mode()
def run(dataset, ds_name: str, seg: DinoV3Segmenter) -> dict:
    acc = MetricAccumulator(APPROACH, ds_name)
    pred_dir = RESULTS / "preds" / APPROACH / ds_name
    device = torch.device("cuda")
    for sample in tqdm(dataset, desc=f"{APPROACH}/{ds_name}", total=len(dataset)):
        h0, w0 = sample.image.shape[:2]
        enc = seg.processor(
            images=sample.image, return_tensors="pt",
            size={"height": seg.input_size, "width": seg.input_size},
        )
        pv = enc["pixel_values"].to(device, dtype=torch.bfloat16)
        t0 = time.perf_counter()
        patch_tokens, h, w = seg.backbone_features(pv)
        logits = seg.head(patch_tokens.float(), h, w, (h0, w0))
        mask = (torch.sigmoid(logits)[0, 0] > 0.5).cpu().numpy()
        dt = time.perf_counter() - t0
        acc.add(sample.image_id, mask, sample.mask, fps=1.0 / max(dt, 1e-6))
        save_pred(pred_dir / f"{sample.image_id}.png", mask)
    acc.save(RESULTS / "metrics" / APPROACH / ds_name)
    print(f"[{APPROACH}/{ds_name}] ", acc.summary())
    return acc.summary()


def main():
    seg = load()
    kvasir = KvasirSEG(KVASIR_ROOT, split="val", held_out_ids_path=KVASIR_TEST_SPLIT_DIR)
    cvc = CVCClinicDB(CVC_TEST_ROOT)
    for s in [run(kvasir, "kvasir", seg), run(cvc, "cvc_clinicdb", seg)]:
        append_to_master(METRICS_MASTER, s)


if __name__ == "__main__":
    main()
