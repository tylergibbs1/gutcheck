"""Confirmatory experiment: does SAM 3.1 recover if we crop out the scope vignette?

For every image in both test sets:
  1. detect the scope-visible region (non-dark connected area)
  2. crop tightly to that region
  3. run SAM 3.1 zero-shot on the crop
  4. paste the predicted mask back into the original frame
  5. compare to the un-cropped SAM 3.1 Dice

Produces a paired-bar chart and per-image delta table.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path("/workspace/gutcheck")
DATA = ROOT / "data"
RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "failure_analysis" / "scope_crop"
CKPT = ROOT / "checkpoints" / "sam3.1" / "sam3.1_multiplex.pt"

OUT_DIR.mkdir(parents=True, exist_ok=True)

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def detect_scope_bbox(rgb: np.ndarray, dark_thresh: int = 20, min_frac: float = 0.1) -> tuple[int, int, int, int]:
    """Return (y0, x0, y1, x1) of the scope-visible region.
    Strategy: threshold luminance, keep the largest bright connected component, return its bbox.
    If the scope fills the whole image, returns full extent."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bright = (gray > dark_thresh).astype(np.uint8)
    h, w = gray.shape
    if bright.mean() > 0.97:
        return 0, 0, h, w
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    if n <= 1:
        return 0, 0, h, w
    best = max(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA])
    x, y, bw, bh, area = stats[best]
    if area / (h * w) < min_frac:
        return 0, 0, h, w
    # small inset margin so we don't clip polyps at the very edge
    pad = 4
    return max(0, y - pad), max(0, x - pad), min(h, y + bh + pad), min(w, x + bw + pad)


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    denom = p.sum() + g.sum()
    return 1.0 if denom == 0 else float(2 * inter / (denom + 1e-7))


def sam31_predict(proc, rgb: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(rgb)
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        state = proc.set_image(pil)
        out = proc.set_text_prompt(state=state, prompt="polyp")
    masks = out["masks"].cpu().numpy()
    if masks.ndim == 4:
        masks = masks[:, 0]
    scores = out["scores"].float().cpu().numpy()
    if len(masks) == 0:
        return np.zeros(rgb.shape[:2], dtype=bool)
    top = int(scores.argmax())
    p = masks[top].astype(bool)
    if p.shape != rgb.shape[:2]:
        p = cv2.resize(p.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return p


def load_datasets():
    rows = []
    # Kvasir test
    kv_dir = DATA / "TestDataset" / "Kvasir"
    for p in sorted((kv_dir / "images").glob("*.png")):
        img = np.asarray(Image.open(p).convert("RGB"))
        gt = cv2.imread(str(kv_dir / "masks" / p.name), 0) > 127
        rows.append(("kvasir", p.stem, img, gt))
    # CVC-ClinicDB
    cvc_dir = DATA / "CVC-ClinicDB"
    for p in sorted((cvc_dir / "images").glob("*.png")):
        img = np.asarray(Image.open(p).convert("RGB"))
        gt = cv2.imread(str(cvc_dir / "masks" / p.name), 0) > 127
        rows.append(("cvc_clinicdb", p.stem, img, gt))
    return rows


def main():
    print("Loading SAM 3.1...")
    model = build_sam3_image_model(checkpoint_path=str(CKPT), load_from_HF=False)
    proc = Sam3Processor(model, confidence_threshold=0.0)

    samples = load_datasets()
    print(f"running {len(samples)} images through SAM 3.1 twice (full frame + scope-crop)...")

    rows = []
    for ds, image_id, rgb, gt in tqdm(samples):
        h, w = rgb.shape[:2]

        # original
        t0 = time.perf_counter()
        pred_full = sam31_predict(proc, rgb)
        dt_full = time.perf_counter() - t0

        # scope-cropped
        y0, x0, y1, x1 = detect_scope_bbox(rgb)
        crop = rgb[y0:y1, x0:x1]
        scope_frac = 1.0 - (crop.shape[0] * crop.shape[1]) / (h * w)
        t0 = time.perf_counter()
        pred_crop_local = sam31_predict(proc, crop)
        dt_crop = time.perf_counter() - t0

        # paste back
        pred_crop = np.zeros_like(gt, dtype=bool)
        pred_crop[y0:y1, x0:x1] = pred_crop_local

        rows.append({
            "dataset": ds,
            "image_id": image_id,
            "dice_full": dice_score(pred_full, gt),
            "dice_crop": dice_score(pred_crop, gt),
            "scope_removed_frac": float(scope_frac),
            "fps_full": 1.0 / max(dt_full, 1e-6),
            "fps_crop": 1.0 / max(dt_crop, 1e-6),
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    df["delta"] = df["dice_crop"] - df["dice_full"]
    df.to_csv(OUT_DIR / "per_image.csv", index=False)

    # Summary by dataset
    for ds in ["kvasir", "cvc_clinicdb"]:
        sub = df[df["dataset"] == ds]
        print(f"\n{ds} (n={len(sub)}):")
        print(f"  full-frame Dice mean = {sub['dice_full'].mean():.3f}  median = {sub['dice_full'].median():.3f}")
        print(f"  scope-crop Dice mean = {sub['dice_crop'].mean():.3f}  median = {sub['dice_crop'].median():.3f}")
        print(f"  delta mean           = {sub['delta'].mean():+.3f}  median = {sub['delta'].median():+.3f}")
        # recovery rate for catastrophic cases
        bad = sub[sub["dice_full"] < 0.2]
        if len(bad):
            recovered = bad[bad["dice_crop"] >= 0.5]
            print(f"  catastrophic images (Dice<0.2): {len(bad)}, recovered to >=0.5: {len(recovered)} ({len(recovered)/len(bad)*100:.0f}%)")

    # Save a simple summary JSON
    summary = {}
    for ds in ["kvasir", "cvc_clinicdb"]:
        sub = df[df["dataset"] == ds]
        bad = sub[sub["dice_full"] < 0.2]
        summary[ds] = {
            "n": len(sub),
            "dice_full_mean": float(sub["dice_full"].mean()),
            "dice_full_median": float(sub["dice_full"].median()),
            "dice_crop_mean": float(sub["dice_crop"].mean()),
            "dice_crop_median": float(sub["dice_crop"].median()),
            "n_catastrophic": int(len(bad)),
            "n_recovered": int(len(bad[bad["dice_crop"] >= 0.5])),
        }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
