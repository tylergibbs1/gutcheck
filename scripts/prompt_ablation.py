"""Run SAM 3.1 with two prompts ("polyp" vs "growth") on both test sets,
save per-image Dice for each, compute deltas, and produce the visuals."""

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
CKPT = ROOT / "checkpoints" / "sam3.1" / "sam3.1_multiplex.pt"
OUT_DIR = RESULTS / "prompt_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def detect_scope_bbox(rgb: np.ndarray, dark_thresh: int = 20, min_frac: float = 0.1):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bright = (gray > dark_thresh).astype(np.uint8)
    h, w = gray.shape
    if bright.mean() > 0.97:
        return 0, 0, h, w
    n, _, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    if n <= 1:
        return 0, 0, h, w
    best = max(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA])
    x, y, bw, bh, area = stats[best]
    if area / (h * w) < min_frac:
        return 0, 0, h, w
    pad = 4
    return max(0, y - pad), max(0, x - pad), min(h, y + bh + pad), min(w, x + bw + pad)


def preprocess(rgb: np.ndarray):
    y0, x0, y1, x1 = detect_scope_bbox(rgb)
    if (y1 - y0) == rgb.shape[0] and (x1 - x0) == rgb.shape[1]:
        return rgb, None
    return rgb[y0:y1, x0:x1], (y0, x0, y1, x1)


def predict_one(proc, rgb_in: np.ndarray, prompt: str) -> np.ndarray:
    pil = Image.fromarray(rgb_in)
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        st = proc.set_image(pil)
        o = proc.set_text_prompt(state=st, prompt=prompt)
    m = o["masks"].cpu().numpy()
    if m.ndim == 4: m = m[:, 0]
    if len(m) == 0:
        return np.zeros(rgb_in.shape[:2], dtype=bool)
    s = o["scores"].float().cpu().numpy()
    top = int(s.argmax())
    return m[top].astype(bool)


def paste_back(pred_local, crop, gt_shape):
    if crop is None:
        pred = pred_local
    else:
        y0, x0, y1, x1 = crop
        if pred_local.shape != (y1 - y0, x1 - x0):
            pred_local = cv2.resize(pred_local.astype(np.uint8), (x1 - x0, y1 - y0),
                                     interpolation=cv2.INTER_NEAREST) > 0
        pred = np.zeros(gt_shape, dtype=bool)
        pred[y0:y1, x0:x1] = pred_local
    if pred.shape != gt_shape:
        pred = cv2.resize(pred.astype(np.uint8), (gt_shape[1], gt_shape[0]),
                          interpolation=cv2.INTER_NEAREST) > 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    pred = cv2.morphologyEx(pred.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
    if pred.any():
        flood = pred.astype(np.uint8).copy()
        h2, w2 = flood.shape
        mask_ff = np.zeros((h2 + 2, w2 + 2), np.uint8)
        cv2.floodFill(flood, mask_ff, (0, 0), 2)
        pred = (flood != 2)
    return pred


def dice(p, g):
    p, g = p.astype(bool), g.astype(bool)
    inter = np.logical_and(p, g).sum()
    denom = p.sum() + g.sum()
    return 1.0 if denom == 0 else float(2 * inter / (denom + 1e-7))


def load_samples():
    rows = []
    for base, ds in [(DATA / "TestDataset" / "Kvasir", "kvasir"), (DATA / "CVC-ClinicDB", "cvc_clinicdb")]:
        for p in sorted((base / "images").glob("*.png")):
            img = np.asarray(Image.open(p).convert("RGB"))
            gt = cv2.imread(str(base / "masks" / p.name), 0) > 127
            rows.append((ds, p.stem, img, gt))
    return rows


def main():
    prompts = ["polyp", "growth"]
    print("Loading SAM 3.1...")
    model = build_sam3_image_model(checkpoint_path=str(CKPT), load_from_HF=False)
    proc = Sam3Processor(model, confidence_threshold=0.0)

    samples = load_samples()
    print(f"running {len(samples)} images with hflip TTA x {len(prompts)} prompts = {len(samples) * 2 * len(prompts)} forward passes")

    per_prompt = {p: {"dataset": [], "image_id": [], "dice": []} for p in prompts}
    pred_cache = {p: {} for p in prompts}

    for ds, image_id, rgb, gt in tqdm(samples, desc="ablation"):
        rgb_in, crop = preprocess(rgb)
        for prompt in prompts:
            p_orig = predict_one(proc, rgb_in, prompt)
            p_flip = predict_one(proc, rgb_in[:, ::-1].copy(), prompt)[:, ::-1]
            p_local = p_orig | p_flip
            pred = paste_back(p_local, crop, gt.shape)
            pred_cache[prompt][(ds, image_id)] = pred
            per_prompt[prompt]["dataset"].append(ds)
            per_prompt[prompt]["image_id"].append(image_id)
            per_prompt[prompt]["dice"].append(dice(pred, gt))
            pred_dir = OUT_DIR / prompt / ds
            pred_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(pred_dir / f"{image_id}.png"), pred.astype(np.uint8) * 255)

    # save per-image tables
    import pandas as pd
    df_rows = []
    ids = list(zip(per_prompt["polyp"]["dataset"], per_prompt["polyp"]["image_id"]))
    for (ds, iid), d_polyp, d_growth in zip(ids, per_prompt["polyp"]["dice"], per_prompt["growth"]["dice"]):
        df_rows.append({"dataset": ds, "image_id": iid,
                        "dice_polyp": d_polyp, "dice_growth": d_growth,
                        "delta": d_growth - d_polyp})
    df = pd.DataFrame(df_rows)
    df.to_csv(OUT_DIR / "per_image.csv", index=False)

    # summary by dataset
    summary = {}
    for ds in ["kvasir", "cvc_clinicdb"]:
        sub = df[df["dataset"] == ds]
        summary[ds] = {
            "n": len(sub),
            "polyp_mean": float(sub["dice_polyp"].mean()),
            "polyp_median": float(sub["dice_polyp"].median()),
            "growth_mean": float(sub["dice_growth"].mean()),
            "growth_median": float(sub["dice_growth"].median()),
            "delta_mean": float(sub["delta"].mean()),
            "n_helped_big": int((sub["delta"] > 0.2).sum()),
            "n_hurt_big": int((sub["delta"] < -0.2).sum()),
            "n_unchanged": int(sub["delta"].abs().lt(0.05).sum()),
        }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
