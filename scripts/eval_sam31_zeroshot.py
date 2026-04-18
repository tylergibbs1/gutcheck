"""Zero-shot SAM 3.1 (facebookresearch/sam3 repo + `sam3.1_multiplex.pt`).

Runs in the Python 3.12 sam3 venv. Uses the image model wrapper with the
SAM 3.1 multiplex checkpoint (4 FPN conv weights fall back to random init
as of the SAM 3.1 March 2026 release — the rest is genuine 3.1). For each
image we pick the top-scoring proposal mask, since presence scores collapse
for medical text prompts outside SAM's open-vocabulary training set.
"""

from __future__ import annotations

import argparse
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
METRICS_MASTER = RESULTS / "metrics.json"

APPROACH = "sam31_zs"
APPROACH_LABEL = "SAM 3.1 zero-shot"

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    denom = p.sum() + g.sum()
    return 1.0 if denom == 0 else float(2.0 * inter / (denom + 1e-7))


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    union = np.logical_or(p, g).sum()
    return 1.0 if union == 0 else float(np.logical_and(p, g).sum() / (union + 1e-7))


def load_samples_kvasir():
    images = sorted(p for p in (DATA / "TestDataset" / "Kvasir" / "images").glob("*.png"))
    out = []
    for p in images:
        img = np.asarray(Image.open(p).convert("RGB"))
        m = cv2.imread(str(DATA / "TestDataset" / "Kvasir" / "masks" / p.name), 0) > 127
        out.append((p.stem, img, m))
    return out, "kvasir"


def load_samples_cvc():
    base = DATA / "CVC-ClinicDB"
    images = sorted(p for p in (base / "images").glob("*.png"))
    out = []
    for p in images:
        img = np.asarray(Image.open(p).convert("RGB"))
        m = cv2.imread(str(base / "masks" / p.name), 0) > 127
        out.append((p.stem, img, m))
    return out, "cvc_clinicdb"


def run(proc, samples, ds_name: str) -> dict:
    per_dice, per_iou, ids, fps = [], [], [], []
    pred_dir = RESULTS / "preds" / APPROACH / ds_name
    pred_dir.mkdir(parents=True, exist_ok=True)

    for image_id, rgb, gt in tqdm(samples, desc=f"{APPROACH}/{ds_name}"):
        pil = Image.fromarray(rgb)
        t0 = time.perf_counter()
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            state = proc.set_image(pil)
            out = proc.set_text_prompt(state=state, prompt="polyp")
        dt = time.perf_counter() - t0

        masks = out["masks"].cpu().numpy()
        if masks.ndim == 4:
            masks = masks[:, 0]
        scores = out["scores"].float().cpu().numpy()

        if len(masks) == 0:
            pred = np.zeros_like(gt, dtype=bool)
        else:
            top = int(scores.argmax())
            pred = masks[top].astype(bool)
            if pred.shape != gt.shape:
                pred = cv2.resize(pred.astype(np.uint8), (gt.shape[1], gt.shape[0]),
                                  interpolation=cv2.INTER_NEAREST) > 0

        per_dice.append(dice(pred, gt))
        per_iou.append(iou(pred, gt))
        ids.append(image_id)
        fps.append(1.0 / max(dt, 1e-6))

        cv2.imwrite(str(pred_dir / f"{image_id}.png"), pred.astype(np.uint8) * 255)

    summary = {
        "approach": APPROACH,
        "dataset": ds_name,
        "n": len(per_dice),
        "dice_mean": float(np.mean(per_dice)),
        "dice_median": float(np.median(per_dice)),
        "iou_mean": float(np.mean(per_iou)),
        "iou_median": float(np.median(per_iou)),
        "fps_mean": float(np.mean(fps)),
    }
    metrics_dir = RESULTS / "metrics" / APPROACH / ds_name
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (metrics_dir / "per_image.json").write_text(json.dumps(
        [{"image_id": i, "dice": d, "iou": u} for i, d, u in zip(ids, per_dice, per_iou)], indent=2))
    print(f"[{APPROACH}/{ds_name}]", summary)
    return summary


def append_to_master(summary: dict):
    if METRICS_MASTER.exists():
        data = json.loads(METRICS_MASTER.read_text())
    else:
        data = []
    data = [r for r in data if not (r.get("approach") == summary["approach"] and r.get("dataset") == summary["dataset"])]
    data.append(summary)
    METRICS_MASTER.write_text(json.dumps(data, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="processor confidence threshold (0 = keep all 200 proposals, pick top by score)")
    args = ap.parse_args()

    print("Building SAM 3.1 image model with multiplex checkpoint...")
    model = build_sam3_image_model(checkpoint_path=str(CKPT), load_from_HF=False)
    proc = Sam3Processor(model, confidence_threshold=args.threshold)

    summaries = {}
    for loader in (load_samples_kvasir, load_samples_cvc):
        samples, name = loader()
        summary = run(proc, samples, name)
        append_to_master(summary)
        summaries[name] = summary
    k = summaries.get("kvasir", {}).get("dice_mean", 0.0)
    c = summaries.get("cvc_clinicdb", {}).get("dice_mean", 0.0)
    print(f"METRICS kvasir_dice={k:.6f} cvc_dice={c:.6f} combined={(k + c) / 2:.6f}")


if __name__ == "__main__":
    main()
