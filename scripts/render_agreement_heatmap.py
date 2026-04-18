"""Render agreement heatmaps: how many approaches agree a pixel is polyp."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gutcheck import APPROACHES
from gutcheck.data import KvasirSEG, CVCClinicDB
from gutcheck.viz import agreement_heatmap, save_png
from scripts._common import (
    RESULTS,
    KVASIR_ROOT,
    KVASIR_TEST_SPLIT_DIR,
    CVC_TEST_ROOT,
)


def load_pred(approach: str, dataset_name: str, image_id: str) -> np.ndarray | None:
    p = RESULTS / "preds" / approach / dataset_name / f"{image_id}.png"
    if not p.exists():
        return None
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return m > 127


def main():
    out_dir = RESULTS / "overlays" / "agreement"
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset, ds_name in (
        (KvasirSEG(KVASIR_ROOT, split="val", held_out_ids_path=KVASIR_TEST_SPLIT_DIR), "kvasir"),
        (CVCClinicDB(CVC_TEST_ROOT), "cvc_clinicdb"),
    ):
        # render all images so the video editor has choices
        for sample in dataset:
            preds = {}
            for a in APPROACHES:
                p = load_pred(a, ds_name, sample.image_id)
                if p is not None:
                    preds[a] = p
            if len(preds) < 2:
                continue
            heat = agreement_heatmap(sample.image, preds)
            save_png(out_dir / f"{ds_name}_{sample.image_id}.png", heat)

    print(f"wrote heatmaps to {out_dir}")


if __name__ == "__main__":
    main()
