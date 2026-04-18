"""Render 2x3 comparison grids (GT + 4 approaches) for a handful of test images.

Picks examples to show: 3 strong cases, 3 hard cases, 3 where approaches disagree.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gutcheck import APPROACHES
from gutcheck.data import KvasirSEG, CVCClinicDB
from gutcheck.metrics import dice
from gutcheck.viz import comparison_grid, save_png
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


def pick_examples(n_strong: int, n_hard: int, n_disagree: int, dataset_name: str, samples: list):
    """Pick representative examples based on per-image Dice per approach."""
    # load per-image metrics
    per_image = {}  # approach -> id -> dice
    for approach in APPROACHES:
        fp = RESULTS / "metrics" / approach / dataset_name / "per_image.json"
        if not fp.exists():
            continue
        entries = json.loads(fp.read_text())
        per_image[approach] = {e["image_id"]: e["dice"] for e in entries}

    # Align across approaches
    ids = set(per_image[next(iter(per_image))].keys()) if per_image else set()
    for d in per_image.values():
        ids &= set(d.keys())

    rows = []
    for i in ids:
        dices = [per_image[a][i] for a in per_image]
        rows.append((i, np.mean(dices), np.std(dices)))

    rows.sort(key=lambda r: r[1], reverse=True)
    strong = [r[0] for r in rows[:n_strong]]
    hard = [r[0] for r in sorted(rows, key=lambda r: r[1])[:n_hard]]
    disagree = [r[0] for r in sorted(rows, key=lambda r: r[2], reverse=True)[:n_disagree]]
    return strong, hard, disagree


def render_all(dataset, dataset_name: str, out_dir: Path):
    samples = list(dataset)
    strong, hard, disagree = pick_examples(3, 3, 3, dataset_name, samples)

    id_to_sample = {s.image_id: s for s in samples}

    for label, ids in [("strong", strong), ("hard", hard), ("disagree", disagree)]:
        for i in ids:
            s = id_to_sample[i]
            preds = {}
            for a in APPROACHES:
                p = load_pred(a, dataset_name, s.image_id)
                if p is not None:
                    preds[a] = p
            if not preds:
                continue
            grid = comparison_grid(s.image, preds, gt=s.mask)
            save_png(out_dir / f"{label}_{dataset_name}_{s.image_id}.png", grid)


def main():
    out_dir = RESULTS / "overlays" / "comparison_grids"
    out_dir.mkdir(parents=True, exist_ok=True)

    kvasir = KvasirSEG(KVASIR_ROOT, split="val", held_out_ids_path=KVASIR_TEST_SPLIT_DIR)
    cvc = CVCClinicDB(CVC_TEST_ROOT)

    render_all(kvasir, "kvasir", out_dir)
    render_all(cvc, "cvc_clinicdb", out_dir)
    print(f"wrote grids to {out_dir}")


if __name__ == "__main__":
    main()
