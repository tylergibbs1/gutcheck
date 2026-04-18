"""Zero-shot SAM 3 on Kvasir-SEG test split and CVC-ClinicDB.
Logs Dice/IoU/FPS, saves binary predictions as PNG for later viz."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gutcheck.data import KvasirSEG, CVCClinicDB
from gutcheck.metrics import MetricAccumulator, append_to_master
from gutcheck.models.sam_wrapper import Sam3Wrapper
from scripts._common import (
    SAM3_DIR,
    KVASIR_ROOT,
    CVC_TEST_ROOT,
    KVASIR_TEST_SPLIT_DIR,
    RESULTS,
    METRICS_MASTER,
)

APPROACH = "sam_zs"


def save_pred(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask.astype(np.uint8)) * 255)


def run(dataset, ds_name: str, wrapper: Sam3Wrapper) -> dict:
    acc = MetricAccumulator(APPROACH, ds_name)
    pred_dir = RESULTS / "preds" / APPROACH / ds_name
    for sample in tqdm(dataset, desc=f"{APPROACH}/{ds_name}", total=len(dataset)):
        t0 = time.perf_counter()
        mask = wrapper.predict(sample.image)
        dt = time.perf_counter() - t0
        acc.add(sample.image_id, mask, sample.mask, fps=1.0 / max(dt, 1e-6))
        save_pred(pred_dir / f"{sample.image_id}.png", mask)
    out_dir = RESULTS / "metrics" / APPROACH / ds_name
    acc.save(out_dir)
    print(f"[{APPROACH}/{ds_name}] ", acc.summary())
    return acc.summary()


def main() -> None:
    wrapper = Sam3Wrapper(SAM3_DIR)

    kvasir = KvasirSEG(KVASIR_ROOT, split="val", held_out_ids_path=KVASIR_TEST_SPLIT_DIR)
    cvc = CVCClinicDB(CVC_TEST_ROOT)

    summaries = []
    summaries.append(run(kvasir, "kvasir", wrapper))
    summaries.append(run(cvc, "cvc_clinicdb", wrapper))
    for s in summaries:
        append_to_master(METRICS_MASTER, s)


if __name__ == "__main__":
    main()
