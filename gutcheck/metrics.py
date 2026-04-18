from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

EPS = 1e-7


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred.astype(bool)
    g = gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    denom = p.sum() + g.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * inter / (denom + EPS))


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred.astype(bool)
    g = gt.astype(bool)
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0
    inter = np.logical_and(p, g).sum()
    return float(inter / (union + EPS))


@dataclass
class MetricAccumulator:
    approach: str
    dataset: str
    per_image_dice: list[float] = field(default_factory=list)
    per_image_iou: list[float] = field(default_factory=list)
    image_ids: list[str] = field(default_factory=list)
    fps_samples: list[float] = field(default_factory=list)

    def add(self, image_id: str, pred: np.ndarray, gt: np.ndarray, fps: float | None = None) -> None:
        self.image_ids.append(image_id)
        self.per_image_dice.append(dice(pred, gt))
        self.per_image_iou.append(iou(pred, gt))
        if fps is not None:
            self.fps_samples.append(fps)

    def summary(self) -> dict:
        return {
            "approach": self.approach,
            "dataset": self.dataset,
            "n": len(self.per_image_dice),
            "dice_mean": float(np.mean(self.per_image_dice)) if self.per_image_dice else 0.0,
            "dice_median": float(np.median(self.per_image_dice)) if self.per_image_dice else 0.0,
            "iou_mean": float(np.mean(self.per_image_iou)) if self.per_image_iou else 0.0,
            "iou_median": float(np.median(self.per_image_iou)) if self.per_image_iou else 0.0,
            "fps_mean": float(np.mean(self.fps_samples)) if self.fps_samples else None,
        }

    def save(self, dir_path: str | Path) -> None:
        d = Path(dir_path)
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(json.dumps(self.summary(), indent=2))
        per_image = [
            {"image_id": i, "dice": di, "iou": io}
            for i, di, io in zip(self.image_ids, self.per_image_dice, self.per_image_iou)
        ]
        (d / "per_image.json").write_text(json.dumps(per_image, indent=2))


def append_to_master(master_path: str | Path, summary: dict) -> None:
    p = Path(master_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        data = json.loads(p.read_text())
    else:
        data = []
    data = [r for r in data if not (r.get("approach") == summary["approach"] and r.get("dataset") == summary["dataset"])]
    data.append(summary)
    p.write_text(json.dumps(data, indent=2))
