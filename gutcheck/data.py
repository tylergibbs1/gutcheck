"""Dataset loaders for Kvasir-SEG and CVC-ClinicDB.

Each loader yields (image_rgb_uint8_HxWx3, mask_bool_HxW, image_id_str).
Resizing and model-specific preprocessing is the caller's job.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class Sample:
    image: np.ndarray
    mask: np.ndarray
    image_id: str
    source: str


def _read_image(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _read_mask(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return m > 127


def _deterministic_split(ids: list[str], val_frac: float, seed: int) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    ids_sorted = sorted(ids)
    perm = rng.permutation(len(ids_sorted))
    n_val = int(round(len(ids_sorted) * val_frac))
    val_set = {ids_sorted[i] for i in perm[:n_val]}
    train = [i for i in ids_sorted if i not in val_set]
    val = [i for i in ids_sorted if i in val_set]
    return train, val


class KvasirSEG:
    """Kvasir-SEG. Directory layout expected:
        <root>/images/*.jpg
        <root>/masks/*.jpg

    Splits use the canonical PraNet/Polyp-PVT test split of 100 images
    (file IDs provided via `held_out_ids_path`). The remaining 900
    images form the training set. `split="all"` returns all 1000 images.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        held_out_ids_path: str | Path | None = None,
    ):
        self.root = Path(root)
        self.split = split
        img_dir = self.root / "images"
        msk_dir = self.root / "masks"
        if not img_dir.is_dir():
            raise FileNotFoundError(f"Missing {img_dir}")
        ids = [p.stem for p in img_dir.glob("*.jpg")]
        if len(ids) == 0:
            raise RuntimeError(f"No images found in {img_dir}")

        held_out: set[str] = set()
        if held_out_ids_path is not None and Path(held_out_ids_path).exists():
            held_out_dir = Path(held_out_ids_path)
            if held_out_dir.is_dir():
                held_out = {p.stem for p in held_out_dir.glob("*.png")} | {
                    p.stem for p in held_out_dir.glob("*.jpg")
                }
            else:
                held_out = set(held_out_dir.read_text().split())

        if split == "train":
            self.ids = sorted(i for i in ids if i not in held_out)
        elif split == "val":
            self.ids = sorted(i for i in ids if i in held_out)
        elif split == "all":
            self.ids = sorted(ids)
        else:
            raise ValueError(split)
        self.img_dir = img_dir
        self.msk_dir = msk_dir

    def __len__(self) -> int:
        return len(self.ids)

    def __iter__(self) -> Iterator[Sample]:
        for i in self.ids:
            img = _read_image(self.img_dir / f"{i}.jpg")
            msk = _read_mask(self.msk_dir / f"{i}.jpg")
            yield Sample(img, msk, i, "kvasir_seg")

    def get(self, idx: int) -> Sample:
        i = self.ids[idx]
        img = _read_image(self.img_dir / f"{i}.jpg")
        msk = _read_mask(self.msk_dir / f"{i}.jpg")
        return Sample(img, msk, i, "kvasir_seg")


class CVCClinicDB:
    """CVC-ClinicDB (canonical 62-image test split from PraNet/Polyp-PVT).
    Common directory layouts:
        <root>/Original/*.png + <root>/Ground Truth/*.png      (original release)
        <root>/images/*.png  + <root>/masks/*.png               (reprocessed)
        <root>/PNG/Original/*.png + <root>/PNG/Ground Truth/*.png
    We auto-detect.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        img_dir, msk_dir, ext = self._detect(self.root)
        ids = sorted({p.stem for p in img_dir.glob(f"*.{ext}")})
        if not ids:
            raise RuntimeError(f"No images in {img_dir}")
        self.ids = ids
        self.img_dir = img_dir
        self.msk_dir = msk_dir
        self.ext = ext

    @staticmethod
    def _detect(root: Path) -> tuple[Path, Path, str]:
        candidates = [
            (root / "Original", root / "Ground Truth"),
            (root / "images", root / "masks"),
            (root / "PNG" / "Original", root / "PNG" / "Ground Truth"),
        ]
        for img_dir, msk_dir in candidates:
            if img_dir.is_dir() and msk_dir.is_dir():
                for ext in ("png", "tif", "jpg"):
                    if any(img_dir.glob(f"*.{ext}")):
                        return img_dir, msk_dir, ext
        raise FileNotFoundError(f"Could not find image/mask dirs under {root}")

    def __len__(self) -> int:
        return len(self.ids)

    def __iter__(self) -> Iterator[Sample]:
        for i in self.ids:
            img = _read_image(self.img_dir / f"{i}.{self.ext}")
            msk = _read_mask(self.msk_dir / f"{i}.{self.ext}")
            yield Sample(img, msk, i, "cvc_clinicdb")

    def get(self, idx: int) -> Sample:
        i = self.ids[idx]
        img = _read_image(self.img_dir / f"{i}.{self.ext}")
        msk = _read_mask(self.msk_dir / f"{i}.{self.ext}")
        return Sample(img, msk, i, "cvc_clinicdb")


def resize_pair(img: np.ndarray, mask: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    img_r = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    msk_r = cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST) > 0
    return img_r, msk_r


def save_split_manifest(dataset, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"source": getattr(dataset, "source", None), "ids": dataset.ids}
    p.write_text(json.dumps(payload, indent=2))
