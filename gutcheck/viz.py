"""Overlay rendering, comparison grids, agreement heatmaps, summary charts."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np

from . import COLORS, LABELS

FILL_ALPHA = 0.40
OUTLINE_THICKNESS = 2
GRID_GUTTER = 8
LABEL_HEIGHT = 26


def _to_bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return (rgb[2], rgb[1], rgb[0])


def overlay_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    approach: str,
    fill_alpha: float = FILL_ALPHA,
    outline: bool = True,
    dashed: bool = False,
) -> np.ndarray:
    """Return a new RGB uint8 image with mask overlaid in the approach's color."""
    assert image_rgb.dtype == np.uint8 and image_rgb.shape[2] == 3
    h, w = image_rgb.shape[:2]
    mask_bool = mask.astype(bool)
    if mask_bool.shape != (h, w):
        mask_bool = cv2.resize(mask_bool.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0

    color_rgb = COLORS.get(approach, (255, 255, 255))
    out = image_rgb.copy()

    if fill_alpha > 0 and mask_bool.any():
        color_layer = np.zeros_like(out)
        color_layer[:] = color_rgb
        alpha = np.where(mask_bool[..., None], fill_alpha, 0.0).astype(np.float32)
        out = (out.astype(np.float32) * (1 - alpha) + color_layer.astype(np.float32) * alpha).astype(np.uint8)

    if outline and mask_bool.any():
        contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bgr = _to_bgr(color_rgb)
        out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        if dashed:
            for contour in contours:
                pts = contour[:, 0, :]
                for k in range(0, len(pts), 10):
                    a = tuple(pts[k])
                    b = tuple(pts[min(k + 5, len(pts) - 1)])
                    cv2.line(out_bgr, a, b, bgr, OUTLINE_THICKNESS, lineType=cv2.LINE_AA)
        else:
            cv2.drawContours(out_bgr, contours, -1, bgr, OUTLINE_THICKNESS, lineType=cv2.LINE_AA)
        out = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

    return out


def _label_strip(width: int, text: str, color_rgb: tuple[int, int, int]) -> np.ndarray:
    strip = np.full((LABEL_HEIGHT, width, 3), 20, dtype=np.uint8)
    swatch_w = 18
    strip[:, 8 : 8 + swatch_w] = np.array(color_rgb, dtype=np.uint8)
    bgr = cv2.cvtColor(strip, cv2.COLOR_RGB2BGR)
    cv2.putText(bgr, text, (8 + swatch_w + 8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def comparison_grid(
    image_rgb: np.ndarray,
    preds: Mapping[str, np.ndarray],
    gt: np.ndarray | None = None,
    show_gt_panel: bool = True,
    ordering: Sequence[str] = ("sam_zs", "sam_lora", "pranet", "dinov3"),
) -> np.ndarray:
    panels = []
    labels = []
    colors = []
    if show_gt_panel and gt is not None:
        panels.append(overlay_mask(image_rgb, gt, "ground_truth", fill_alpha=0.0, dashed=True))
        labels.append(LABELS["ground_truth"])
        colors.append(COLORS["ground_truth"])
    for approach in ordering:
        if approach not in preds:
            continue
        panels.append(overlay_mask(image_rgb, preds[approach], approach))
        labels.append(LABELS.get(approach, approach))
        colors.append(COLORS.get(approach, (200, 200, 200)))

    h, w = image_rgb.shape[:2]
    n = len(panels)
    cols = 2 if n <= 4 else 3
    rows = (n + cols - 1) // cols

    cell_h = h + LABEL_HEIGHT
    cell_w = w
    grid_h = rows * cell_h + (rows - 1) * GRID_GUTTER
    grid_w = cols * cell_w + (cols - 1) * GRID_GUTTER
    grid = np.full((grid_h, grid_w, 3), 12, dtype=np.uint8)

    for i, (panel, lbl, clr) in enumerate(zip(panels, labels, colors)):
        r, c = divmod(i, cols)
        y0 = r * (cell_h + GRID_GUTTER)
        x0 = c * (cell_w + GRID_GUTTER)
        grid[y0 : y0 + LABEL_HEIGHT, x0 : x0 + cell_w] = _label_strip(cell_w, lbl, clr)
        grid[y0 + LABEL_HEIGHT : y0 + cell_h, x0 : x0 + cell_w] = panel
    return grid


def agreement_heatmap(
    image_rgb: np.ndarray,
    preds: Mapping[str, np.ndarray],
    weight_by_approach: Mapping[str, float] | None = None,
) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    stack = np.zeros((h, w), dtype=np.float32)
    count = 0
    for approach, pred in preds.items():
        m = pred.astype(bool)
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        wt = 1.0 if weight_by_approach is None else float(weight_by_approach.get(approach, 1.0))
        stack += m.astype(np.float32) * wt
        count += wt
    if count > 0:
        norm = (stack / count * 255).clip(0, 255).astype(np.uint8)
    else:
        norm = np.zeros((h, w), dtype=np.uint8)
    heat_bgr = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    alpha = (norm.astype(np.float32) / 255 * 0.75)[..., None]
    out = (image_rgb.astype(np.float32) * (1 - alpha) + heat_rgb.astype(np.float32) * alpha).astype(np.uint8)
    return out


def save_png(path: str | Path, rgb: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(p), bgr)
