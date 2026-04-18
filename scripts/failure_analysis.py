"""Why does SAM 3.1 catastrophically fail on some images?

For every test image across both datasets, compute:
  - ground-truth polyp area fraction
  - image brightness (mean V in HSV)
  - mean saturation
  - specular-reflection area fraction (bright, low-saturation pixels)
  - polyp centrality (distance of centroid from frame center)
  - polyp aspect ratio and solidity
  - frame blurriness (variance of Laplacian)
  - vignette/scope-border fraction (very dark edges)

Cross-reference with per-image Dice for every approach. Report:
  1. which features separate SAM 3.1 failures from successes
  2. whether those same images also break PraNet / DINOv3 (shared failures vs SAM-specific)
  3. a small failure gallery showing the top 12 worst SAM-3.1-specific cases
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gutcheck import APPROACHES, COLORS, LABELS
from gutcheck.viz import overlay_mask, save_png
from scripts._common import (
    RESULTS,
    KVASIR_ROOT,
    KVASIR_TEST_SPLIT_DIR,
    CVC_TEST_ROOT,
)

OUT_DIR = RESULTS / "failure_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Image features
# ---------------------------------------------------------------------------


def specular_fraction(rgb: np.ndarray) -> float:
    """Fraction of pixels that look like specular-reflection highlights:
    very bright AND very low saturation. Scope mucosa is reddish-pink, so
    white spots with high V and low S are almost always specular."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    v, s = hsv[..., 2], hsv[..., 1]
    spec = (v > 230) & (s < 40)
    return float(spec.sum()) / spec.size


def scope_border_fraction(rgb: np.ndarray) -> float:
    """Fraction of pixels on the image border that are very dark — proxy
    for the black cap of a colonoscope view (vignette)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    border = np.concatenate([
        gray[:10, :].ravel(),
        gray[-10:, :].ravel(),
        gray[:, :10].ravel(),
        gray[:, -10:].ravel(),
    ])
    return float((border < 20).sum()) / border.size


def mask_shape_stats(gt: np.ndarray) -> dict:
    h, w = gt.shape
    area_frac = float(gt.sum()) / gt.size

    if not gt.any():
        return {"area_frac": 0.0, "centrality": 0.0, "aspect": 0.0, "solidity": 0.0}

    ys, xs = np.where(gt)
    cy, cx = ys.mean(), xs.mean()
    centrality = float(np.hypot(cy - h / 2, cx - w / 2) / np.hypot(h / 2, w / 2))

    contours, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(biggest)
    aspect = max(bw, bh) / max(1, min(bw, bh))
    hull = cv2.convexHull(biggest)
    hull_area = max(1.0, cv2.contourArea(hull))
    solidity = float(cv2.contourArea(biggest) / hull_area)

    return {
        "area_frac": area_frac,
        "centrality": centrality,
        "aspect": aspect,
        "solidity": solidity,
    }


def image_features(rgb: np.ndarray, gt: np.ndarray) -> dict:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    feats = {
        "brightness": float(hsv[..., 2].mean()) / 255.0,
        "saturation": float(hsv[..., 1].mean()) / 255.0,
        "specular_frac": specular_fraction(rgb),
        "scope_border_frac": scope_border_fraction(rgb),
        "blur_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }
    feats.update(mask_shape_stats(gt))
    return feats


# ---------------------------------------------------------------------------
# Data plumbing
# ---------------------------------------------------------------------------


def load_samples():
    from gutcheck.data import KvasirSEG, CVCClinicDB
    kv = KvasirSEG(KVASIR_ROOT, split="val", held_out_ids_path=KVASIR_TEST_SPLIT_DIR)
    cvc = CVCClinicDB(CVC_TEST_ROOT)
    return [(s, "kvasir") for s in kv] + [(s, "cvc_clinicdb") for s in cvc]


def load_per_image() -> Dict[tuple[str, str], Dict[str, float]]:
    """Returns {(approach, dataset, image_id): dice}."""
    out: Dict = {}
    for approach in APPROACHES + ["sam_zs"]:
        for ds in ("kvasir", "cvc_clinicdb"):
            p = RESULTS / "metrics" / approach / ds / "per_image.json"
            if not p.exists():
                continue
            for r in json.loads(p.read_text()):
                out[(approach, ds, r["image_id"])] = float(r["dice"])
    return out


def load_pred(approach: str, ds: str, image_id: str) -> np.ndarray | None:
    p = RESULTS / "preds" / approach / ds / f"{image_id}.png"
    if not p.exists():
        return None
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return None if m is None else (m > 127)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def build_table() -> pd.DataFrame:
    samples = load_samples()
    per_image = load_per_image()
    rows = []
    for s, ds in samples:
        feats = image_features(s.image, s.mask)
        row = {"image_id": s.image_id, "dataset": ds, "source": s.source, **feats}
        for approach in APPROACHES + ["sam_zs"]:
            row[f"dice_{approach}"] = per_image.get((approach, ds, s.image_id), np.nan)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def categorize(df: pd.DataFrame, approach: str) -> pd.Series:
    d = df[f"dice_{approach}"]
    out = pd.Series("ok", index=df.index)
    out[d < 0.2] = "catastrophic"
    out[(d >= 0.2) & (d < 0.6)] = "weak"
    out[d.isna()] = "missing"
    return out


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    features = [
        "area_frac", "centrality", "aspect", "solidity",
        "brightness", "saturation", "specular_frac",
        "scope_border_frac", "blur_var",
    ]
    rows = []
    for approach in APPROACHES + ["sam_zs"]:
        col = f"dice_{approach}"
        if col not in df:
            continue
        sub = df.dropna(subset=[col])
        if len(sub) < 10:
            continue
        for f in features:
            corr = sub[f].corr(sub[col])
            rows.append({"approach": approach, "feature": f, "pearson_r": corr, "n": len(sub)})
    return pd.DataFrame(rows)


def split_stats(df: pd.DataFrame, approach: str) -> pd.DataFrame:
    col = f"dice_{approach}"
    df = df.dropna(subset=[col])
    cat = categorize(df, approach)
    out = []
    for label in ["catastrophic", "weak", "ok"]:
        sub = df[cat == label]
        if len(sub) == 0:
            continue
        out.append({
            "bucket": label,
            "n": len(sub),
            "area_frac": sub["area_frac"].mean(),
            "specular_frac": sub["specular_frac"].mean(),
            "scope_border_frac": sub["scope_border_frac"].mean(),
            "blur_var": sub["blur_var"].mean(),
            "centrality": sub["centrality"].mean(),
            "brightness": sub["brightness"].mean(),
        })
    return pd.DataFrame(out)


def sam31_specific_failures(df: pd.DataFrame, top_k: int = 12) -> pd.DataFrame:
    """Catastrophic for SAM 3.1, but PraNet handles it fine.
    Ranking = PraNet_dice - SAM31_dice (higher = more SAM-specific)."""
    if "dice_sam31_zs" not in df or "dice_pranet" not in df:
        return df.iloc[:0]
    m = df.dropna(subset=["dice_sam31_zs", "dice_pranet"]).copy()
    m["sam31_gap"] = m["dice_pranet"] - m["dice_sam31_zs"]
    m = m[m["dice_sam31_zs"] < 0.2].sort_values("sam31_gap", ascending=False)
    return m.head(top_k)


def render_failure_gallery(df_failures: pd.DataFrame, all_samples):
    id_to_sample = {(s.image_id, ds): s for s, ds in all_samples}
    cells = []
    for _, row in df_failures.iterrows():
        key = (row["image_id"], row["dataset"])
        if key not in id_to_sample:
            continue
        s = id_to_sample[key]
        preds = {a: load_pred(a, row["dataset"], s.image_id) for a in APPROACHES}
        preds = {k: v for k, v in preds.items() if v is not None}
        # 1 cell = ground truth | sam31 | pranet (only the 3 relevant panels)
        gt_panel = overlay_mask(s.image, s.mask, "ground_truth", fill_alpha=0.0, dashed=True)
        sam_panel = overlay_mask(s.image, preds.get("sam31_zs", np.zeros_like(s.mask)), "sam31_zs")
        pra_panel = overlay_mask(s.image, preds.get("pranet", np.zeros_like(s.mask)), "pranet")

        h, w = s.image.shape[:2]
        gap = 4
        row_img = np.full((h, 3 * w + 2 * gap, 3), 12, dtype=np.uint8)
        row_img[:, :w] = gt_panel
        row_img[:, w + gap:2 * w + gap] = sam_panel
        row_img[:, 2 * w + 2 * gap:] = pra_panel

        # caption bar
        cap_h = 24
        cap = np.full((cap_h, row_img.shape[1], 3), 20, dtype=np.uint8)
        cap_bgr = cv2.cvtColor(cap, cv2.COLOR_RGB2BGR)
        txt = f"{row['dataset']} / {row['image_id'][:20]}  SAM 3.1={row['dice_sam31_zs']:.2f}  PraNet={row['dice_pranet']:.2f}  polyp-area={row['area_frac']*100:.1f}%  specular={row['specular_frac']*100:.1f}%"
        cv2.putText(cap_bgr, txt, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
        cap = cv2.cvtColor(cap_bgr, cv2.COLOR_BGR2RGB)

        cells.append(np.concatenate([cap, row_img], axis=0))

    if not cells:
        return
    # resize every cell to same width
    target_w = max(c.shape[1] for c in cells)
    resized = []
    for c in cells:
        if c.shape[1] != target_w:
            scale = target_w / c.shape[1]
            c = cv2.resize(c, (target_w, int(c.shape[0] * scale)))
        resized.append(c)
    gallery = np.concatenate(resized, axis=0)
    save_png(OUT_DIR / "sam31_specific_failure_gallery.png", gallery)


def render_feature_dist_plot(df: pd.DataFrame):
    features = [("area_frac", "polyp area (fraction of image)", 0, 0.6),
                ("specular_frac", "specular-highlight area", 0, 0.12),
                ("scope_border_frac", "scope-vignette border", 0, 1.0),
                ("blur_var", "sharpness (Laplacian var)", 0, 600)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=140)
    fig.patch.set_facecolor("#0c0c0f")
    for ax, (feat, label, lo, hi) in zip(axes.ravel(), features):
        ax.set_facecolor("#0c0c0f")
        for approach, color_key in [("sam31_zs", "sam31_zs"), ("pranet", "pranet")]:
            col = f"dice_{approach}"
            sub = df.dropna(subset=[col, feat])
            cat = categorize(sub, approach)
            for bucket, marker, alpha in [("ok", "o", 0.35), ("catastrophic", "X", 0.95)]:
                pick = sub[cat == bucket]
                if len(pick) == 0:
                    continue
                c = tuple(v / 255 for v in COLORS[color_key])
                ax.scatter(pick[feat], pick[col], s=(38 if bucket == "catastrophic" else 14),
                           c=[c], marker=marker, alpha=alpha,
                           label=f"{LABELS[color_key]}  {bucket}")
        ax.set_xlabel(label, color="white", fontsize=9)
        ax.set_ylabel("Dice (this approach)", color="white", fontsize=9)
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, 1.02)
        ax.tick_params(colors="white", labelsize=8)
        for s in ax.spines.values():
            s.set_color("#444")
        ax.grid(True, color="#2a2a2a", linewidth=0.4)
        ax.set_axisbelow(True)
    axes[0, 0].legend(fontsize=7, loc="lower right", framealpha=0.3, labelcolor="white")
    fig.suptitle("Per-image features vs Dice — SAM 3.1 failure modes",
                 color="white", fontsize=12, y=0.99)
    plt.tight_layout()
    out = OUT_DIR / "feature_vs_dice.png"
    plt.savefig(out, facecolor=fig.get_facecolor())
    print(f"wrote {out}")


def render_overlap_chart(df: pd.DataFrame):
    """How many SAM-3.1 catastrophic failures are ALSO catastrophic for the other approaches?"""
    rows = []
    sam_fail = df[df["dice_sam31_zs"] < 0.2].dropna(subset=["dice_sam31_zs"])
    for other in ["sam_zs", "sam_lora", "pranet", "dinov3"]:
        col = f"dice_{other}"
        if col not in df:
            continue
        sub = sam_fail.dropna(subset=[col])
        shared = (sub[col] < 0.2).sum()
        rows.append({"other": other, "shared_catastrophic": int(shared), "sam31_failures": len(sub)})
    df_rows = pd.DataFrame(rows)
    df_rows.to_csv(OUT_DIR / "failure_overlap.csv", index=False)
    print(df_rows.to_string(index=False))


def main():
    print("computing per-image features...")
    df = build_table()
    df.to_csv(OUT_DIR / "per_image_features.csv", index=False)

    print("\ncorrelation of features with Dice:")
    corr = correlation_table(df)
    corr_pivot = corr.pivot(index="feature", columns="approach", values="pearson_r")
    print(corr_pivot.round(3).to_string())
    corr.to_csv(OUT_DIR / "feature_dice_correlations.csv", index=False)

    print("\nSAM 3.1 per-bucket feature means:")
    print(split_stats(df, "sam31_zs").round(3).to_string(index=False))

    print("\nPraNet per-bucket feature means:")
    print(split_stats(df, "pranet").round(3).to_string(index=False))

    print("\ncatastrophic-failure overlap (SAM 3.1 vs others):")
    render_overlap_chart(df)

    print("\nSAM-3.1-specific failure gallery...")
    failures = sam31_specific_failures(df, top_k=12)
    failures.to_csv(OUT_DIR / "sam31_specific_failures.csv", index=False)
    all_samples = load_samples()
    render_failure_gallery(failures, all_samples)

    print("\nfeature scatter plot...")
    render_feature_dist_plot(df)

    print(f"\nall outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
