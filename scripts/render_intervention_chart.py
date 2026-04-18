"""Render the final scope-crop intervention chart + writeup."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gutcheck import COLORS
from scripts._common import RESULTS


OUT_DIR = RESULTS / "failure_analysis" / "scope_crop"


def main():
    df = pd.read_csv(OUT_DIR / "per_image.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=140)
    fig.patch.set_facecolor("#0c0c0f")

    # Panel 1: paired bar chart (Dice full vs crop, per dataset)
    ax = axes[0]
    ax.set_facecolor("#0c0c0f")
    datasets = ["kvasir", "cvc_clinicdb"]
    labels = ["Kvasir-SEG", "CVC-ClinicDB"]
    dice_full = [df[df["dataset"] == d]["dice_full"].mean() for d in datasets]
    dice_crop = [df[df["dataset"] == d]["dice_crop"].mean() for d in datasets]
    x = np.arange(2)
    w = 0.35
    blue = tuple(c / 255 for c in COLORS["sam31_zs"])
    faded = tuple(min(1.0, c * 1.4) for c in blue)
    b1 = ax.bar(x - w / 2, dice_full, w, color=blue, edgecolor="white", linewidth=0.8, label="full frame")
    b2 = ax.bar(x + w / 2, dice_crop, w, color=faded, edgecolor="white", linewidth=0.8, label="scope-cropped")
    for rects, vals in ((b1, dice_full), (b2, dice_crop)):
        for r, v in zip(rects, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.012, f"{v:.3f}",
                    ha="center", va="bottom", color="white", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, color="white")
    ax.set_ylabel("Dice (mean)", color="white")
    ax.set_ylim(0, 1.02)
    ax.set_title("SAM 3.1 zero-shot: effect of cropping out scope vignette",
                 color="white", fontsize=11)
    ax.tick_params(colors="white")
    for s in ax.spines.values(): s.set_color("#444")
    ax.grid(True, axis="y", color="#2a2a2a", linewidth=0.4)
    ax.set_axisbelow(True)
    leg = ax.legend(loc="lower right", frameon=False, labelcolor="white")

    # Panel 2: per-image delta scatter (CVC only, catastrophic cases)
    ax = axes[1]
    ax.set_facecolor("#0c0c0f")
    for ds, color_key, marker in (("cvc_clinicdb", "sam31_zs", "o"), ("kvasir", "pranet", "^")):
        sub = df[df["dataset"] == ds]
        c = tuple(v / 255 for v in COLORS[color_key])
        ax.scatter(sub["dice_full"], sub["dice_crop"], s=26, c=[c], alpha=0.75,
                   marker=marker, label=ds.replace("_", "-"))
    ax.plot([0, 1], [0, 1], "--", color="#666", linewidth=0.8)
    ax.set_xlabel("Dice — full frame", color="white")
    ax.set_ylabel("Dice — scope-cropped", color="white")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Per-image effect (points above diagonal = crop helped)",
                 color="white", fontsize=11)
    ax.tick_params(colors="white")
    for s in ax.spines.values(): s.set_color("#444")
    ax.grid(True, color="#2a2a2a", linewidth=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, labelcolor="white")

    plt.tight_layout()
    out = RESULTS / "failure_analysis" / "scope_crop_summary.png"
    plt.savefig(out, facecolor=fig.get_facecolor())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
