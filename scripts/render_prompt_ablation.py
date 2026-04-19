"""Render the two panels for the prompt-ablation finding:
  1. Per-image scatter: polyp vs growth Dice (points above diagonal = growth helped)
  2. Updated headline bar chart including both SAM 3.1 prompts"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gutcheck import COLORS, LABELS
from scripts._common import RESULTS, METRICS_MASTER


def render_scatter():
    df = pd.read_csv(RESULTS / "prompt_ablation" / "per_image.csv")
    summary = json.loads((RESULTS / "prompt_ablation" / "summary.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), dpi=140)
    fig.patch.set_facecolor("#0c0c0f")

    # left: paired bar chart
    ax = axes[0]
    ax.set_facecolor("#0c0c0f")
    ds_labels = ["Kvasir-SEG", "CVC-ClinicDB"]
    polyp_means = [summary["kvasir"]["polyp_mean"], summary["cvc_clinicdb"]["polyp_mean"]]
    growth_means = [summary["kvasir"]["growth_mean"], summary["cvc_clinicdb"]["growth_mean"]]
    x = np.arange(2)
    w = 0.38
    col_polyp = tuple(c / 255 for c in COLORS["sam31_zs"])
    col_growth = tuple(min(1.0, c * 1.5 / 255) for c in COLORS["sam31_zs"])  # lighter variant
    b1 = ax.bar(x - w / 2, polyp_means, w, color=col_polyp, edgecolor="white", linewidth=0.8, label='prompt = "polyp"')
    b2 = ax.bar(x + w / 2, growth_means, w, color=col_growth, edgecolor="white", linewidth=0.8, label='prompt = "growth"')
    for rects, vals in ((b1, polyp_means), (b2, growth_means)):
        for r, v in zip(rects, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.012, f"{v:.3f}",
                    ha="center", va="bottom", color="white", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(ds_labels, color="white")
    ax.set_ylabel("Dice (mean)", color="white")
    ax.set_ylim(0, 1.02)
    ax.set_title('SAM 3.1 zero-shot: prompt "polyp" vs "growth"',
                 color="white", fontsize=11)
    ax.tick_params(colors="white")
    for s in ax.spines.values(): s.set_color("#444")
    ax.grid(True, axis="y", color="#2a2a2a", linewidth=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, labelcolor="white")

    # right: per-image scatter
    ax = axes[1]
    ax.set_facecolor("#0c0c0f")
    for ds, marker, c in (("kvasir", "^", "pranet"), ("cvc_clinicdb", "o", "sam31_zs")):
        sub = df[df["dataset"] == ds]
        colr = tuple(v / 255 for v in COLORS[c])
        ax.scatter(sub["dice_polyp"], sub["dice_growth"], s=28, c=[colr], alpha=0.75,
                   marker=marker, label=ds.replace("_", "-"))
    ax.plot([0, 1], [0, 1], "--", color="#666", linewidth=0.8)
    ax.set_xlabel('Dice — prompt "polyp"', color="white")
    ax.set_ylabel('Dice — prompt "growth"', color="white")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Per-image effect (above diagonal = growth wins)",
                 color="white", fontsize=11)
    ax.tick_params(colors="white")
    for s in ax.spines.values(): s.set_color("#444")
    ax.grid(True, color="#2a2a2a", linewidth=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, labelcolor="white")

    plt.tight_layout()
    out = RESULTS / "prompt_ablation" / "ablation_summary.png"
    plt.savefig(out, facecolor=fig.get_facecolor())
    print(f"wrote {out}")


def render_headline():
    """Redo summary.png with a SAM 3.1 'growth' bar added alongside 'polyp'."""
    summary = json.loads((RESULTS / "prompt_ablation" / "summary.json").read_text())
    master = json.loads(METRICS_MASTER.read_text())
    by_key = {(r["approach"], r["dataset"]): r for r in master}

    approaches = ["sam31_zs", "sam31_growth", "sam_lora", "pranet", "dinov3"]
    labels = {
        "sam31_zs": 'SAM 3.1 zs ("polyp")',
        "sam31_growth": 'SAM 3.1 zs ("growth")',
        "sam_lora": "SAM 3 + LoRA",
        "pranet": "PraNet (2020)",
        "dinov3": "DINOv3 + decoder",
    }
    colors = {
        "sam31_zs": (60, 100, 200),
        "sam31_growth": (0, 180, 255),
        "sam_lora": COLORS["sam_lora"],
        "pranet": COLORS["pranet"],
        "dinov3": COLORS["dinov3"],
    }

    dataset_order = ["kvasir", "cvc_clinicdb"]
    dataset_labels = ["Kvasir-SEG test (in-distribution)", "CVC-ClinicDB (cross-dataset)"]

    fig, ax = plt.subplots(figsize=(11.5, 5.5), dpi=140)
    fig.patch.set_facecolor("#0c0c0f")
    ax.set_facecolor("#0c0c0f")

    x = np.arange(len(approaches))
    w = 0.38
    for i, ds in enumerate(dataset_order):
        vals = []
        for a in approaches:
            if a == "sam31_growth":
                vals.append(summary[ds]["growth_mean"])
            elif a == "sam31_zs":
                vals.append(summary[ds]["polyp_mean"])
            else:
                r = by_key.get((a, ds))
                vals.append(r["dice_mean"] if r else 0.0)
        offs = (i - 0.5) * w
        bars = ax.bar(
            x + offs, vals, w,
            color=[tuple(v / 255 for v in colors[a]) for a in approaches],
            edgecolor="white", linewidth=0.8,
            alpha=0.55 if i == 0 else 1.0,
            label=dataset_labels[i],
        )
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.012,
                    f"{v:.3f}", ha="center", va="bottom", color="white", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([labels[a] for a in approaches], color="white", fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Dice (mean)", color="white")
    ax.set_title("Polyp segmentation — paradigm comparison (with prompt ablation)",
                 color="white", fontsize=12, pad=12)
    ax.tick_params(colors="white")
    for s in ax.spines.values(): s.set_color("#444")
    ax.grid(True, axis="y", color="#2a2a2a", linewidth=0.4)
    ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#cccccc", alpha=0.55, edgecolor="white", label=dataset_labels[0]),
        Patch(facecolor="#cccccc", alpha=1.0, edgecolor="white", label=dataset_labels[1]),
    ], loc="lower right", frameon=False, labelcolor="white")

    plt.tight_layout()
    out = RESULTS / "summary_with_prompt_ablation.png"
    plt.savefig(out, facecolor=fig.get_facecolor())
    print(f"wrote {out}")


if __name__ == "__main__":
    render_scatter()
    render_headline()
