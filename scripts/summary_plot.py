"""Final summary chart: bars of Dice on Kvasir vs CVC-ClinicDB per approach."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gutcheck import APPROACHES, COLORS, LABELS
from scripts._common import METRICS_MASTER, RESULTS


def _rgb(c: tuple[int, int, int]) -> tuple[float, float, float]:
    return (c[0] / 255, c[1] / 255, c[2] / 255)


def main():
    data = json.loads(METRICS_MASTER.read_text())
    by_key = {(r["approach"], r["dataset"]): r for r in data}

    datasets = ["kvasir", "cvc_clinicdb"]
    dataset_labels = ["Kvasir-SEG test (in-distribution)", "CVC-ClinicDB (cross-dataset)"]

    n_ds = len(datasets)
    x = np.arange(len(APPROACHES))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=140)
    fig.patch.set_facecolor("#0c0c0f")
    ax.set_facecolor("#0c0c0f")

    for i, (ds, ds_label) in enumerate(zip(datasets, dataset_labels)):
        vals = []
        for a in APPROACHES:
            r = by_key.get((a, ds))
            vals.append(r["dice_mean"] if r else 0.0)
        offs = (i - (n_ds - 1) / 2) * width
        bars = ax.bar(
            x + offs, vals, width,
            color=[_rgb(COLORS[a]) for a in APPROACHES],
            edgecolor="white", linewidth=0.8,
            alpha=0.55 if i == 0 else 1.0,
            label=ds_label,
        )
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.01,
                    f"{v:.3f}", ha="center", va="bottom",
                    color="white", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[a] for a in APPROACHES], color="white", fontsize=10)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Dice (mean)", color="white")
    ax.set_title("Polyp segmentation — paradigm comparison", color="white", fontsize=13, pad=14)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.grid(True, axis="y", color="#333", linewidth=0.5)
    ax.set_axisbelow(True)

    # Legend entries: transparent box = in-dist, solid = cross-dataset
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#cccccc", alpha=0.55, edgecolor="white", label=dataset_labels[0]),
        Patch(facecolor="#cccccc", alpha=1.0, edgecolor="white", label=dataset_labels[1]),
    ]
    legend = ax.legend(handles=legend_handles, loc="lower right", frameon=False, labelcolor="white")

    out = RESULTS / "summary.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, facecolor=fig.get_facecolor())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
