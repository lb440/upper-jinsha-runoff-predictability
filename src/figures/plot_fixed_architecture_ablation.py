"""Plot fixed-architecture deletion effects relative to the Full input set."""

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = Path(
    os.environ.get(
        "ABLATION_OUTPUT_ROOT",
        str(REPOSITORY_ROOT / "outputs" / "fixed_architecture_ablation_H7_H15"),
    )
)
SOURCE = RESULT_ROOT / "fixed_architecture_ablation_H7_H15_metrics.csv"
OUT = RESULT_ROOT / "fixed_architecture_ablation_delta_vs_full"

mpl.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

DISPLAY = {
    "Fast_only": "Fast-only\n(Qz, Pz, Pi)",
    "No_snow": "No snow\n(Full - Sz, Si)",
    "No_thermal_evaporative": "No thermal-evaporative\n(Full - Tz, Ti, Ez, Ei)",
}
COLORS = {
    "Fast_only": "#D55E00",
    "No_snow": "#0072B2",
    "No_thermal_evaporative": "#6B6B6B",
}


def panel(ax, data, metric, label):
    sub = data.loc[data["Input configuration"].isin(DISPLAY)].copy()
    sub["order"] = sub["Input configuration"].map(
        {key: index for index, key in enumerate(DISPLAY)}
    )
    sub = sub.sort_values("order")
    y = np.arange(len(sub))[::-1]
    ax.axvline(0, color="#777777", linewidth=0.8, zorder=0)
    for y_pos, (_, row) in zip(y, sub.iterrows()):
        color = COLORS[row["Input configuration"]]
        ax.hlines(y_pos, min(0, row[metric]), max(0, row[metric]), color=color, linewidth=1.8)
        ax.scatter(row[metric], y_pos, s=42, color=color, edgecolors="white", linewidths=0.6, zorder=3)
        ha = "right" if row[metric] < 0 else "left"
        dx = -0.00035 if row[metric] < 0 else 0.00035
        ax.text(row[metric] + dx, y_pos + 0.10, f"{row[metric]:+.3f}", ha=ha, va="bottom", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([DISPLAY[x] for x in sub["Input configuration"]], fontsize=8)
    ax.set_ylim(-0.15, 2.35)
    ax.set_xlabel(label)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.5, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    data = pd.read_csv(SOURCE)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.15), dpi=600, sharey=True)
    panel(axes[0], data.loc[data["Lead time (d)"] == 7], "Delta NSE vs Full", "Delta NSE relative to Full")
    panel(axes[1], data.loc[data["Lead time (d)"] == 15], "Delta NSE vs Full", "Delta NSE relative to Full")
    axes[0].set_title("(a) 7-day lead", loc="left", fontweight="bold", fontsize=9, pad=11)
    axes[1].set_title("(b) 15-day lead", loc="left", fontweight="bold", fontsize=9, pad=11)
    x_min = data.loc[data["Input configuration"].isin(DISPLAY), "Delta NSE vs Full"].min()
    for ax in axes:
        ax.set_xlim(min(-0.020, x_min - 0.002), 0.010)
        ax.tick_params(axis="both", which="major", length=3, width=0.8)
    fig.subplots_adjust(left=0.30, right=0.99, bottom=0.20, top=0.89, wspace=0.28)
    for suffix, kwargs in {".tiff": {"dpi": 600}, ".png": {"dpi": 600}, ".pdf": {}, ".svg": {}}.items():
        fig.savefig(OUT.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
