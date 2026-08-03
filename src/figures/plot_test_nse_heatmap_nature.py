from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "source_data" / "Test_NSE_matrix.csv"
OUT_DIR = BASE_DIR / "outputs"
OUT_BASE = OUT_DIR / "test_nse_heatmap_nature"


mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})


def label_to_code(label: str) -> str:
    mapping = {
        "Qz only": "C1",
        "Qz + Pz + Pi": "C2",
        "Qz + Pz + Pi + Tz + Ti": "C3",
        "Qz + Pz + Pi + Ez + Ei": "C4",
        "Qz + Pz + Pi + Sz + Si": "C5",
        "Qz + Pz + Pi + Tz + Ti + Ez + Ei": "C6",
        "Full predictors": "C7",
    }
    return mapping[label]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    df["Code"] = df["Label"].map(label_to_code)
    df = df.set_index("Code")[["1", "3", "7", "15"]]
    df = df.loc[["C7", "C6", "C5", "C4", "C3", "C2", "C1"]]
    values = df.to_numpy(dtype=float)

    cmap = LinearSegmentedColormap.from_list(
        "test_nse_blue",
        ["#F8FBFF", "#DCECF5", "#B7D7EA", "#78B5D6", "#2F7FB9"],
        N=256,
    )
    norm = Normalize(vmin=0.79, vmax=0.97)

    fig, ax = plt.subplots(figsize=(4.5, 3.6), constrained_layout=False)
    im = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")

    ax.set_xticks(np.arange(df.shape[1]))
    ax.set_xticklabels(df.columns.tolist(), fontsize=8, color="black")
    ax.set_yticks(np.arange(df.shape[0]))
    ax.set_yticklabels(df.index.tolist(), fontsize=8, color="black")
    ax.set_xlabel("Lead time (d)", fontsize=9, labelpad=6, color="black")
    ax.set_ylabel("Input combination", fontsize=9, labelpad=7, color="black")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)

    ax.tick_params(axis="both", direction="out", length=3, width=0.8, pad=3, colors="black")

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(
                col,
                row,
                f"{values[row, col]:.3f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_title("NSE", fontsize=8, pad=5, color="black")
    cbar_ticks = np.linspace(0.79, 0.97, 7)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f"{tick:.2f}" for tick in cbar_ticks])
    cbar.ax.tick_params(labelsize=7, width=0.8, length=3, colors="black")
    cbar.outline.set_linewidth(0.8)

    fig.subplots_adjust(left=0.15, right=0.86, bottom=0.16, top=0.96)
    fig.savefig(OUT_BASE.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
