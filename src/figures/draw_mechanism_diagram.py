from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs"


def choose_font() -> str:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["Times New Roman", "Cambria", "DejaVu Serif"]:
        if name in installed:
            return name
    return "DejaVu Serif"


FONT = choose_font()
EDGE = "#111111"
TEXT = "#111111"
COLORS = {
    "peach": "#F6C991",
    "green": "#D9EBC5",
    "blue": "#C6D9F1",
    "lavender": "#D7C1EA",
    "cream": "#F1E4C6",
    "yellow": "#F7E7A5",
    "pink": "#F2A7A7",
    "gray": "#ECECEC",
}


def add_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    fc: str,
    fs: float = 12,
    weight: str = "bold",
    lw: float = 1.1,
    radius: float = 0.16,
    linestyle: str | tuple = "-",
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.035,rounding_size={radius}",
        linewidth=lw,
        edgecolor=EDGE,
        facecolor=fc,
        linestyle=linestyle,
    )
    ax.add_patch(patch)
    if text:
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fs,
            fontweight=weight,
            fontname=FONT,
            color=TEXT,
            linespacing=1.18,
        )


def add_arrow(ax, x1: float, y1: float, x2: float, y2: float, *, lw: float = 1.3) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=lw,
            color=EDGE,
            shrinkA=0,
            shrinkB=0,
        )
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [FONT],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )

    fig, ax = plt.subplots(figsize=(10.9, 8.2), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis("off")

    # Hydro-meteorological input block.
    add_box(ax, 0.7, 2.35, 4.4, 6.45, "", fc="none", lw=1.1, radius=0.22, linestyle=(0, (5, 4)))
    ax.text(1.05, 8.45, "Hydro-meteorological drivers", fontsize=13.5, fontweight="bold", fontname=FONT)
    add_box(ax, 1.05, 6.95, 3.7, 1.05, "Upstream runoff\n$Q_z$", fc=COLORS["peach"], fs=12.2)
    add_box(ax, 1.05, 5.35, 3.7, 1.15, "Upstream catchment\n$P_z,\\ T_z,\\ E_z,\\ S_z$", fc=COLORS["green"], fs=11.5)
    add_box(
        ax,
        1.05,
        3.70,
        3.7,
        1.15,
        "Zhimenda-Shigu interval\n$P_i,\\ T_i,\\ E_i,\\ S_i$",
        fc=COLORS["blue"],
        fs=10.8,
    )
    ax.text(2.9, 3.18, r"$x_t\ \in\ R^9$", ha="center", va="center", fontsize=13.0, fontname=FONT)
    ax.text(2.9, 2.78, "(9 variables)", ha="center", va="center", fontsize=10.0, fontweight="bold", fontname=FONT)

    add_arrow(ax, 2.9, 2.35, 2.9, 1.92)
    add_box(ax, 0.85, 0.75, 3.9, 0.95, "90-d sliding window\n$X_t\\ \\in\\ R^{90\\times9}$", fc=COLORS["lavender"], fs=11.0)
    add_arrow(ax, 4.75, 1.22, 5.95, 1.22)

    # LSTM-Transformer stem.
    cx, bw = 6.05, 2.55
    add_box(ax, cx, 0.75, bw, 0.95, "Input sequence\n$90\\times9$", fc=COLORS["peach"], fs=10.9)
    add_arrow(ax, cx + bw / 2, 1.70, cx + bw / 2, 2.28)
    add_box(ax, cx, 2.28, bw, 0.72, "LSTM encoder", fc=COLORS["peach"], fs=10.4)
    add_arrow(ax, cx + bw / 2, 3.00, cx + bw / 2, 3.52)
    add_box(ax, cx, 3.52, bw, 0.72, "Layer Normalization", fc=COLORS["blue"], fs=9.8)
    add_arrow(ax, cx + bw / 2, 4.24, cx + bw / 2, 4.78)
    add_box(ax, cx, 4.78, bw, 0.72, "Linear projection", fc=COLORS["cream"], fs=10.1)
    add_arrow(ax, cx + bw / 2, 5.50, cx + bw / 2, 6.05)
    add_box(ax, cx, 6.05, bw, 0.72, "Positional encoding", fc=COLORS["lavender"], fs=9.6)

    # Transformer encoder.
    ex, ey, ew, eh = 9.25, 1.90, 4.25, 5.35
    add_box(ax, ex, ey, ew, eh, "", fc="none", lw=1.1, radius=0.18, linestyle=(0, (5, 4)))
    ax.text(ex + 0.08, ey + eh - 0.55, "Transformer encoder layer (\u00d71)", fontsize=13.0, fontweight="bold", fontname=FONT)
    add_arrow(ax, cx + bw, 6.41, ex + 0.45, 6.41)

    tx, tw, th = ex + 0.55, 3.20, 0.65
    add_box(ax, tx, 2.45, tw, th, "Multi-head self-attention", fc=COLORS["pink"], fs=9.5)
    add_arrow(ax, tx + tw / 2, 3.10, tx + tw / 2, 3.55)
    add_box(ax, tx, 3.55, tw, th, "Add & Norm", fc=COLORS["yellow"], fs=10.2)
    add_arrow(ax, tx + tw / 2, 4.20, tx + tw / 2, 4.68)
    add_box(ax, tx, 4.68, tw, th, "Feed Forward", fc=COLORS["green"], fs=10.0)
    add_arrow(ax, tx + tw / 2, 5.33, tx + tw / 2, 5.78)
    add_box(ax, tx, 5.78, tw, th, "Add & Norm", fc=COLORS["yellow"], fs=10.2)

    # Residual side arrows.
    ax.plot([tx + tw, tx + tw + 0.55, tx + tw + 0.55], [2.78, 2.78, 3.88], color=EDGE, lw=1.1)
    add_arrow(ax, tx + tw + 0.55, 3.88, tx + tw, 3.88, lw=1.1)
    ax.plot([tx + tw, tx + tw + 0.55, tx + tw + 0.55], [5.00, 5.00, 6.10], color=EDGE, lw=1.1)
    add_arrow(ax, tx + tw + 0.55, 6.10, tx + tw, 6.10, lw=1.1)

    add_arrow(ax, ex + ew / 2, ey + eh, ex + ew / 2, 7.80)
    add_box(ax, 9.65, 7.80, 3.60, 0.63, "Last time step extraction", fc=COLORS["gray"], fs=9.6)
    add_arrow(ax, 11.45, 8.43, 11.45, 8.90)
    add_box(ax, 9.65, 8.90, 3.75, 0.63, "FC + ReLU + Dropout", fc=COLORS["peach"], fs=9.8)
    add_arrow(ax, 11.45, 9.53, 11.45, 10.03)
    add_box(ax, 10.00, 10.03, 3.10, 0.62, "Linear output", fc=COLORS["green"], fs=10.0)
    add_arrow(ax, 11.45, 10.65, 11.45, 11.13)
    add_box(ax, 10.45, 11.13, 2.05, 0.72, r"$\hat{Q}^{Shigu}_{t+h}$", fc=COLORS["green"], fs=15.0)

    ax.text(12.75, 11.63, "Predicted runoff at Shigu", fontsize=13.5, fontweight="bold", fontname=FONT)
    ax.text(13.45, 11.10, r"$h\ =\ 1,\ 3,\ 7,\ 15\ d$", fontsize=13.0, fontweight="bold", fontname=FONT)

    for ext in [".png", ".tiff"]:
        fig.savefig(OUT_DIR / f"mechanism_diagram_600dpi{ext}", dpi=600, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


if __name__ == "__main__":
    main()
