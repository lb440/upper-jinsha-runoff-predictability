from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


def choose_font() -> str:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["Times New Roman", "Arial", "DejaVu Serif", "DejaVu Sans"]:
        if name in installed:
            return name
    return "DejaVu Sans"


FONT = choose_font()
BLUE = "#0B63FF"
ARROW = "#2793D1"
GREEN = "#4CAF30"
TEXT = "#111111"
BG = ["#EAF3FF", "#EBF8F3", "#F1FAEA", "#FFF0E2"]


def add_round_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str = "",
    *,
    fontsize: float = 12,
    weight: str = "normal",
    facecolor: str | None = "white",
    edgecolor: str = BLUE,
    linestyle: str | tuple = "-",
    linewidth: float = 1.35,
    radius: float = 0.45,
    zorder: int = 2,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.16,rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor="none" if facecolor is None else facecolor,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    if text:
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight=weight,
            fontname=FONT,
            linespacing=1.12,
            color=TEXT,
            zorder=zorder + 1,
        )
    return patch


def add_split_text(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    fontsize: float,
    zorder: int = 4,
) -> None:
    """Draw multiline text with the first line bold, matching the journal-style workflow boxes."""
    lines = text.split("\n")
    # Convert point size to y-axis data units for centered line placement.
    pt_to_y = 60.0 / (9.48 * 72.0)
    line_step = fontsize * 1.70 * pt_to_y
    y_center = y + h / 2
    y_top = y_center + line_step * (len(lines) - 1) / 2
    for idx, line in enumerate(lines):
        ax.text(
            x + w / 2,
            y_top - idx * line_step,
            line,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold" if idx == 0 else "normal",
            fontname=FONT,
            color=TEXT,
            zorder=zorder,
        )


def add_content_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    fontsize: float,
    radius: float = 0.45,
    zorder: int = 2,
) -> None:
    add_round_box(ax, x, y, w, h, facecolor="white", radius=radius, zorder=zorder)
    add_split_text(ax, x, y, w, h, text, fontsize=fontsize, zorder=zorder + 1)


def add_arrow(
    ax,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str = ARROW,
    mutation_scale: float = 16,
    lw: float = 2.0,
    arrowstyle: str = "-|>",
    zorder: int = 3,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle=arrowstyle,
            mutation_scale=mutation_scale,
            linewidth=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
        )
    )


def add_block_arrow(ax, x1: float, y: float, x2: float, *, color: str = GREEN) -> None:
    """Small filled inter-column arrow matching the uploaded workflow style."""
    h = 1.05
    tail_h = 0.46
    head_len = min(0.95, max(0.55, (x2 - x1) * 0.45))
    pts = [
        (x1, y - tail_h / 2),
        (x2 - head_len, y - tail_h / 2),
        (x2 - head_len, y - h / 2),
        (x2, y),
        (x2 - head_len, y + h / 2),
        (x2 - head_len, y + tail_h / 2),
        (x1, y + tail_h / 2),
    ]
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor=color, linewidth=0, zorder=4))


def add_column(ax, x: float, y: float, w: float, h: float, color: str, title: str) -> None:
    add_round_box(ax, x, y, w, h, facecolor=color, linewidth=1.8, radius=1.0, zorder=0)
    add_round_box(
        ax,
        x + 1.05,
        y + h - 7.0,
        w - 2.1,
        5.7,
        title,
        fontsize=14.4,
        weight="bold",
        facecolor=None,
        linestyle=(0, (14, 10)),
        linewidth=1.05,
        radius=3.0,
        zorder=1,
    )


def vertical_stack(ax, x: float, box_w: float, boxes: list[tuple[float, float, str, float]]) -> None:
    for i, (y, h, text, fs) in enumerate(boxes):
        add_content_box(ax, x, y, box_w, h, text, fontsize=fs)
        if i < len(boxes) - 1:
            y_next, h_next, *_ = boxes[i + 1]
            add_arrow(ax, x + box_w / 2, y - 0.55, x + box_w / 2, y_next + h_next + 0.65)


def main() -> None:
    out_dir = Path.home() / "Desktop" / "".join([chr(0x65B0), chr(0x7684), chr(0x56FE)])
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.family"] = FONT
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(16.59, 9.48), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    col_w = 22.4
    gap = 2.25
    xs = [0.7 + i * (col_w + gap) for i in range(4)]
    titles = [
        "Basin data\npreparation",
        "Multi-lead sample\ngeneration",
        "Lead-time-specific\nmodel development",
        "Test-period assessment\nand interpretation",
    ]
    for x, bg, title in zip(xs, BG, titles):
        add_column(ax, x, 1.0, col_w, 58.0, bg, title)

    vertical_stack(
        ax,
        xs[0] + 2.0,
        col_w - 4.0,
        [
            (40.4, 6.4, "Runoff observations\n$Q_{\\mathrm{Shigu}}$, $Q_{\\mathrm{z}}$", 15.0),
            (29.8, 7.2, "Hydro-meteorological forcing\n$P_{\\mathrm{z}}$, $T_{\\mathrm{z}}$, $E_{\\mathrm{z}}$, $S_{\\mathrm{z}}$\n$P_{\\mathrm{i}}$, $T_{\\mathrm{i}}$, $E_{\\mathrm{i}}$, $S_{\\mathrm{i}}$", 12.8),
            (17.2, 8.8, "Spatial subregions\nZhimenda catchment\nZhimenda-Shigu interval", 13.8),
            (4.2, 8.8, "Preprocessing\nHourly-to-daily aggregation\nSpatial averaging", 13.8),
        ],
    )

    vertical_stack(
        ax,
        xs[1] + 2.0,
        col_w - 4.0,
        [
            (40.4, 6.4, "90-d sliding window\n$X_t \\in R^{90\\times9} \\rightarrow Q_{\\mathrm{Shigu}}(t+h)$", 13.7),
            (29.8, 6.8, "Lead times\n1, 3, 7, and 15 d", 14.7),
            (16.4, 9.5, "Chronological split\nTrain: 2006-2014\nVal: 2015-2016\nTest: 2017-2020", 13.6),
            (4.2, 7.8, "Training-set normalization\nFitted on training set\nApplied to validation\nand test sets", 12.4),
        ],
    )

    # Model development.
    ix = xs[2] + 1.1
    iw = col_w - 2.2
    add_round_box(ax, ix, 34.7, iw, 12.0, facecolor=None, linestyle=(0, (6, 5)), linewidth=1.05, radius=0.6, zorder=1)
    model_items = [
        (ix + 0.6, 41.4, 5.8, "Persistence", 8.7),
        (ix + 6.85, 41.4, 5.0, "XGBoost", 9.0),
        (ix + 12.30, 41.4, 3.35, "MLP", 9.4),
        (ix + 16.10, 41.4, 3.50, "GRU", 9.4),
        (ix + 0.6, 36.3, 4.8, "LSTM", 9.3),
        (ix + 5.95, 36.3, 6.5, "Transformer", 8.8),
        (ix + 13.0, 36.3, 6.6, "LSTM-\nTransformer", 8.1),
    ]
    for x, y, w, label, fs in model_items:
        add_round_box(ax, x, y, w, 3.7, label, fontsize=fs, weight="bold", radius=0.35)
    add_arrow(ax, xs[2] + col_w / 2, 34.25, xs[2] + col_w / 2, 32.25)
    add_round_box(ax, ix, 3.6, iw, 28.0, facecolor=None, linestyle=(0, (6, 5)), linewidth=1.05, radius=0.6, zorder=1)
    bx, bw = ix + 1.0, iw - 2.0
    train_boxes = [
        (26.0, 4.6, "Lead-specific training", 14.0),
        (18.5, 4.6, "Bayesian optimization\nfor tunable models", 12.5),
        (11.0, 4.6, "Early stopping and\nvalidation selection", 12.5),
        (3.95, 3.9, "Best model per lead time", 12.8),
    ]
    for i, (y, h, text, fs) in enumerate(train_boxes):
        add_content_box(ax, bx, y, bw, h, text, fontsize=fs, radius=0.4)
        if i < len(train_boxes) - 1:
            y_next, h_next, *_ = train_boxes[i + 1]
            add_arrow(ax, bx + bw / 2, y - 0.48, bx + bw / 2, y_next + h_next + 0.48, mutation_scale=13, lw=2.2)

    vertical_stack(
        ax,
        xs[3] + 1.5,
        col_w - 3.0,
        [
            (41.0, 5.8, "Final test predictions\n2017-2020", 13.8),
            (31.9, 6.2, "Overall performance\nNSE, RMSE\nMAE, KGE", 12.8),
            (22.6, 6.2, "Input-factor combinations\n3-d lead time; C1-C8\nNSE, RMSE, MAE, KGE", 11.7),
            (12.3, 7.2, "Wet-season evaluation\nJune-October pooled samples\nHydrographs and absolute errors", 11.45),
            (3.8, 5.0, "Hydrological interpretation\nDominant predictors\nand lag effects", 11.8),
        ],
    )

    for i in range(3):
        add_block_arrow(ax, xs[i] + col_w + 0.35, 27.0, xs[i + 1] - 0.45)

    path = out_dir / "技术路线图_新版_600dpi.png"
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.08)
    print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
