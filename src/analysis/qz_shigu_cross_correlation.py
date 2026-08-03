# -*- coding: utf-8 -*-
"""
Cross-correlation analysis between upstream Zhimenda runoff (Qz) and
downstream Shigu runoff (QShigu).

Positive lag k is defined as corr[Qz(t-k), QShigu(t)], meaning that Qz leads
QShigu by k days. This definition is consistent with hydrological propagation
from the upstream boundary station to the downstream target station.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# -----------------------------
# Paths and analysis settings
# -----------------------------
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = Path(os.environ.get(
    'RUNOFF_DATA_PATH', str(REPOSITORY_ROOT / 'data' / 'example_model_input_synthetic.csv')
))
OUTPUT_DIR = Path(os.environ.get(
    'CCF_OUTPUT_DIR', str(REPOSITORY_ROOT / 'outputs' / 'qz_propagation_validation')
))

DATE_COL = "Date"
UPSTREAM_COL = "Qz"
DOWNSTREAM_COL = "Q_shigu"

# Use the test period to match the SHAP attribution period.
ANALYSIS_START_DATE = "2017-01-01"
ANALYSIS_END_DATE = "2020-12-31"
MAX_LAG_DAYS = 30

FIG_WIDTH_MM = 89
FIG_HEIGHT_MM = 62
EXPORT_DPI = 600


def mm_to_inch(value_mm):
    return value_mm / 25.4


def read_csv_auto(path):
    encodings = ("utf-8-sig", "utf-8", "gbk", "gb2312")
    last_error = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as exc:
            last_error = exc
    raise RuntimeError("Cannot read CSV file: %s; last error: %s" % (path, last_error))


def set_publication_style():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "legend.frameon": False,
        }
    )


def load_runoff_data():
    df = read_csv_auto(DATA_CSV)
    required_cols = {DATE_COL, UPSTREAM_COL, DOWNSTREAM_COL}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise ValueError("Missing required columns: %s" % sorted(missing_cols))

    df = df[[DATE_COL, UPSTREAM_COL, DOWNSTREAM_COL]].copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[UPSTREAM_COL] = pd.to_numeric(df[UPSTREAM_COL], errors="coerce")
    df[DOWNSTREAM_COL] = pd.to_numeric(df[DOWNSTREAM_COL], errors="coerce")

    before_drop = len(df)
    df = df.dropna(subset=[DATE_COL, UPSTREAM_COL, DOWNSTREAM_COL]).sort_values(DATE_COL)
    after_drop = len(df)
    if after_drop < before_drop:
        print("Dropped invalid rows: %d" % (before_drop - after_drop))

    period_mask = (df[DATE_COL] >= ANALYSIS_START_DATE) & (df[DATE_COL] <= ANALYSIS_END_DATE)
    df = df.loc[period_mask].reset_index(drop=True)
    if df.empty:
        raise ValueError("No records remain in the selected analysis period.")
    return df


def compute_ccf(df, max_lag_days):
    rows = []
    downstream = df[DOWNSTREAM_COL].astype(float)

    for lag in range(max_lag_days + 1):
        upstream_lagged = df[UPSTREAM_COL].astype(float).shift(lag)
        valid = upstream_lagged.notna() & downstream.notna()
        corr = upstream_lagged.loc[valid].corr(downstream.loc[valid])
        rows.append(
            {
                "lag_days": lag,
                "cross_correlation": corr,
                "sample_count": int(valid.sum()),
            }
        )

    ccf = pd.DataFrame(rows)
    peak = ccf.loc[ccf["cross_correlation"].idxmax()]
    return ccf, int(peak["lag_days"]), float(peak["cross_correlation"])


def plot_ccf(ccf, peak_lag, peak_corr, output_prefix):
    set_publication_style()
    fig, ax = plt.subplots(
        figsize=(mm_to_inch(FIG_WIDTH_MM), mm_to_inch(FIG_HEIGHT_MM))
    )

    ax.plot(
        ccf["lag_days"],
        ccf["cross_correlation"],
        color="#1f77b4",
        lw=1.6,
        marker="o",
        ms=3.0,
        mfc="#1f77b4",
        mec="#1f77b4",
    )
    ax.axvline(peak_lag, color="#d62728", lw=1.2, ls="--")
    ax.text(
        peak_lag + 0.45,
        peak_corr - 0.014,
        "Peak lag = %d d" % peak_lag,
        color="#d62728",
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.5},
    )

    ax.set_xlim(0, MAX_LAG_DAYS)
    ax.set_ylim(0.68, 0.94)
    ax.set_xticks(np.arange(0, MAX_LAG_DAYS + 1, 5))
    ax.set_yticks(np.arange(0.70, 0.95, 0.05))

    ax.set_xlabel("Lag of $Q_z$ relative to $Q_{\\mathrm{Shigu}}$ (d)")
    ax.set_ylabel("Cross-correlation coefficient")
    ax.set_title("$Q_z$-$Q_{\\mathrm{Shigu}}$ cross-correlation", pad=5)

    ax.grid(axis="y", color="0.88", lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.19, top=0.87)

    fig.savefig(str(output_prefix) + ".tiff", dpi=EXPORT_DPI)
    fig.savefig(str(output_prefix) + ".png", dpi=300)
    fig.savefig(str(output_prefix) + ".pdf")
    fig.savefig(str(output_prefix) + ".svg")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_runoff_data()
    ccf, peak_lag, peak_corr = compute_ccf(data, MAX_LAG_DAYS)

    output_prefix = OUTPUT_DIR / "qz_shigu_ccf_publication"
    ccf_path = OUTPUT_DIR / "qz_shigu_ccf_values.csv"
    ccf.to_csv(ccf_path, index=False, encoding="utf-8-sig")
    plot_ccf(ccf, peak_lag, peak_corr, output_prefix)

    print("Analysis period: %s to %s" % (ANALYSIS_START_DATE, ANALYSIS_END_DATE))
    print("Sample size: %d" % len(data))
    print("Peak CCF lag: %d d" % peak_lag)
    print("Peak correlation: %.4f" % peak_corr)
    print("Saved table: %s" % ccf_path)
    print("Saved figure: %s.tiff" % output_prefix)


if __name__ == "__main__":
    main()
