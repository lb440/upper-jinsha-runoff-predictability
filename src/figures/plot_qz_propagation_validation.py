# -*- coding: utf-8 -*-
"""
Plot a hydrological propagation check for Qz lag attribution.

The figure includes:
1) cross-correlation between antecedent Zhimenda runoff and Shigu runoff;
2) dominant target-relative Qz lag ranges inferred from fixed-architecture SHAP;
3) an optional hydraulic travel-time range if channel distance and velocity are supplied.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# -----------------------------
# User-editable paths and options
# -----------------------------
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = Path(os.environ.get(
    'RUNOFF_DATA_PATH', str(REPOSITORY_ROOT / 'data' / 'example_model_input_synthetic.csv')
))
SHAP_ROOT = Path(os.environ.get('SHAP_OUTPUT_ROOT', str(REPOSITORY_ROOT / 'outputs' / 'fixed_shap')))
OUT_DIR = Path(os.environ.get('CCF_OUTPUT_DIR', str(REPOSITORY_ROOT / 'outputs' / 'qz_propagation_validation')))

DATE_COL = "Date"
UPSTREAM_COL = "Qz"
DOWNSTREAM_COL = "Q_shigu"

# Use the same period as the SHAP test attribution by default.
CCF_START_DATE = "2017-01-01"
CCF_END_DATE = "2020-12-31"

# CCF lags are defined as corr[Qz(t-k), Q_shigu(t)].
MAX_CCF_LAG_DAYS = 30

# Dominant SHAP lag band: keep lags with Qz mean |SHAP| >= this fraction of
# the lead-specific maximum. A high threshold avoids overinterpreting the long,
# low tail of the SHAP curve.
SHAP_DOMINANT_THRESHOLD_FRACTION = 0.50

# Optional physical travel-time band.
# Leave CHANNEL_DISTANCE_KM as None unless a reliable along-channel distance is
# available. If supplied, travel time = distance / velocity.
CHANNEL_DISTANCE_KM = None
VELOCITY_RANGE_MS = (0.5, 2.0)


def set_publication_style():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_runoff_data():
    df = pd.read_csv(DATA_CSV)
    required = {DATE_COL, UPSTREAM_COL, DOWNSTREAM_COL}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError("Missing required columns in data CSV: %s" % sorted(missing))

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    mask = (df[DATE_COL] >= CCF_START_DATE) & (df[DATE_COL] <= CCF_END_DATE)
    df = df.loc[mask, [DATE_COL, UPSTREAM_COL, DOWNSTREAM_COL]].dropna().reset_index(drop=True)
    if df.empty:
        raise ValueError("No data left after applying the selected CCF date range.")
    return df


def compute_cross_correlation(df, max_lag):
    rows = []
    downstream = df[DOWNSTREAM_COL].astype(float)
    for lag in range(max_lag + 1):
        upstream_lagged = df[UPSTREAM_COL].astype(float).shift(lag)
        valid = upstream_lagged.notna() & downstream.notna()
        corr = upstream_lagged.loc[valid].corr(downstream.loc[valid])
        rows.append({"lag_days": lag, "cross_correlation": corr, "n": int(valid.sum())})

    ccf = pd.DataFrame(rows)
    peak_row = ccf.loc[ccf["cross_correlation"].idxmax()]
    return ccf, int(peak_row["lag_days"]), float(peak_row["cross_correlation"])


def read_qz_shap_lag(horizon):
    fixed_path = (
        SHAP_ROOT
        / ("H%dd_manual_rerun" % horizon)
        / ("shap_analysis_H%dd_fixed" % horizon)
        / ("SHAP_feature_lag_mean_abs_matrix_H%dd.csv" % horizon)
    )
    fallback_path = (
        SHAP_ROOT / ("H%dd" % horizon) / ("SHAP_feature_lag_mean_abs_matrix_H%dd.csv" % horizon)
    )
    path = fixed_path if fixed_path.exists() else fallback_path
    if not path.exists():
        raise FileNotFoundError("Cannot find SHAP lag matrix for H%dd." % horizon)

    mat = pd.read_csv(path, index_col=0)
    if UPSTREAM_COL not in mat.index:
        raise ValueError("%s not found in SHAP lag matrix: %s" % (UPSTREAM_COL, path))

    qz = mat.loc[UPSTREAM_COL].astype(float)
    qz.index = qz.index.astype(int)
    return path, qz


def summarize_shap_lags(horizons):
    rows = []
    curves = {}
    for horizon in horizons:
        path, qz = read_qz_shap_lag(horizon)
        max_value = float(qz.max())
        peak_input_lag = int(qz.idxmax())
        threshold = SHAP_DOMINANT_THRESHOLD_FRACTION * max_value
        dominant_input_lags = qz.index[qz >= threshold].to_numpy(dtype=int)

        if dominant_input_lags.size:
            input_min = int(dominant_input_lags.min())
            input_max = int(dominant_input_lags.max())
        else:
            input_min = peak_input_lag
            input_max = peak_input_lag

        rows.append(
            {
                "lead_time_days": horizon,
                "source_file": str(path),
                "qz_peak_input_window_lag_days": peak_input_lag,
                "qz_peak_target_relative_lag_days": peak_input_lag + horizon,
                "dominant_threshold_fraction": SHAP_DOMINANT_THRESHOLD_FRACTION,
                "qz_dominant_input_lag_min_days": input_min,
                "qz_dominant_input_lag_max_days": input_max,
                "qz_dominant_target_relative_lag_min_days": input_min + horizon,
                "qz_dominant_target_relative_lag_max_days": input_max + horizon,
                "qz_max_mean_abs_shap": max_value,
            }
        )
        curves[horizon] = qz

    return pd.DataFrame(rows), curves


def travel_time_range_days():
    if CHANNEL_DISTANCE_KM is None:
        return None
    v_min, v_max = VELOCITY_RANGE_MS
    if v_min <= 0 or v_max <= 0:
        raise ValueError("VELOCITY_RANGE_MS must contain positive values.")
    low = CHANNEL_DISTANCE_KM / (max(v_min, v_max) * 86.4)
    high = CHANNEL_DISTANCE_KM / (min(v_min, v_max) * 86.4)
    return float(low), float(high)


def plot_figure(ccf, peak_lag, peak_corr, shap_summary, out_prefix):
    set_publication_style()
    fig, (ax, ax2) = plt.subplots(
        2,
        1,
        figsize=(4.0, 3.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.3, 1.0], "hspace": 0.06},
    )

    ax.plot(
        ccf["lag_days"],
        ccf["cross_correlation"],
        color="#1f77b4",
        lw=1.6,
        marker="o",
        ms=2.4,
        markevery=2,
        label="Cross-correlation",
    )
    ax.axhline(0, color="0.35", lw=0.8)
    ax.axvline(
        peak_lag,
        color="#d62728",
        lw=1.1,
        ls="--",
        label="Peak CCF lag",
    )

    travel = travel_time_range_days()
    if travel is not None:
        ax.axvspan(
            travel[0],
            travel[1],
            color="#f2c14e",
            alpha=0.28,
            lw=0,
            label="Estimated travel-time range",
        )

    ax.text(
        peak_lag + 0.4,
        peak_corr - 0.015,
        "Peak = %d d" % peak_lag,
        color="#d62728",
        fontsize=8,
        va="top",
    )

    ax.set_xlim(0, MAX_CCF_LAG_DAYS)
    ymin = max(-0.05, float(ccf["cross_correlation"].min()) - 0.03)
    ymax = min(1.05, float(ccf["cross_correlation"].max()) + 0.04)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("Cross-correlation\ncoefficient")
    ax.set_title("$Q_z$ propagation-lag validation", pad=5)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], loc="lower left", frameon=False)

    # Show target-relative Qz SHAP dominant lag ranges for the four lead times
    # on a separate evidence strip to avoid overloading the CCF panel.
    colors = {1: "#4c78a8", 3: "#59a14f", 7: "#f28e2b", 15: "#b07aa1"}
    y_positions = {1: 3, 3: 2, 7: 1, 15: 0}
    if travel is not None:
        ax2.axvspan(
            travel[0],
            travel[1],
            color="#f2c14e",
            alpha=0.28,
            lw=0,
        )
    ax2.axvline(peak_lag, color="#d62728", lw=1.1, ls="--")
    for _, row in shap_summary.iterrows():
        horizon = int(row["lead_time_days"])
        x0 = float(row["qz_dominant_target_relative_lag_min_days"])
        x1 = float(row["qz_dominant_target_relative_lag_max_days"])
        y = y_positions[horizon]
        ax2.hlines(y, x0, x1, color=colors.get(horizon, "0.3"), lw=4.0)
        ax2.plot(
            float(row["qz_peak_target_relative_lag_days"]),
            y,
            marker="|",
            color=colors.get(horizon, "0.3"),
            ms=9,
            mew=1.6,
        )

    ax2.set_yticks([3, 2, 1, 0])
    ax2.set_yticklabels(["1 d", "3 d", "7 d", "15 d"])
    ax2.set_ylabel("Lead time")
    ax2.set_xlabel("Lag of $Q_z$ relative to $Q_{\\mathrm{Shigu}}$ (d)")
    ax2.set_ylim(-0.7, 3.7)

    for axis in (ax, ax2):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    ax2.text(
        0.98,
        0.94,
        "Dominant $Q_z$ SHAP lag range",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.19, right=0.98, top=0.88, bottom=0.16, hspace=0.06)
    fig.savefig(str(out_prefix) + ".tiff", dpi=600, bbox_inches="tight")
    fig.savefig(str(out_prefix) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(out_prefix) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out_prefix) + ".svg", bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_runoff_data()
    ccf, peak_lag, peak_corr = compute_cross_correlation(df, MAX_CCF_LAG_DAYS)
    shap_summary, _ = summarize_shap_lags([1, 3, 7, 15])

    ccf_path = OUT_DIR / "qz_shigu_cross_correlation.csv"
    shap_path = OUT_DIR / "qz_shap_dominant_lag_summary.csv"
    figure_prefix = OUT_DIR / "qz_propagation_lag_validation"

    ccf.to_csv(ccf_path, index=False, encoding="utf-8-sig")
    shap_summary.to_csv(shap_path, index=False, encoding="utf-8-sig")
    plot_figure(ccf, peak_lag, peak_corr, shap_summary, figure_prefix)

    print("CCF period: %s to %s" % (CCF_START_DATE, CCF_END_DATE))
    print("Peak CCF lag: %d d; r = %.4f" % (peak_lag, peak_corr))
    print("Saved CCF table:", ccf_path)
    print("Saved SHAP lag summary:", shap_path)
    print("Saved figure:", str(figure_prefix) + ".tiff")


if __name__ == "__main__":
    main()
