# -*- coding: utf-8 -*-
"""Seasonality-control baselines for the fixed LSTM-Transformer experiment.

This script implements seasonal baseline checks only. It does not retrain the
deep model and does not recompute SHAP values.
"""

import math
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = Path(os.environ.get('RUNOFF_DATA_PATH', str(PROJECT_ROOT / 'data' / 'example_model_input_synthetic.csv')))
FIXED_OUTPUT_ROOT = Path(os.environ.get('FIXED_SHAP_OUTPUT_ROOT', str(PROJECT_ROOT / 'outputs' / 'fixed_shap')))
OUTPUT_DIR = Path(os.environ.get('SEASONAL_BASELINE_OUTPUT_ROOT', str(PROJECT_ROOT / 'outputs' / 'seasonality_control')))

TIME_COL = "Date"
TARGET_COL = "Q_shigu"
HORIZONS = [1, 3, 7, 15]
TRAIN_END_DATE = pd.Timestamp("2014-12-31")
VAL_END_DATE = pd.Timestamp("2016-12-31")
TEST_END_DATE = pd.Timestamp("2020-12-31")
DOY_WINDOW_DAYS = int(os.environ.get("DOY_CLIMATOLOGY_WINDOW_DAYS", "15"))
WET_MONTHS = {6, 7, 8, 9, 10}


MODEL_LABEL = "Fixed LSTM-Transformer"
DOY_LABEL = "Day-of-year climatology"
PREV_YEAR_LABEL = "Previous-year same-day flow"
SEASONAL_MEAN_LABEL = "Wet/dry seasonal mean"


def read_csv_auto(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk")


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if y_true.size == 0:
        return {
            "NSE": np.nan,
            "RMSE (m3/s)": np.nan,
            "MAE (m3/s)": np.nan,
            "PBIAS (%)": np.nan,
            "KGE": np.nan,
            "R": np.nan,
            "Samples": 0,
        }

    residual = y_pred - y_true
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    nse = float(1.0 - np.sum((y_true - y_pred) ** 2) / denom) if denom != 0 else np.nan
    pbias = float(100.0 * np.sum(y_pred - y_true) / (np.sum(y_true) + 1e-8))

    if y_true.size > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        r = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        r = np.nan

    alpha = float(np.std(y_pred) / (np.std(y_true) + 1e-8))
    beta = float(np.mean(y_pred) / (np.mean(y_true) + 1e-8))
    kge = float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)) if np.isfinite(r) else np.nan

    return {
        "NSE": nse,
        "RMSE (m3/s)": rmse,
        "MAE (m3/s)": mae,
        "PBIAS (%)": pbias,
        "KGE": kge,
        "R": r,
        "Samples": int(y_true.size),
    }


def circular_doy_distance(a, b, period=366):
    diff = np.abs(np.asarray(a) - b)
    return np.minimum(diff, period - diff)


def fit_doy_climatology(train_df, window_days):
    train = train_df.copy()
    train["doy"] = train[TIME_COL].dt.dayofyear
    global_mean = float(train[TARGET_COL].mean())
    doy_values = train["doy"].to_numpy()
    q_values = train[TARGET_COL].to_numpy(dtype=np.float64)

    lookup = {}
    for doy in range(1, 367):
        mask = circular_doy_distance(doy_values, doy) <= window_days
        lookup[doy] = float(np.mean(q_values[mask])) if np.any(mask) else global_mean
    return lookup


def predict_doy_climatology(dates, lookup):
    return np.asarray([lookup[int(pd.Timestamp(date).dayofyear)] for date in dates], dtype=np.float64)


def fit_wet_dry_means(train_df):
    train = train_df.copy()
    wet_mask = train[TIME_COL].dt.month.isin(WET_MONTHS)
    return {
        "wet": float(train.loc[wet_mask, TARGET_COL].mean()),
        "dry": float(train.loc[~wet_mask, TARGET_COL].mean()),
    }


def predict_wet_dry_mean(dates, means):
    return np.asarray(
        [
            means["wet"] if pd.Timestamp(date).month in WET_MONTHS else means["dry"]
            for date in dates
        ],
        dtype=np.float64,
    )


def previous_year_date(date):
    date = pd.Timestamp(date)
    try:
        return date.replace(year=date.year - 1)
    except ValueError:
        return date.replace(year=date.year - 1, day=28)


def predict_previous_year_same_day(dates, full_df, fallback):
    q_lookup = dict(zip(full_df[TIME_COL], full_df[TARGET_COL]))
    values = []
    missing = 0
    for date in dates:
        prev = previous_year_date(date)
        value = q_lookup.get(prev)
        if value is None or not np.isfinite(value):
            value = fallback[int(pd.Timestamp(date).dayofyear)]
            missing += 1
        values.append(float(value))
    return np.asarray(values, dtype=np.float64), missing


def load_fixed_prediction(horizon):
    path = FIXED_OUTPUT_ROOT / ("H%dd" % horizon) / ("prediction_test_H%dd.csv" % horizon)
    if not path.exists():
        raise FileNotFoundError("Missing fixed LSTM-Transformer prediction file: %s" % path)
    df = read_csv_auto(path)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    return df


def add_metric_row(rows, horizon, model, y_true, y_pred, note):
    metrics = calculate_metrics(y_true, y_pred)
    rows.append(
        {
            "Lead time (d)": horizon,
            "Model": model,
            **metrics,
            "Note": note,
        }
    )


def configure_matplotlib():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def save_main_figure(nse_df):
    configure_matplotlib()

    main_df = nse_df[nse_df["Model"].isin([MODEL_LABEL, DOY_LABEL])]
    pivot = main_df.pivot(index="Lead time (d)", columns="Model", values="NSE").loc[HORIZONS]

    fig, ax = plt.subplots(figsize=(3.35, 2.35), dpi=300)
    ax.plot(
        HORIZONS,
        pivot[MODEL_LABEL],
        marker="o",
        linewidth=1.4,
        markersize=4,
        color="#2B6CB0",
        label=MODEL_LABEL,
    )
    ax.plot(
        HORIZONS,
        pivot[DOY_LABEL],
        marker="s",
        linewidth=1.4,
        markersize=4,
        color="#C2410C",
        label=DOY_LABEL,
    )

    ax.set_xlabel("Lead time (d)")
    ax.set_ylabel("Test NSE")
    ax.set_xticks(HORIZONS)
    ax.set_ylim(min(0.0, np.nanmin(pivot.to_numpy()) - 0.05), 1.0)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.45)
    ax.legend(loc="lower left")
    fig.tight_layout()

    base = OUTPUT_DIR / "seasonal_baseline_nse_main"
    fig.savefig(str(base) + ".tif", dpi=600, bbox_inches="tight")
    fig.savefig(str(base) + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(str(base) + ".svg", bbox_inches="tight")
    fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    plt.close(fig)


def save_all_baseline_figure(nse_df):
    configure_matplotlib()

    pivot = nse_df.pivot(index="Lead time (d)", columns="Model", values="NSE").loc[HORIZONS]
    plot_order = [MODEL_LABEL, DOY_LABEL, PREV_YEAR_LABEL, SEASONAL_MEAN_LABEL]
    styles = {
        MODEL_LABEL: ("#2B6CB0", "o"),
        DOY_LABEL: ("#C2410C", "s"),
        PREV_YEAR_LABEL: ("#6B7280", "^"),
        SEASONAL_MEAN_LABEL: ("#15803D", "D"),
    }

    fig, ax = plt.subplots(figsize=(4.15, 2.55), dpi=300)
    for label in plot_order:
        color, marker = styles[label]
        ax.plot(
            HORIZONS,
            pivot[label],
            marker=marker,
            linewidth=1.2,
            markersize=3.8,
            color=color,
            label=label,
        )

    ax.set_xlabel("Lead time (d)")
    ax.set_ylabel("Test NSE")
    ax.set_xticks(HORIZONS)
    ax.set_ylim(min(0.0, np.nanmin(pivot.to_numpy()) - 0.05), 1.0)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.45)
    ax.legend(loc="lower left", ncol=1)
    fig.tight_layout()

    base = OUTPUT_DIR / "seasonal_baseline_nse_all_baselines"
    fig.savefig(str(base) + ".tif", dpi=600, bbox_inches="tight")
    fig.savefig(str(base) + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(str(base) + ".svg", bbox_inches="tight")
    fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = read_csv_auto(DATA_PATH)
    data.columns = data.columns.str.strip()
    data[TIME_COL] = pd.to_datetime(data[TIME_COL])
    data = data.sort_values(TIME_COL).reset_index(drop=True)
    data = data[(data[TIME_COL] >= "2006-01-01") & (data[TIME_COL] <= "2020-12-31")].copy()
    data[TARGET_COL] = pd.to_numeric(data[TARGET_COL], errors="coerce")
    rows_before_dropna = len(data)
    data = data.dropna(subset=[TIME_COL, TARGET_COL]).reset_index(drop=True)
    rows_after_dropna = len(data)

    train_df = data[data[TIME_COL] <= TRAIN_END_DATE].copy()
    doy_lookup = fit_doy_climatology(train_df, DOY_WINDOW_DAYS)
    seasonal_means = fit_wet_dry_means(train_df)

    metrics_rows = []
    all_prediction_frames = {}

    for horizon in HORIZONS:
        fixed_df = load_fixed_prediction(horizon)
        dates = fixed_df[TIME_COL]
        observed = pd.to_numeric(fixed_df["Observed"], errors="coerce").to_numpy(dtype=np.float64)
        fixed_pred = pd.to_numeric(fixed_df["Predicted"], errors="coerce").to_numpy(dtype=np.float64)

        doy_pred = predict_doy_climatology(dates, doy_lookup)
        prev_year_pred, prev_year_missing = predict_previous_year_same_day(dates, data, doy_lookup)
        seasonal_pred = predict_wet_dry_mean(dates, seasonal_means)

        pred_df = pd.DataFrame(
            {
                "Date": dates.dt.strftime("%Y-%m-%d"),
                "Observed": observed,
                MODEL_LABEL: fixed_pred,
                DOY_LABEL: doy_pred,
                PREV_YEAR_LABEL: prev_year_pred,
                SEASONAL_MEAN_LABEL: seasonal_pred,
            }
        )
        all_prediction_frames[horizon] = pred_df
        pred_df.to_csv(
            OUTPUT_DIR / ("seasonal_baseline_predictions_H%dd.csv" % horizon),
            index=False,
            encoding="utf-8-sig",
        )

        add_metric_row(
            metrics_rows,
            horizon,
            MODEL_LABEL,
            observed,
            fixed_pred,
            "Existing fixed-architecture LSTM-Transformer test prediction.",
        )
        add_metric_row(
            metrics_rows,
            horizon,
            DOY_LABEL,
            observed,
            doy_pred,
            "Training-period day-of-year runoff climatology with +/- %d d smoothing window." % DOY_WINDOW_DAYS,
        )
        add_metric_row(
            metrics_rows,
            horizon,
            PREV_YEAR_LABEL,
            observed,
            prev_year_pred,
            "Chronological previous-year same-calendar-day runoff; fallback count=%d." % prev_year_missing,
        )
        add_metric_row(
            metrics_rows,
            horizon,
            SEASONAL_MEAN_LABEL,
            observed,
            seasonal_pred,
            "Training-period wet/dry seasonal mean; wet season is June-October.",
        )

    metrics_df = pd.DataFrame(metrics_rows)
    model_order = {
        MODEL_LABEL: 0,
        DOY_LABEL: 1,
        PREV_YEAR_LABEL: 2,
        SEASONAL_MEAN_LABEL: 3,
    }
    metrics_df["Model order"] = metrics_df["Model"].map(model_order)
    metrics_df = metrics_df.sort_values(["Lead time (d)", "Model order"]).drop(columns=["Model order"])

    rounded = metrics_df.copy()
    for col in ["NSE", "RMSE (m3/s)", "MAE (m3/s)", "PBIAS (%)", "KGE", "R"]:
        rounded[col] = rounded[col].round(3)

    nse_df = metrics_df[["Lead time (d)", "Model", "NSE"]].copy()
    nse_pivot = nse_df.pivot(index="Lead time (d)", columns="Model", values="NSE").reset_index()
    nse_pivot = nse_pivot[["Lead time (d)", MODEL_LABEL, DOY_LABEL, PREV_YEAR_LABEL, SEASONAL_MEAN_LABEL]]

    metrics_df.to_csv(OUTPUT_DIR / "seasonal_baseline_metrics_full.csv", index=False, encoding="utf-8-sig")
    rounded.to_csv(OUTPUT_DIR / "seasonal_baseline_metrics_rounded.csv", index=False, encoding="utf-8-sig")
    nse_pivot.to_csv(OUTPUT_DIR / "seasonal_baseline_nse_for_origin.csv", index=False, encoding="utf-8-sig")

    try:
        with pd.ExcelWriter(OUTPUT_DIR / "seasonal_baseline_results.xlsx") as writer:
            metrics_df.to_excel(writer, sheet_name="Metrics_full", index=False)
            rounded.to_excel(writer, sheet_name="Metrics_rounded", index=False)
            nse_pivot.to_excel(writer, sheet_name="NSE_for_figure", index=False)
            for horizon, pred_df in all_prediction_frames.items():
                pred_df.to_excel(writer, sheet_name="H%dd_predictions" % horizon, index=False)
    except Exception as exc:
        print("Excel export skipped: %s" % exc)

    save_main_figure(nse_df)
    save_all_baseline_figure(nse_df)

    notes = [
        "Seasonality-control analysis: seasonal baselines only.",
        "Data: %s" % DATA_PATH,
        "Fixed LSTM-Transformer predictions: %s" % FIXED_OUTPUT_ROOT,
        "Rows before/after Date and Q_shigu missing-value filtering: %d/%d." % (rows_before_dropna, rows_after_dropna),
        "Training period for fitted climatological baselines: 2006-01-01 to 2014-12-31.",
        "Day-of-year climatology uses a +/- %d d circular smoothing window over a 366-day calendar." % DOY_WINDOW_DAYS,
        "Wet/dry seasonal mean uses June-October as wet season and November-May as dry season.",
        "Previous-year same-day flow is a chronological historical-flow reference; it may use observations after the training period when they are earlier than the evaluated target date.",
    ]
    (OUTPUT_DIR / "seasonal_baseline_notes.txt").write_text("\n".join(notes), encoding="utf-8")

    print("Seasonality-control baseline outputs saved to: %s" % OUTPUT_DIR)
    print(rounded.to_string(index=False))


if __name__ == "__main__":
    main()
