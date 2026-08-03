import math
import os
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = Path(os.environ.get('RUN_ROOT', str(PROJECT_ROOT / 'outputs' / 'model_runs')))
OUTPUT_ROOT = Path(os.environ.get('WET_SEASON_OUTPUT_ROOT', str(PROJECT_ROOT / 'outputs' / 'wet_season')))

WET_MONTHS = (6, 7, 8, 9, 10)
WET_YEARS = (2017, 2018, 2019, 2020)
HORIZONS = (1, 3, 7, 15)
MODEL_ORDER = [
    "Persistence",
    "XGBoost",
    "MLP",
    "GRU",
    "LSTM",
    "Transformer",
    "LSTM-Transformer",
]
MODEL_COLORS = {
    "Observed runoff": "black",
    "Persistence": "#ff4040",
    "XGBoost": "#1f70d6",
    "GRU": "#8a8a00",
    "MLP": "#b36ade",
    "LSTM": "#13c4c4",
    "Transformer": "#7f4a4a",
    "LSTM-Transformer": "#36ad6b",
}


def prediction_path(model, horizon):
    if model == "Persistence":
        return RUN_ROOT / "new_core" / "Persistence" / f"train_H{horizon}" / "prediction_test.csv"
    if model == "XGBoost":
        return RUN_ROOT / "new_core" / "XGBoost" / f"train_H{horizon}" / "prediction_test.csv"
    if model == "GRU":
        return RUN_ROOT / "new_core" / "GRU" / f"train_H{horizon}" / "prediction_test.csv"
    if model == "LSTM-Transformer":
        return (
            RUN_ROOT
            / "old_core"
            / f"L-T-{horizon}"
            / f"prediction_test_{horizon}d_no_lag.csv"
        )
    return RUN_ROOT / "old_core" / f"{model}-{horizon}" / "prediction_test.csv"


def normalize_prediction_frame(path, model, horizon):
    df = pd.read_csv(path)
    rename_map = {
        "Observed": "True_Q_shigu",
        "Predicted": "Pred_Q_shigu",
    }
    df = df.rename(columns=rename_map)
    required = {"Date", "True_Q_shigu", "Pred_Q_shigu"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    out = df[["Date", "True_Q_shigu", "Pred_Q_shigu"]].copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out["Model"] = model
    out["Lead time (d)"] = horizon
    out["Error"] = out["Pred_Q_shigu"] - out["True_Q_shigu"]
    out["Absolute error"] = out["Error"].abs()
    out["Year"] = out["Date"].dt.year
    out["Month"] = out["Date"].dt.month
    return out


def calculate_metrics(true_vals, pred_vals):
    true_vals = np.asarray(true_vals, dtype=np.float64)
    pred_vals = np.asarray(pred_vals, dtype=np.float64)
    error = pred_vals - true_vals
    mse = float(np.mean(error ** 2))
    rmse = math.sqrt(mse)
    mae = float(np.mean(np.abs(error)))
    ss_res = float(np.sum(error ** 2))
    ss_tot = float(np.sum((true_vals - np.mean(true_vals)) ** 2))
    nse = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan
    r = float(np.corrcoef(true_vals, pred_vals)[0, 1])
    alpha = float(np.std(pred_vals) / (np.std(true_vals) + 1e-8))
    beta = float(np.mean(pred_vals) / (np.mean(true_vals) + 1e-8))
    kge = 1.0 - math.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    pbias = float(np.sum(error) / (np.sum(true_vals) + 1e-8) * 100.0)
    mape = float(np.mean(np.abs(error) / (true_vals + 1e-8)) * 100.0)
    return {
        "NSE": nse,
        "RMSE (m3/s)": rmse,
        "MAE (m3/s)": mae,
        "PBIAS (%)": pbias,
        "KGE": kge,
        "R": r,
        "MAPE (%)": mape,
        "Bias (m3/s)": float(np.mean(error)),
        "Samples": int(len(true_vals)),
    }


def load_all_predictions():
    frames = []
    for horizon in HORIZONS:
        for model in MODEL_ORDER:
            path = prediction_path(model, horizon)
            if not path.exists():
                raise FileNotFoundError(path)
            frames.append(normalize_prediction_frame(path, model, horizon))
    return pd.concat(frames, ignore_index=True)


def build_wet_season_tables(all_predictions):
    wet = all_predictions[
        all_predictions["Month"].isin(WET_MONTHS)
        & all_predictions["Year"].isin(WET_YEARS)
    ].copy()

    overall_rows = []
    yearly_rows = []
    for horizon in HORIZONS:
        for model in MODEL_ORDER:
            subset = wet[
                (wet["Lead time (d)"] == horizon) & (wet["Model"] == model)
            ].copy()
            row = {
                "Lead time (d)": horizon,
                "Model": model,
                **calculate_metrics(subset["True_Q_shigu"], subset["Pred_Q_shigu"]),
            }
            overall_rows.append(row)

            for year in WET_YEARS:
                year_subset = subset[subset["Year"] == year].copy()
                yearly_rows.append(
                    {
                        "Lead time (d)": horizon,
                        "Model": model,
                        "Year": year,
                        **calculate_metrics(
                            year_subset["True_Q_shigu"],
                            year_subset["Pred_Q_shigu"],
                        ),
                    }
                )

    overall = pd.DataFrame(overall_rows)
    yearly = pd.DataFrame(yearly_rows)
    overall["ModelOrder"] = overall["Model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    yearly["ModelOrder"] = yearly["Model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    overall = overall.sort_values(["Lead time (d)", "ModelOrder"]).drop(columns=["ModelOrder"])
    yearly = yearly.sort_values(["Lead time (d)", "ModelOrder", "Year"]).drop(columns=["ModelOrder"])
    return wet, overall.reset_index(drop=True), yearly.reset_index(drop=True)


def export_table4(overall):
    table4_cols = [
        "Lead time (d)",
        "Model",
        "NSE",
        "RMSE (m3/s)",
        "MAE (m3/s)",
        "PBIAS (%)",
        "KGE",
    ]
    table4 = overall[table4_cols].copy()
    table4.to_csv(OUTPUT_ROOT / "table4_wet_season_metrics_corrected.csv", index=False, encoding="utf-8-sig")

    rounded = table4.copy()
    for col in ["NSE", "RMSE (m3/s)", "MAE (m3/s)", "PBIAS (%)", "KGE"]:
        rounded[col] = rounded[col].map(lambda x: f"{x:.3f}")
    rounded.to_csv(
        OUTPUT_ROOT / "table4_wet_season_metrics_corrected_rounded.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return table4, rounded


def set_plot_style():
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 14,
        }
    )


def plot_fig7(wet):
    h3 = wet[wet["Lead time (d)"] == 3].copy()
    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.8), dpi=600, sharey=True)
    axes = axes.flatten()

    for ax, year, panel in zip(axes, WET_YEARS, ["(a)", "(b)", "(c)", "(d)"]):
        year_df = h3[h3["Year"] == year].copy()
        obs = year_df[year_df["Model"] == MODEL_ORDER[0]][["Date", "True_Q_shigu"]].drop_duplicates()
        obs = obs.sort_values("Date").reset_index(drop=True)
        day_index = np.arange(1, len(obs) + 1)
        ax.plot(day_index, obs["True_Q_shigu"], color=MODEL_COLORS["Observed runoff"], lw=1.5)

        for model in MODEL_ORDER:
            model_df = year_df[year_df["Model"] == model].sort_values("Date").reset_index(drop=True)
            ax.plot(
                day_index,
                model_df["Pred_Q_shigu"],
                color=MODEL_COLORS[model],
                lw=1.1,
                alpha=0.95,
            )

        ax.set_title(f"{panel} {year}", fontsize=15)
        ax.set_xlabel("Time (d)", fontsize=15)
        ax.set_ylabel("Runoff (m$^3$/s)", fontsize=15)
        ax.set_ylim(500, 6000)
        ax.grid(False)
        ax.tick_params(labelsize=13)

    handles = [plt.Line2D([0], [0], color=MODEL_COLORS["Observed runoff"], lw=2.2)]
    labels = ["Observed runoff"]
    for model in MODEL_ORDER:
        handles.append(plt.Line2D([0], [0], color=MODEL_COLORS[model], lw=2.2))
        labels.append(model)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        frameon=False,
        fontsize=12,
        handlelength=2.6,
        columnspacing=0.9,
        handletextpad=0.35,
    )
    fig.subplots_adjust(top=0.88, left=0.08, right=0.98, bottom=0.08, wspace=0.22, hspace=0.34)

    for ext, kwargs in [
        ("png", {"dpi": 600}),
        ("tif", {"dpi": 600}),
        ("pdf", {}),
    ]:
        fig.savefig(
            OUTPUT_ROOT / f"Fig7_wet_season_hydrographs_H3_corrected.{ext}",
            bbox_inches="tight",
            **kwargs,
        )
    plt.close(fig)


def plot_fig8(wet):
    h3 = wet[wet["Lead time (d)"] == 3].copy()
    data = [
        h3[h3["Model"] == model]["Absolute error"].to_numpy(dtype=np.float64)
        for model in MODEL_ORDER
    ]
    set_plot_style()
    fig, ax = plt.subplots(figsize=(9.2, 7.2), dpi=600)
    bp = ax.boxplot(
        data,
        patch_artist=True,
        showmeans=True,
        widths=0.62,
        whis=1.5,
        meanprops={
            "marker": "s",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 4.2,
        },
        medianprops={"color": "black", "linewidth": 1.2},
        boxprops={"linewidth": 1.2},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
        flierprops={
            "marker": "o",
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "markersize": 3.6,
            "alpha": 1.0,
        },
    )

    for patch, model in zip(bp["boxes"], MODEL_ORDER):
        patch.set_facecolor(MODEL_COLORS[model])
        patch.set_alpha(0.95)

    ax.set_xticklabels(MODEL_ORDER, rotation=25, ha="right", fontsize=14)
    ax.set_ylabel("Absolute error (m$^3$/s)", fontsize=16)
    ax.tick_params(axis="y", labelsize=14)
    max_error = max(float(np.nanmax(values)) for values in data)
    ylim_top = max(3000, math.ceil(max_error / 500.0) * 500)
    ax.set_ylim(0, ylim_top)
    ax.grid(False)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#ff4040", edgecolor="black", label="25%-75%"),
        plt.Line2D([0], [0], color="black", lw=1.2, marker="|", markersize=18, label="1.5xIQR whisker"),
        plt.Line2D([0], [0], color="black", lw=1.2, label="Median line"),
        plt.Line2D([0], [0], color="black", marker="s", markerfacecolor="white", lw=0, markersize=5, label="Mean"),
        plt.Line2D([0], [0], color="black", marker="o", lw=0, markersize=5, label="Outlier"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=5,
        frameon=False,
        fontsize=12,
        handlelength=1.6,
        columnspacing=1.0,
    )
    fig.subplots_adjust(top=0.84, bottom=0.19, left=0.12, right=0.98)

    for ext, kwargs in [
        ("png", {"dpi": 600}),
        ("tif", {"dpi": 600}),
        ("pdf", {}),
    ]:
        fig.savefig(
            OUTPUT_ROOT / f"Fig8_absolute_error_distribution_H3_corrected.{ext}",
            bbox_inches="tight",
            **kwargs,
        )
    plt.close(fig)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_predictions = load_all_predictions()
    wet, overall, yearly = build_wet_season_tables(all_predictions)

    all_predictions.to_csv(OUTPUT_ROOT / "all_test_predictions_corrected.csv", index=False, encoding="utf-8-sig")
    wet.to_csv(OUTPUT_ROOT / "wet_season_predictions_corrected.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(OUTPUT_ROOT / "wet_season_metrics_corrected_full.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(OUTPUT_ROOT / "wet_season_yearly_metrics_corrected.csv", index=False, encoding="utf-8-sig")
    _, rounded = export_table4(overall)
    plot_fig7(wet)
    plot_fig8(wet)

    try:
        with pd.ExcelWriter(OUTPUT_ROOT / "wet_season_summary_corrected.xlsx") as writer:
            rounded.to_excel(writer, sheet_name="Table4_rounded", index=False)
            overall.to_excel(writer, sheet_name="Overall_full", index=False)
            yearly.to_excel(writer, sheet_name="Yearly", index=False)
            wet.to_excel(writer, sheet_name="WetSeasonSamples", index=False)
    except Exception as exc:
        print(f"Skipped xlsx export: {exc}")

    print("Wet-season samples per model and lead time:")
    print(
        wet.groupby(["Lead time (d)", "Model"]).size().unstack("Model").loc[list(HORIZONS), MODEL_ORDER]
    )
    print("\nTable 4 rounded:")
    print(rounded.to_string(index=False))
    print(f"\nOutputs saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
