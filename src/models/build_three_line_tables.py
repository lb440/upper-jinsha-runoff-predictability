import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_shared_utils import FLOOD_MONTHS, FLOOD_YEARS, calculate_metrics, calculate_pbias
from model_shared_config import FEATURE_COLS, FILE_PATH, TARGET_COL, TEST_END_DATE, TIME_COL, TRAIN_END_DATE


MODEL_ORDER = {
    "Persistence": 1,
    "XGBoost": 2,
    "MLP": 3,
    "GRU": 4,
    "LSTM": 5,
    "Transformer": 6,
    "LSTM-Transformer": 7,
}
MODEL_DIRS = {
    "Persistence": "C:/Persistence-{horizon}",
    "XGBoost": "C:/XGBoost-{horizon}",
    "MLP": "C:/MLP-{horizon}",
    "GRU": "C:/GRU-{horizon}",
    "LSTM": "C:/LSTM-{horizon}",
    "Transformer": "C:/Transformer-{horizon}",
    "LSTM-Transformer": "C:/L-T-{horizon}",
}
BOXPLOT_MODEL_ORDER = [
    "Persistence",
    "XGBoost",
    "GRU",
    "MLP",
    "LSTM",
    "Transformer",
    "LSTM-Transformer",
]
TRAIN_TEST_METRICS = ["R", "NSE", "RMSE", "MAE", "MAPE(%)", "Bias", "KGE"]
SUMMARY_METRICS = ["NSE", "RMSE", "MAE", "KGE"]
FLOOD_METRICS = [
    "HighFlowNSE",
    "Top5RMSE",
    "FloodSeasonPBIAS(%)",
    "MeanAbsTimeToPeakError(days)",
    "MeanAbsEventPFE(%)",
    "EventCount",
    "HighFlowSampleCount",
]
ABLATION_GROUPS = {
    "G1_Full": {"desc": "All nine predictors."},
    "G2_Upstream_Only": {"desc": "Upstream station variables only."},
    "G3_Local_Only": {"desc": "Local interval variables only."},
    "G4_Q_Only": {"desc": "Upstream discharge only."},
    "G5_No_Q": {"desc": "All predictors except upstream discharge."},
    "G6_Core_Hydro": {"desc": "Core hydrologic variables."},
    "G7_No_Snow": {"desc": "All predictors except snow variables."},
    "G8_No_TempEvap": {"desc": "All predictors except temperature and evapotranspiration."},
}
COMBO_GROUP_ORDER = {group_name: index + 1 for index, group_name in enumerate(ABLATION_GROUPS)}
COMBO_SHORT_LABELS = {
    group_name: f"C{index + 1}" for index, group_name in enumerate(ABLATION_GROUPS)
}
COMBO_MODEL_DIRS = {
    "Persistence": "C:/Persistence-3(XR)",
    "XGBoost": "C:/XGBoost-3(XR)",
    "MLP": "C:/MLP-3(XR)",
    "GRU": "C:/GRU-3(XR)",
    "LSTM": "C:/LSTM-3(XR)",
    "Transformer": "C:/Transformer-3(XR)",
    "LSTM-Transformer": "C:/L-T-3(XR)",
}
EN_METRIC_SECTION_RE = re.compile(r"^(Train Metrics|Validation Metrics|Test Metrics)\s*$")
METRIC_LINE_RE = re.compile(r"^([^:]+):\s*(.+)$")
OPTIMIZED_PARAMETER_ROWS = [
    {
        "Family": "Feed-forward NN",
        "Model": "MLP",
        "TunedHyperparameters": "batch size; hidden units; learning rate",
        "SearchRange": "batch size {16, 32, 64, 128}; hidden units {32, 64, 128, 256}; learning rate {1e-4, 3e-4, 5e-4, 1e-3}",
    },
    {
        "Family": "Recurrent NN",
        "Model": "LSTM",
        "TunedHyperparameters": "batch size; hidden units; learning rate",
        "SearchRange": "batch size {16, 32, 64, 128}; hidden units {32, 64, 128, 256}; learning rate {1e-4, 3e-4, 5e-4, 1e-3}",
    },
    {
        "Family": "Attention-based NN",
        "Model": "Transformer",
        "TunedHyperparameters": "batch size; d_model; nhead; learning rate",
        "SearchRange": "batch size {16, 32, 64, 128}; d_model {64, 128, 256}; nhead {2, 4, 8, 16}; learning rate {1e-4, 3e-4, 5e-4, 1e-3}; d_model mod nhead = 0",
    },
    {
        "Family": "Hybrid NN",
        "Model": "LSTM-Transformer",
        "TunedHyperparameters": "batch size; d_model; nhead; LSTM hidden units; fusion hidden units; learning rate",
        "SearchRange": "batch size {16, 32, 64, 128}; d_model {64, 128, 256}; nhead {2, 4, 8, 16}; LSTM hidden units {32, 64, 128, 256}; fusion hidden units {64, 128, 256}; learning rate {1e-4, 3e-4, 5e-4, 1e-3}; d_model mod nhead = 0",
    },
    {
        "Family": "Baseline",
        "Model": "Persistence",
        "TunedHyperparameters": "None",
        "SearchRange": "-",
    },
    {
        "Family": "Tree-based ML",
        "Model": "XGBoost",
        "TunedHyperparameters": "n_estimators; max_depth; learning_rate; subsample; column sampling; min_child_weight; reg_alpha; reg_lambda; gamma",
        "SearchRange": "n_estimators 200-800 (step 100); max_depth 2-5; learning_rate 1e-2-8e-2 (log); subsample 0.5-0.85; column sampling 0.3-0.8; min_child_weight 5-30; reg_alpha 1e-3-10 (log); reg_lambda 0.1-100 (log); gamma 1-10",
    },
    {
        "Family": "Recurrent NN",
        "Model": "GRU",
        "TunedHyperparameters": "batch size; hidden units; learning rate",
        "SearchRange": "batch size {16, 32, 64, 128}; hidden units {32, 64, 128, 256}; learning rate {1e-4, 3e-4, 5e-4, 1e-3}; dropout = 0.1 fixed",
    },
]
OPTIMIZED_PARAMETER_FAMILY_ORDER = {
    "Baseline": 1,
    "Tree-based ML": 2,
    "Feed-forward NN": 3,
    "Recurrent NN": 4,
    "Attention-based NN": 5,
    "Hybrid NN": 6,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build train/test, flood-analysis, and H=3 test-set three-line tables."
    )
    parser.add_argument(
        "--base-output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "comparison_tables_seed222"),
    )
    return parser.parse_args()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_metadata_file(model_dir):
    for pattern in ("*run_metadata.json", "*metadata.json", "*.json"):
        for candidate in sorted(Path(model_dir).glob(pattern)):
            try:
                payload = load_json(candidate)
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(payload, dict):
                return candidate, payload
    return None, {}


def extract_compute_device(metadata):
    if "compute_device" in metadata:
        return metadata["compute_device"]
    if "device" in metadata:
        return metadata["device"]
    params = metadata.get("params")
    if isinstance(params, dict):
        return (
            params.get("resolved_compute_device")
            or params.get("requested_compute_device")
            or params.get("device")
            or ""
        )
    return ""


def parse_metrics_txt(metrics_path):
    sections = {}
    current = None
    with Path(metrics_path).open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            section_match = EN_METRIC_SECTION_RE.match(line)
            if section_match:
                current = section_match.group(1).split()[0].lower()
                sections[current] = {}
                continue
            metric_match = METRIC_LINE_RE.match(line)
            if current and metric_match:
                metric_name = metric_match.group(1).strip()
                metric_value = metric_match.group(2).strip()
                try:
                    sections[current][metric_name] = float(metric_value)
                except ValueError:
                    sections[current][metric_name] = metric_value
    return sections


def build_model_metric_record(model_name, horizon):
    model_dir = Path(MODEL_DIRS[model_name].format(horizon=horizon))
    metadata_file, metadata = find_metadata_file(model_dir)
    train_metrics = metadata.get("train_metrics")
    test_metrics = metadata.get("test_metrics")
    metrics_path = model_dir / "metrics.txt"
    if (not train_metrics or not test_metrics) and metrics_path.exists():
        sections = parse_metrics_txt(metrics_path)
        train_metrics = train_metrics or sections.get("train")
        test_metrics = test_metrics or sections.get("test")
    if not train_metrics or not test_metrics:
        raise FileNotFoundError(f"Missing train/test metrics for {model_name} H={horizon}: {model_dir}")

    row = {
        "Model": model_name,
        "LeadTime(d)": horizon,
        "ComputeDevice": extract_compute_device(metadata),
        "SourceDir": str(model_dir),
        "MetadataFile": str(metadata_file) if metadata_file else "",
    }
    for metric in TRAIN_TEST_METRICS:
        row[f"Train_{metric}"] = train_metrics.get(metric)
        row[f"Test_{metric}"] = test_metrics.get(metric)
    if "R2" in test_metrics:
        row["Test_R2"] = test_metrics.get("R2")
    return row


def find_prediction_file(model_name, horizon):
    model_dir = Path(MODEL_DIRS[model_name].format(horizon=horizon))
    candidate_paths = []
    if model_name == "LSTM-Transformer":
        candidate_paths.append(model_dir / f"prediction_test_{horizon}d_no_lag.csv")
    candidate_paths.append(model_dir / "prediction_test.csv")
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing prediction file for {model_name} H={horizon}: {model_dir}")


def read_model_prediction_frame(model_name, horizon):
    prediction_file = find_prediction_file(model_name, horizon)
    prediction_df = pd.read_csv(prediction_file, encoding="utf-8-sig")
    required_columns = {"Date", "True_Q_shigu", "Pred_Q_shigu"}
    missing_columns = required_columns.difference(prediction_df.columns)
    if missing_columns:
        raise ValueError(
            f"Prediction file missing columns for {model_name} H={horizon}: {sorted(missing_columns)}"
        )
    prediction_df = prediction_df[["Date", "True_Q_shigu", "Pred_Q_shigu"]].copy()
    prediction_df["Date"] = pd.to_datetime(prediction_df["Date"])
    return prediction_df


def build_wet_season_metrics_table():
    rows = []
    for horizon in (1, 3, 7, 15):
        for model_name in sorted(MODEL_ORDER, key=lambda item: MODEL_ORDER[item]):
            prediction_df = read_model_prediction_frame(model_name, horizon)
            prediction_df = prediction_df[
                prediction_df["Date"].dt.month.isin(FLOOD_MONTHS)
                & prediction_df["Date"].dt.year.isin(FLOOD_YEARS)
            ].copy()
            y_true = prediction_df["True_Q_shigu"].to_numpy(dtype=float)
            y_pred = prediction_df["Pred_Q_shigu"].to_numpy(dtype=float)
            metrics = calculate_metrics(y_true, y_pred)
            rows.append(
                {
                    "LeadTime(d)": horizon,
                    "Model": model_name,
                    "NSE": metrics["NSE"],
                    "RMSE": metrics["RMSE"],
                    "MAE": metrics["MAE"],
                    "PBIAS(%)": calculate_pbias(y_true, y_pred),
                    "KGE": metrics["KGE"],
                    "ModelOrder": MODEL_ORDER[model_name],
                }
            )
    metrics_df = pd.DataFrame(rows).sort_values(["LeadTime(d)", "ModelOrder"]).reset_index(drop=True)
    metrics_df = metrics_df.drop(columns=["ModelOrder"])
    return format_numeric_columns(metrics_df, digits=4)


def collect_train_test_metrics():
    rows = []
    for model_name in sorted(MODEL_ORDER, key=lambda item: MODEL_ORDER[item]):
        for horizon in (1, 3, 7, 15):
            rows.append(build_model_metric_record(model_name, horizon))
    df = pd.DataFrame(rows)
    df["ModelOrder"] = df["Model"].map(MODEL_ORDER)
    return df.sort_values(["LeadTime(d)", "ModelOrder"]).reset_index(drop=True)


def build_prediction_series_table(horizon, flood_only=False):
    merged_df = None
    for model_name in sorted(MODEL_ORDER, key=lambda item: MODEL_ORDER[item]):
        prediction_df = read_model_prediction_frame(model_name, horizon).rename(
            columns={
                "True_Q_shigu": "Observed_Q_shigu",
                "Pred_Q_shigu": model_name,
            }
        )

        if merged_df is None:
            merged_df = prediction_df
            continue

        merged_df = merged_df.merge(
            prediction_df,
            on="Date",
            how="outer",
            suffixes=("", f"__{model_name}"),
            sort=True,
        )
        observed_candidate_col = f"Observed_Q_shigu__{model_name}"
        comparable_mask = (
            merged_df["Observed_Q_shigu"].notna()
            & merged_df[observed_candidate_col].notna()
        )
        if comparable_mask.any() and not np.allclose(
            merged_df.loc[comparable_mask, "Observed_Q_shigu"].to_numpy(dtype=float),
            merged_df.loc[comparable_mask, observed_candidate_col].to_numpy(dtype=float),
        ):
            raise ValueError(f"Observed values differ across models for H={horizon}: {model_name}")
        merged_df["Observed_Q_shigu"] = merged_df["Observed_Q_shigu"].combine_first(
            merged_df[observed_candidate_col]
        )
        merged_df = merged_df.drop(columns=[observed_candidate_col])

    merged_df = merged_df.sort_values("Date").reset_index(drop=True)
    value_columns = ["Observed_Q_shigu"] + [
        model_name for model_name in sorted(MODEL_ORDER, key=lambda item: MODEL_ORDER[item])
    ]
    if merged_df[value_columns].isna().any().any():
        missing_summary = merged_df[value_columns].isna().sum()
        raise ValueError(
            f"Prediction series contain missing values for H={horizon}: "
            f"{missing_summary[missing_summary > 0].to_dict()}"
        )

    if flood_only:
        merged_df = merged_df[
            merged_df["Date"].dt.month.isin(FLOOD_MONTHS)
            & merged_df["Date"].dt.year.isin(FLOOD_YEARS)
        ].reset_index(drop=True)

    merged_df["Date"] = merged_df["Date"].dt.strftime("%Y-%m-%d")
    return format_numeric_columns(merged_df[["Date"] + value_columns], digits=4)


def build_prediction_series_tables():
    tables = {}
    for horizon in (1, 3, 7, 15):
        tables[f"test_series_H{horizon}"] = build_prediction_series_table(
            horizon=horizon,
            flood_only=False,
        )
        tables[f"flood_series_H{horizon}"] = build_prediction_series_table(
            horizon=horizon,
            flood_only=True,
        )
    for horizon in (3, 7):
        tables[f"wet_season_process_H{horizon}"] = build_prediction_series_table(
            horizon=horizon,
            flood_only=True,
        )
    return tables


def build_boxplot_absolute_error_tables():
    series_df = build_prediction_series_table(horizon=3, flood_only=True)
    series_df["Date"] = pd.to_datetime(series_df["Date"])
    for column in ["Observed_Q_shigu"] + BOXPLOT_MODEL_ORDER:
        series_df[column] = pd.to_numeric(series_df[column], errors="coerce")

    wide_df = pd.DataFrame(
        {
            "Date": series_df["Date"].dt.strftime("%Y-%m-%d"),
            "Year": series_df["Date"].dt.year,
            "Observed_Q_shigu": series_df["Observed_Q_shigu"],
        }
    )
    for model_name in BOXPLOT_MODEL_ORDER:
        wide_df[model_name] = (series_df[model_name] - series_df["Observed_Q_shigu"]).abs()

    long_rows = []
    for model_name in BOXPLOT_MODEL_ORDER:
        model_error = series_df[model_name] - series_df["Observed_Q_shigu"]
        model_abs_error = model_error.abs()
        for date_value, observed, predicted, error_value, abs_error in zip(
            series_df["Date"],
            series_df["Observed_Q_shigu"],
            series_df[model_name],
            model_error,
            model_abs_error,
        ):
            long_rows.append(
                {
                    "Date": date_value.strftime("%Y-%m-%d"),
                    "Year": int(date_value.year),
                    "Model": model_name,
                    "Observed_Q_shigu": observed,
                    "Predicted_Q_shigu": predicted,
                    "Error(m3/s)": error_value,
                    "AbsoluteError(m3/s)": abs_error,
                }
            )
    long_df = pd.DataFrame(long_rows)

    summary_rows = []
    for model_name in BOXPLOT_MODEL_ORDER:
        errors = wide_df[model_name].to_numpy(dtype=float)
        summary_rows.append(
            {
                "Model": model_name,
                "N": int(np.isfinite(errors).sum()),
                "MedianAbsError(m3/s)": np.nanmedian(errors),
                "MeanAbsError(m3/s)": np.nanmean(errors),
                "Q1AbsError(m3/s)": np.nanpercentile(errors, 25),
                "Q3AbsError(m3/s)": np.nanpercentile(errors, 75),
                "IQRAbsError(m3/s)": np.nanpercentile(errors, 75) - np.nanpercentile(errors, 25),
                "P95AbsError(m3/s)": np.nanpercentile(errors, 95),
                "MaxAbsError(m3/s)": np.nanmax(errors),
            }
        )

    return {
        "boxplot_abs_error_H3": format_numeric_columns(wide_df, digits=4),
        "boxplot_abs_error_H3_long": format_numeric_columns(long_df, digits=4),
        "boxplot_abs_error_H3_summary": format_numeric_columns(pd.DataFrame(summary_rows), digits=4),
    }


def format_numeric_columns(df, integer_columns=None, digits=4):
    integer_columns = set(integer_columns or [])
    result = df.copy()
    for column in result.columns:
        if column in integer_columns:
            result[column] = result[column].apply(
                lambda value: "" if pd.isna(value) else int(value)
            )
            continue
        if pd.api.types.is_numeric_dtype(result[column]):
            result[column] = result[column].apply(
                lambda value: "" if pd.isna(value) else round(float(value), digits)
            )
    return result


def build_horizon_train_test_tables(train_test_df):
    tables = {}
    columns = ["Model"]
    for metric in TRAIN_TEST_METRICS:
        columns.append(f"Train_{metric}")
    for metric in TRAIN_TEST_METRICS:
        columns.append(f"Test_{metric}")

    for horizon in (1, 3, 7, 15):
        subset = (
            train_test_df[train_test_df["LeadTime(d)"] == horizon]
            .sort_values("ModelOrder")
            .reset_index(drop=True)
        )
        tables[f"train_test_H{horizon}"] = format_numeric_columns(subset[columns])
    return tables


def build_optimized_parameter_table():
    df = pd.DataFrame(OPTIMIZED_PARAMETER_ROWS)
    df["FamilyOrder"] = df["Family"].map(OPTIMIZED_PARAMETER_FAMILY_ORDER)
    df["ModelOrder"] = df["Model"].map(MODEL_ORDER)
    table = df.sort_values(["FamilyOrder", "ModelOrder"]).reset_index(drop=True)[
        ["Family", "Model", "TunedHyperparameters", "SearchRange"]
    ]
    return table.rename(
        columns={
            "Model": "Model",
            "TunedHyperparameters": "Tuned hyperparameters",
            "SearchRange": "Search range",
        }
    )[["Model", "Tuned hyperparameters", "Search range"]]


def build_optimized_parameter_table_vertical():
    grouped_display = {
        "Persistence": {
            "Tuned hyperparameters": "None",
            "Search range": "-",
        },
        "XGBoost": {
            "Tuned hyperparameters": "trees; depth; lr; subsample; column sampling; child weight; alpha; lambda; gamma",
            "Search range": "200-800; 2-5; 1e-2-8e-2; 0.5-0.85; 0.3-0.8; 5-30; 1e-3-10; 0.1-100; 1-10",
        },
        "MLP / GRU / LSTM": {
            "Tuned hyperparameters": "batch size; hidden units; lr",
            "Search range": "16-128; 32-256; 1e-4-1e-3",
        },
        "Transformer": {
            "Tuned hyperparameters": "batch size; d_model; nhead; lr",
            "Search range": "16-128; 64-256; 2-16; 1e-4-1e-3",
        },
        "LSTM-Transformer": {
            "Tuned hyperparameters": "batch size; d_model; nhead; LSTM units; fusion units; lr",
            "Search range": "16-128; 64-256; 2-16; 32-256; 64-256; 1e-4-1e-3",
        },
    }
    grouped_columns = [
        "Persistence",
        "XGBoost",
        "MLP / GRU / LSTM",
        "Transformer",
        "LSTM-Transformer",
    ]
    matrix = {"Item": ["Tuned hyperparameters", "Search range"]}
    for column_name in grouped_columns:
        matrix[column_name] = [
            grouped_display[column_name]["Tuned hyperparameters"],
            grouped_display[column_name]["Search range"],
        ]
    return pd.DataFrame(matrix, columns=["Item"] + grouped_columns)


def build_performance_summary_table(train_test_df):
    subset = (
        train_test_df.sort_values(["LeadTime(d)", "ModelOrder"])
        .reset_index(drop=True)
        .copy()
    )
    summary = subset[
        [
            "LeadTime(d)",
            "Model",
            "Train_NSE",
            "Train_RMSE",
            "Train_MAE",
            "Train_KGE",
            "Test_NSE",
            "Test_RMSE",
            "Test_MAE",
            "Test_KGE",
        ]
    ].rename(
        columns={
            "LeadTime(d)": "LeadTime(d)",
            "Model": "Models",
            "Train_NSE": "Training_NSE",
            "Train_RMSE": "Training_RMSE(m3/s)",
            "Train_MAE": "Training_MAE(m3/s)",
            "Train_KGE": "Training_KGE",
            "Test_NSE": "Testing_NSE",
            "Test_RMSE": "Testing_RMSE(m3/s)",
            "Test_MAE": "Testing_MAE(m3/s)",
            "Test_KGE": "Testing_KGE",
        }
    )
    return format_numeric_columns(summary, digits=3)


def read_project_csv(file_path=FILE_PATH):
    try:
        data = pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        data = pd.read_csv(file_path, encoding="gbk")
    data.columns = data.columns.str.strip()
    return data


def read_prediction_frame(path):
    return pd.read_csv(path, encoding="utf-8-sig")


def compute_nse(y_true, y_pred):
    numerator = float(np.sum((y_true - y_pred) ** 2))
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denominator == 0:
        return np.nan
    return 1.0 - numerator / denominator


def compute_rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def build_h3_combination_tables():
    rows = []
    for model_name in sorted(MODEL_ORDER, key=lambda item: MODEL_ORDER[item]):
        combo_root = Path(COMBO_MODEL_DIRS[model_name]) / "ablation"
        for group_name, group_meta in ABLATION_GROUPS.items():
            group_dir = combo_root / group_name
            prediction_files = sorted(group_dir.glob("*_prediction_test.csv"))
            if not prediction_files:
                raise FileNotFoundError(f"Missing combination prediction file: {group_dir}")
            prediction_df = read_prediction_frame(prediction_files[0])
            y_true = prediction_df["True_Q_shigu"].to_numpy(dtype=float)
            y_pred = prediction_df["Pred_Q_shigu"].to_numpy(dtype=float)
            metrics = calculate_metrics(y_true, y_pred)
            mae = float(metrics["MAE"])
            rmse = float(metrics["RMSE"])
            rows.append(
                {
                    "Model": model_name,
                    "Code": COMBO_SHORT_LABELS[group_name],
                    "Combination": group_name,
                    "Description": group_meta["desc"],
                    "NSE": float(metrics["NSE"]),
                    "RMSE": rmse,
                    "MAE": mae,
                    "KGE": float(metrics["KGE"]),
                    "PredictionFile": str(prediction_files[0]),
                    "ModelOrder": MODEL_ORDER[model_name],
                    "GroupOrder": COMBO_GROUP_ORDER[group_name],
                }
            )
    combo_df = pd.DataFrame(rows).sort_values(["ModelOrder", "GroupOrder"]).reset_index(drop=True)
    export_df = combo_df[
        ["Model", "Code", "Combination", "Description", "NSE", "RMSE", "MAE", "KGE"]
    ]

    metric_tables = {}
    for metric_name in ("NSE", "MAE", "RMSE", "KGE"):
        matrix_rows = []
        for group_name in ABLATION_GROUPS:
            row = {metric_name: COMBO_SHORT_LABELS[group_name]}
            for model_name in sorted(MODEL_ORDER, key=lambda item: MODEL_ORDER[item]):
                value = combo_df.loc[
                    (combo_df["Model"] == model_name)
                    & (combo_df["Combination"] == group_name),
                    metric_name,
                ].iloc[0]
                row[model_name] = value
            matrix_rows.append(row)
        metric_tables[f"test_H3_combo_{metric_name}"] = format_numeric_columns(
            pd.DataFrame(matrix_rows),
            digits=4,
        )

    legend_rows = []
    for group_name, group_meta in ABLATION_GROUPS.items():
        legend_rows.append(
            {
                "Code": COMBO_SHORT_LABELS[group_name],
                "Combination": group_name,
                "Description": group_meta["desc"],
            }
        )
    metric_tables["test_H3_combo_legend"] = pd.DataFrame(legend_rows)

    return format_numeric_columns(export_df, digits=4), metric_tables


def build_flood_tables(base_output_dir):
    candidate_paths = [
        Path(base_output_dir) / "table5_overall_latest.csv",
        Path.cwd() / "high_flow_tables" / "table5_overall_latest.csv",
        Path(__file__).resolve().parent / "high_flow_tables" / "table5_overall_latest.csv",
    ]
    flood_path = next((path for path in candidate_paths if path.exists()), None)
    if flood_path is None:
        raise FileNotFoundError("Could not locate table5_overall_latest.csv for flood-table export.")
    flood_df = pd.read_csv(flood_path)
    flood_df = flood_df[flood_df["Model"].isin(MODEL_ORDER)].copy()
    flood_df["ModelOrder"] = flood_df["Model"].map(MODEL_ORDER)
    tables = {}
    columns = ["Model"] + FLOOD_METRICS
    for horizon in (1, 3, 7):
        subset = (
            flood_df[flood_df["LeadTime(d)"] == horizon]
            .sort_values("ModelOrder")
            .reset_index(drop=True)
        )
        tables[f"flood_test_H{horizon}"] = format_numeric_columns(
            subset[columns],
            integer_columns={"EventCount", "HighFlowSampleCount"},
        )
    return tables


def dataframe_to_latex_table(df, caption, label):
    table_df = df.fillna("")
    columns = [str(column) for column in table_df.columns]
    rows = [[str(value) for value in row] for row in table_df.astype(str).values.tolist()]
    column_spec = "l" + "r" * (len(columns) - 1)
    header_line = " & ".join(columns) + r" \\"
    body_lines = [" & ".join(row) + r" \\" for row in rows]
    latex = "\n".join(
        [
            f"\\begin{{tabular}}{{{column_spec}}}",
            "\\toprule",
            header_line,
            "\\midrule",
            *body_lines,
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    return (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{latex}\n"
        "\\end{table}\n"
    )


def write_csv_tables(table_map, output_dir):
    for name, df in table_map.items():
        df.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")


def make_excel_sheet_name(name, used_names):
    sheet_name = re.sub(r"[\[\]\:\*\?\/\\]", "_", str(name)).strip() or "Sheet"
    sheet_name = sheet_name[:31]
    candidate = sheet_name
    suffix = 1
    while candidate in used_names:
        suffix_text = f"_{suffix}"
        candidate = f"{sheet_name[:31 - len(suffix_text)]}{suffix_text}"
        suffix += 1
    used_names.add(candidate)
    return candidate


def write_xlsx_workbooks(table_map, output_dir):
    from openpyxl.styles import Alignment, Font

    workbook_names = [
        "three_line_tables.xlsx",
        "three_line_tables_with_optimized_params.xlsx",
    ]
    for workbook_name in workbook_names:
        output_path = output_dir / workbook_name
        used_names = set()
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for table_name, df in table_map.items():
                sheet_name = make_excel_sheet_name(table_name, used_names)
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                worksheet = writer.sheets[sheet_name]
                worksheet.freeze_panes = "A2"
                for cell in worksheet[1]:
                    cell.font = Font(bold=True)
                    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                for column_cells in worksheet.columns:
                    max_length = max(
                        len(str(cell.value)) if cell.value is not None else 0
                        for cell in column_cells
                    )
                    column_letter = column_cells[0].column_letter
                    worksheet.column_dimensions[column_letter].width = min(max(max_length + 2, 10), 42)
                for row in worksheet.iter_rows():
                    for cell in row:
                        cell.alignment = Alignment(vertical="center", wrap_text=True)


def write_tex_bundle(table_map, output_dir):
    captions = {
        "optimized_parameters_summary": "Publication-style summary of tuned hyperparameters, candidate values or ranges, and model-specific notes.",
        "optimized_parameters_summary_vertical": "Transposed publication-style summary of tuned hyperparameters and search ranges across models.",
        "train_test_H1": "Lead time 1 day: train and test metrics for all models.",
        "train_test_H3": "Lead time 3 days: train and test metrics for all models.",
        "train_test_H7": "Lead time 7 days: train and test metrics for all models.",
        "train_test_H15": "Lead time 15 days: train and test metrics for all models.",
        "flood_test_H1": "Lead time 1 day: flood-analysis test metrics for all models.",
        "flood_test_H3": "Lead time 3 days: flood-analysis test metrics for all models.",
        "flood_test_H7": "Lead time 7 days: flood-analysis test metrics for all models.",
        "汛期": "Wet-season pooled performance from June to October during 2017-2020.",
        "test_H3_combinations": "Lead time 3 days: combination experiment test metrics (NSE, RMSE, MAE, KGE).",
        "test_H3_combo_NSE": "Lead time 3 days: combination experiment NSE matrix by model and combination.",
        "test_H3_combo_MAE": "Lead time 3 days: combination experiment MAE matrix by model and combination.",
        "test_H3_combo_RMSE": "Lead time 3 days: combination experiment RMSE matrix by model and combination.",
        "test_H3_combo_KGE": "Lead time 3 days: combination experiment KGE matrix by model and combination.",
        "test_H3_combo_legend": "Lead time 3 days: combination-code legend for the matrix tables.",
        "boxplot_abs_error_H3": "Lead time 3 days: pooled wet-season absolute-error data for boxplot drawing.",
        "boxplot_abs_error_H3_long": "Lead time 3 days: long-format pooled wet-season error data for boxplot drawing.",
        "boxplot_abs_error_H3_summary": "Lead time 3 days: summary statistics of pooled wet-season absolute errors.",
        "performance_summary_all_leads": "Lead times 1, 3, 7, and 15 days: training and testing performance summary (NSE, RMSE, MAE, KGE).",
    }
    sections = []
    for name, df in table_map.items():
        if (
            name.startswith("test_series_")
            or name.startswith("flood_series_")
            or name.startswith("wet_season_process_")
            or name.startswith("boxplot_abs_error_")
        ):
            continue
        sections.append(
            dataframe_to_latex_table(
                df,
                caption=captions.get(name, name),
                label=f"tab:{name}",
            )
        )
    (output_dir / "three_line_tables.tex").write_text(
        "\n\n".join(sections),
        encoding="utf-8",
    )


def write_manifest(table_map, output_dir):
    manifest = {
        "sheet_order": list(table_map.keys()),
        "summary_sheet": "performance_summary_all_leads",
        "wet_season_metrics_sheet": "汛期",
        "combo_sheet": "test_H3_combinations",
        "wet_season_process_sheets": ["wet_season_process_H3", "wet_season_process_H7"],
        "combo_metric_sheets": ["test_H3_combo_NSE", "test_H3_combo_MAE", "test_H3_combo_RMSE", "test_H3_combo_KGE"],
        "boxplot_sheets": ["boxplot_abs_error_H3", "boxplot_abs_error_H3_long", "boxplot_abs_error_H3_summary"],
    }
    (output_dir / "tables_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_readme(table_map, output_dir):
    lines = [
        "Three-line table outputs",
        "",
        "1. train_test_H1/H3/H7/H15: all models, train and test metrics by lead time.",
        "2. performance_summary_all_leads: lead-time 1/3/7/15 training/testing performance summary (NSE, RMSE, MAE, KGE).",
        "3. optimized_parameters_summary: tuned hyperparameters, candidate values or ranges, and model-specific notes.",
        "4. flood_test_H1/H3/H7: all models, flood-analysis test metrics by lead time.",
        "5. 汛期: pooled wet-season performance from June to October during 2017-2020 (NSE, RMSE, MAE, PBIAS, KGE).",
        "6. test_H3_combinations: 3-day combination experiment metrics (NSE, RMSE, MAE, KGE).",
        "7. test_H3_combo_NSE/MAE/RMSE/KGE: 3-day combination experiment metric matrices in the manuscript-style layout.",
        "8. test_H3_combo_legend: matrix code legend (C1-C8 to G1-G8).",
        "9. test_series_H1/H3/H7/H15: full test-period time, observed values, and all-model predictions.",
        "10. flood_series_H1/H3/H7/H15: flood-season time, observed values, and all-model predictions.",
        "11. wet_season_process_H3/H7: flood-season process data for manuscript hydrographs.",
        "12. boxplot_abs_error_H3: Origin-ready wide table for the 3-day wet-season absolute-error boxplot.",
        "13. boxplot_abs_error_H3_long: long-format error data for audit or alternative plotting.",
        "14. boxplot_abs_error_H3_summary: summary statistics used to verify the boxplot distributions.",
        "",
        "Files:",
    ]
    for name in table_map:
        lines.append(f"- {name}.csv")
    lines.append("- tables_manifest.json")
    lines.append("- three_line_tables.tex")
    (output_dir / "README.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    base_output_dir = Path(args.base_output_dir)
    output_dir = base_output_dir / "three_line_tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_test_df = collect_train_test_metrics()
    train_test_tables = build_horizon_train_test_tables(train_test_df)
    optimized_parameter_table = build_optimized_parameter_table()
    optimized_parameter_vertical_table = build_optimized_parameter_table_vertical()
    flood_tables = build_flood_tables(base_output_dir)
    wet_season_metrics_table = build_wet_season_metrics_table()
    prediction_series_tables = build_prediction_series_tables()
    boxplot_error_tables = build_boxplot_absolute_error_tables()
    combo_h3_table, combo_metric_tables = build_h3_combination_tables()
    performance_summary_table = build_performance_summary_table(train_test_df)

    table_map = {}
    table_map["performance_summary_all_leads"] = performance_summary_table
    table_map["optimized_parameters_summary"] = optimized_parameter_table
    table_map["optimized_parameters_summary_vertical"] = optimized_parameter_vertical_table
    table_map["汛期"] = wet_season_metrics_table
    table_map.update(train_test_tables)
    table_map.update(flood_tables)
    table_map.update(prediction_series_tables)
    table_map.update(boxplot_error_tables)
    table_map["test_H3_combinations"] = combo_h3_table
    table_map.update(combo_metric_tables)

    write_csv_tables(table_map, output_dir)
    write_xlsx_workbooks(table_map, output_dir)
    write_tex_bundle(table_map, output_dir)
    write_manifest(table_map, output_dir)
    write_readme(table_map, output_dir)

    print(f"Output directory: {output_dir}")
    print(f"Tables exported: {len(table_map)}")
    for name in table_map:
        print(name)


if __name__ == "__main__":
    main()
