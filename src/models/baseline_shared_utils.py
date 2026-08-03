from dataclasses import dataclass
from pathlib import Path
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
try:
    from scipy.signal import find_peaks as scipy_find_peaks
except ImportError:
    scipy_find_peaks = None

from model_shared_config import (
    FEATURE_COLS,
    FILE_PATH,
    LOCAL_COLS,
    TARGET_COL,
    TEST_END_DATE,
    TIME_COL,
    TRAIN_END_DATE,
    UPSTREAM_COLS,
    VAL_END_DATE,
    get_split_indices,
)


DEFAULT_RANDOM_SEED = 222
DEFAULT_KEYS_TO_WRITE = ["R", "NSE", "RMSE", "MAE", "MAPE(%)", "Bias", "KGE"]
FLOOD_MONTHS = (6, 7, 8, 9, 10)
FLOOD_YEARS = (2017, 2018, 2019, 2020)
FLOOD_HIGH_FLOW_QUANTILE = 0.95
FLOOD_EVENT_QUANTILE = 0.90
FLOOD_EVENT_MIN_DISTANCE = 7

ABLATION_GROUPS = {
    "G1_Full": {
        "upstream": ["Qz", "Pz", "Tz", "Ez", "Sz"],
        "local": ["Pi", "Ti", "Ei", "Si"],
        "desc": "All nine predictors.",
    },
    "G2_Upstream_Only": {
        "upstream": ["Qz", "Pz", "Tz", "Ez", "Sz"],
        "local": [],
        "desc": "Upstream station variables only.",
    },
    "G3_Local_Only": {
        "upstream": [],
        "local": ["Pi", "Ti", "Ei", "Si"],
        "desc": "Local interval variables only.",
    },
    "G4_Q_Only": {
        "upstream": ["Qz"],
        "local": [],
        "desc": "Upstream discharge only.",
    },
    "G5_No_Q": {
        "upstream": ["Pz", "Tz", "Ez", "Sz"],
        "local": ["Pi", "Ti", "Ei", "Si"],
        "desc": "All predictors except upstream discharge.",
    },
    "G6_Core_Hydro": {
        "upstream": ["Qz", "Pz"],
        "local": ["Pi"],
        "desc": "Core hydrologic variables.",
    },
    "G7_No_Snow": {
        "upstream": ["Qz", "Pz", "Tz", "Ez"],
        "local": ["Pi", "Ti", "Ei"],
        "desc": "All predictors except snow variables.",
    },
    "G8_No_TempEvap": {
        "upstream": ["Qz", "Pz", "Sz"],
        "local": ["Pi", "Si"],
        "desc": "All predictors except temperature and evapotranspiration.",
    },
}
VALID_GROUPS = tuple(ABLATION_GROUPS.keys())


@dataclass
class SupervisedSplits:
    feature_cols: list
    kim: int
    horizon: int
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    history_train: np.ndarray
    history_val: np.ndarray
    history_test: np.ndarray
    time_train: pd.Series
    time_val: pd.Series
    time_test: pd.Series


def ensure_dir(path):
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def prefixed_name(prefix, filename):
    return f"{prefix}_{filename}" if prefix else filename


def default_model_output_dir(model_name, horizon):
    return os.path.normpath(f"C:/{model_name}-{horizon}")


def default_bo_output_dir(model_name, horizon):
    return os.path.normpath(f"C:/{model_name}-BO-{horizon}")


def default_ablation_output_dir(model_name, horizon):
    return os.path.normpath(f"C:/{model_name}-{horizon}(XR)")


def set_plot_style():
    plt.rcParams["font.sans-serif"] = ["SimSun", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def seed_numpy(seed):
    np.random.seed(seed)


def save_json(path, payload):
    path = Path(path)
    ensure_dir(path.parent)
    def to_json_safe(value):
        if isinstance(value, dict):
            return {str(key): to_json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [to_json_safe(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        return value
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_json_safe(payload), handle, indent=4, ensure_ascii=False)


def save_pickle(path, payload):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def load_best_params(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_project_csv(file_path=FILE_PATH):
    try:
        data = pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        data = pd.read_csv(file_path, encoding="gbk")
    data.columns = data.columns.str.strip()
    return data


def load_base_dataframe(file_path=FILE_PATH, required_feature_cols=None):
    required_feature_cols = required_feature_cols or FEATURE_COLS
    data = read_project_csv(file_path=file_path)
    required_cols = [TIME_COL, TARGET_COL] + list(required_feature_cols)
    data = data.dropna(subset=required_cols).reset_index(drop=True)
    data[TIME_COL] = pd.to_datetime(data[TIME_COL])
    data = data[
        (data[TIME_COL] >= "2006-01-01")
        & (data[TIME_COL] <= str(TEST_END_DATE.date()))
    ].reset_index(drop=True)

    data[TARGET_COL] = data[TARGET_COL].clip(lower=0)
    for col in required_feature_cols:
        if "Q" in col:
            data[col] = data[col].clip(lower=0)
    return data


def build_supervised_splits(feature_cols, kim, horizon, file_path=FILE_PATH):
    data = load_base_dataframe(file_path=file_path, required_feature_cols=feature_cols)
    features = data[list(feature_cols)].values.astype(np.float32)
    runoff = data[TARGET_COL].values.astype(np.float32)
    time = data[TIME_COL]

    valid_samples = len(runoff) - kim - horizon + 1
    if valid_samples <= 0:
        raise ValueError("The configured kim/horizon leaves no valid samples.")

    X, y, histories, out_idx = [], [], [], []
    for i in range(valid_samples):
        X.append(features[i : i + kim])
        histories.append(runoff[i : i + kim])
        y.append(runoff[i + kim + horizon - 1])
        out_idx.append(i + kim + horizon - 1)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    histories = np.asarray(histories, dtype=np.float32)
    data_time = time.iloc[out_idx].reset_index(drop=True)

    train_idx, val_idx, test_idx = get_split_indices(
        data_time,
        TRAIN_END_DATE,
        VAL_END_DATE,
        TEST_END_DATE,
    )

    return SupervisedSplits(
        feature_cols=list(feature_cols),
        kim=kim,
        horizon=horizon,
        X_train=X[train_idx],
        X_val=X[val_idx],
        X_test=X[test_idx],
        y_train=y[train_idx],
        y_val=y[val_idx],
        y_test=y[test_idx],
        history_train=histories[train_idx],
        history_val=histories[val_idx],
        history_test=histories[test_idx],
        time_train=data_time.iloc[train_idx].reset_index(drop=True),
        time_val=data_time.iloc[val_idx].reset_index(drop=True),
        time_test=data_time.iloc[test_idx].reset_index(drop=True),
    )


def scale_sequence_splits(splits):
    feature_dim = len(splits.feature_cols)
    x_train = np.zeros_like(splits.X_train)
    x_val = np.zeros_like(splits.X_val)
    x_test = np.zeros_like(splits.X_test)
    x_scalers = {}

    for idx in range(feature_dim):
        scaler = MinMaxScaler()
        scaler.fit(splits.X_train[:, :, idx].reshape(-1, 1))
        x_train[:, :, idx] = scaler.transform(
            splits.X_train[:, :, idx].reshape(-1, 1)
        ).reshape(splits.X_train.shape[0], splits.kim)
        x_val[:, :, idx] = scaler.transform(
            splits.X_val[:, :, idx].reshape(-1, 1)
        ).reshape(splits.X_val.shape[0], splits.kim)
        x_test[:, :, idx] = scaler.transform(
            splits.X_test[:, :, idx].reshape(-1, 1)
        ).reshape(splits.X_test.shape[0], splits.kim)
        x_scalers[splits.feature_cols[idx]] = scaler

    y_scaler = MinMaxScaler()
    y_train = y_scaler.fit_transform(splits.y_train)
    y_val = y_scaler.transform(splits.y_val)
    y_test = y_scaler.transform(splits.y_test)

    return {
        "X_train": x_train,
        "X_val": x_val,
        "X_test": x_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "x_scalers": x_scalers,
        "y_scaler": y_scaler,
    }


def flatten_and_scale_splits(splits):
    x_train = splits.X_train.reshape(splits.X_train.shape[0], -1)
    x_val = splits.X_val.reshape(splits.X_val.shape[0], -1)
    x_test = splits.X_test.reshape(splits.X_test.shape[0], -1)

    x_scaler = MinMaxScaler()
    x_train = x_scaler.fit_transform(x_train)
    x_val = x_scaler.transform(x_val)
    x_test = x_scaler.transform(x_test)

    return {
        "X_train": x_train,
        "X_val": x_val,
        "X_test": x_test,
        "x_scaler": x_scaler,
    }


def serialize_minmax_scaler(scaler):
    return {
        "feature_range": tuple(scaler.feature_range),
        "min_": scaler.min_.tolist(),
        "scale_": scaler.scale_.tolist(),
        "data_min_": scaler.data_min_.tolist(),
        "data_max_": scaler.data_max_.tolist(),
        "data_range_": scaler.data_range_.tolist(),
        "n_features_in_": int(scaler.n_features_in_),
        "n_samples_seen_": int(scaler.n_samples_seen_),
    }


def safe_corrcoef(true_vals, pred_vals):
    if len(true_vals) < 2:
        return np.nan
    if np.allclose(np.std(true_vals), 0) or np.allclose(np.std(pred_vals), 0):
        return np.nan
    return float(np.corrcoef(true_vals, pred_vals)[0, 1])


def calculate_nse(true_vals, pred_vals):
    true_vals = np.asarray(true_vals, dtype=np.float64)
    pred_vals = np.asarray(pred_vals, dtype=np.float64)
    denominator = np.sum((true_vals - np.mean(true_vals)) ** 2)
    if np.isclose(denominator, 0):
        return np.nan
    numerator = np.sum((true_vals - pred_vals) ** 2)
    return float(1 - numerator / denominator)


def calculate_kge(true_vals, pred_vals):
    true_vals = np.asarray(true_vals, dtype=np.float64)
    pred_vals = np.asarray(pred_vals, dtype=np.float64)
    corr = safe_corrcoef(true_vals, pred_vals)
    if np.isnan(corr):
        return np.nan
    std_true = np.std(true_vals)
    mean_true = np.mean(true_vals)
    alpha = np.std(pred_vals) / (std_true + 1e-8)
    beta = np.mean(pred_vals) / (mean_true + 1e-8)
    return float(1 - np.sqrt((corr - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def calculate_metrics(true_vals, pred_vals):
    true_vals = np.asarray(true_vals, dtype=np.float64)
    pred_vals = np.asarray(pred_vals, dtype=np.float64)
    eps = 1e-8
    mse = mean_squared_error(true_vals, pred_vals)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_vals, pred_vals)
    r2 = r2_score(true_vals, pred_vals)
    nse = calculate_nse(true_vals, pred_vals)
    bias = np.mean(pred_vals - true_vals)
    mape = np.mean(np.abs((pred_vals - true_vals) / (true_vals + eps))) * 100
    kge = calculate_kge(true_vals, pred_vals)
    corr = safe_corrcoef(true_vals, pred_vals)
    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2),
        "R": float(corr) if not np.isnan(corr) else np.nan,
        "NSE": float(nse) if not np.isnan(nse) else np.nan,
        "MAPE(%)": float(mape),
        "Bias": float(bias),
        "KGE": float(kge) if not np.isnan(kge) else np.nan,
    }


def build_prediction_frame(dates, y_true, y_pred):
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(dates).reset_index(drop=True),
            "True_Q_shigu": np.asarray(y_true, dtype=np.float64).reshape(-1),
            "Pred_Q_shigu": np.asarray(y_pred, dtype=np.float64).reshape(-1),
        }
    ).assign(Error=lambda df: df["Pred_Q_shigu"] - df["True_Q_shigu"])


def write_metrics_file(
    path,
    train_metrics,
    val_metrics,
    test_metrics,
    keys=None,
    extra_sections=None,
):
    keys = keys or DEFAULT_KEYS_TO_WRITE
    extra_sections = extra_sections or []
    path = Path(path)
    ensure_dir(path.parent)

    with path.open("w", encoding="utf-8") as handle:
        for title, payload in extra_sections:
            handle.write(f"{title}\n")
            if isinstance(payload, dict):
                for key, value in payload.items():
                    handle.write(f"{key}: {value}\n")
            else:
                handle.write(f"{payload}\n")
            handle.write("\n")

        for title, metrics in (
            ("Train Metrics", train_metrics),
            ("Validation Metrics", val_metrics),
            ("Test Metrics", test_metrics),
        ):
            handle.write(f"{title}\n")
            for key in keys:
                value = metrics.get(key, np.nan)
                if isinstance(value, float):
                    handle.write(f"{key}: {value:.4f}\n")
                else:
                    handle.write(f"{key}: {value}\n")
            handle.write("\n")


def plot_training_curves(path, train_losses, val_losses, title):
    if train_losses is None or val_losses is None:
        return
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="Train Loss")
    ax.plot(val_losses, label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_prediction_series(path, dates, y_true, y_pred, title):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, y_true, label="Observed", linewidth=1.5)
    ax.plot(dates, y_pred, label="Predicted", linewidth=1.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Runoff")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_scatter(path, y_true, y_pred, title):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.45, s=12, color="steelblue")
    max_val = max(np.max(y_true), np.max(y_pred)) * 1.05
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=1.0)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_optimization_history(path, trials_df, title, best_value):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(trials_df["number"], trials_df["value"], "o-", markersize=5)
    ax.axhline(y=best_value, color="r", linestyle="--", label=f"Best: {best_value:.4f}")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Validation NSE")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def calculate_pbias(true_vals, pred_vals):
    true_vals = np.asarray(true_vals, dtype=np.float64)
    pred_vals = np.asarray(pred_vals, dtype=np.float64)
    return float(
        np.sum(pred_vals - true_vals) / (np.sum(true_vals) + 1e-8) * 100
    )


def _fallback_find_peaks(values, threshold, min_distance):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return np.asarray([], dtype=int)

    candidate_indices = []
    if len(values) == 1:
        if values[0] >= threshold:
            candidate_indices.append(0)
    else:
        if values[0] >= threshold and values[0] >= values[1]:
            candidate_indices.append(0)
        for idx in range(1, len(values) - 1):
            if (
                values[idx] >= threshold
                and values[idx] >= values[idx - 1]
                and values[idx] >= values[idx + 1]
            ):
                candidate_indices.append(idx)
        if values[-1] >= threshold and values[-1] >= values[-2]:
            candidate_indices.append(len(values) - 1)

    if not candidate_indices:
        return np.asarray([], dtype=int)

    selected = []
    for idx in sorted(candidate_indices, key=lambda item: values[item], reverse=True):
        if all(abs(idx - kept) >= min_distance for kept in selected):
            selected.append(idx)
    return np.asarray(sorted(selected), dtype=int)


def identify_peak_indices(values, threshold, min_distance):
    values = np.asarray(values, dtype=np.float64)
    min_distance = max(1, int(min_distance))
    if len(values) == 0:
        return np.asarray([], dtype=int)

    if scipy_find_peaks is not None:
        peaks, _ = scipy_find_peaks(
            values,
            height=float(threshold),
            distance=min_distance,
        )
        return peaks.astype(int)
    return _fallback_find_peaks(values, threshold, min_distance)


def detect_flood_events(
    true_vals,
    pred_vals,
    dates,
    horizon,
    peak_threshold,
    min_peak_distance=FLOOD_EVENT_MIN_DISTANCE,
):
    true_vals = np.asarray(true_vals, dtype=np.float64)
    pred_vals = np.asarray(pred_vals, dtype=np.float64)
    dates = pd.to_datetime(dates).reset_index(drop=True)

    columns = [
        "EventID",
        "Year",
        "ObsPeakDate",
        "PredPeakDate",
        "ObsPeakFlow",
        "PredPeakFlow",
        "PeakFlowAbsError",
        "PeakFlowRelError(%)",
        "TimeToPeakError(days)",
        "MatchWindowStart",
        "MatchWindowEnd",
    ]
    peak_indices = identify_peak_indices(true_vals, peak_threshold, min_peak_distance)
    if len(peak_indices) == 0:
        return pd.DataFrame(columns=columns)

    events = []
    for event_id, obs_idx in enumerate(peak_indices, start=1):
        window_start = max(0, obs_idx - int(horizon))
        window_end = min(len(pred_vals) - 1, obs_idx + int(horizon))
        pred_idx = window_start + int(np.argmax(pred_vals[window_start : window_end + 1]))

        obs_peak_date = dates.iloc[obs_idx]
        pred_peak_date = dates.iloc[pred_idx]
        obs_peak_flow = float(true_vals[obs_idx])
        pred_peak_flow = float(pred_vals[pred_idx])
        events.append(
            {
                "EventID": event_id,
                "Year": int(obs_peak_date.year),
                "ObsPeakDate": str(obs_peak_date.date()),
                "PredPeakDate": str(pred_peak_date.date()),
                "ObsPeakFlow": obs_peak_flow,
                "PredPeakFlow": pred_peak_flow,
                "PeakFlowAbsError": pred_peak_flow - obs_peak_flow,
                "PeakFlowRelError(%)": float(
                    (pred_peak_flow - obs_peak_flow) / (obs_peak_flow + 1e-8) * 100
                ),
                "TimeToPeakError(days)": int((pred_peak_date - obs_peak_date).days),
                "MatchWindowStart": str(dates.iloc[window_start].date()),
                "MatchWindowEnd": str(dates.iloc[window_end].date()),
            }
        )
    return pd.DataFrame(events, columns=columns)


def summarize_event_metrics(events_df):
    if events_df is None or events_df.empty:
        return {
            "EventCount": 0,
            "MeanAbsTimeToPeakError(days)": np.nan,
            "MeanSignedTimeToPeakError(days)": np.nan,
            "MeanAbsEventPFE(%)": np.nan,
            "MeanSignedEventPFE(%)": np.nan,
        }

    time_errors = events_df["TimeToPeakError(days)"].to_numpy(dtype=np.float64)
    pfes = events_df["PeakFlowRelError(%)"].to_numpy(dtype=np.float64)
    return {
        "EventCount": int(len(events_df)),
        "MeanAbsTimeToPeakError(days)": float(np.mean(np.abs(time_errors))),
        "MeanSignedTimeToPeakError(days)": float(np.mean(time_errors)),
        "MeanAbsEventPFE(%)": float(np.mean(np.abs(pfes))),
        "MeanSignedEventPFE(%)": float(np.mean(pfes)),
    }


def calculate_flood_metrics(
    true_vals,
    pred_vals,
    dates,
    horizon=1,
    high_flow_threshold=None,
    event_peak_threshold=None,
    min_peak_distance=FLOOD_EVENT_MIN_DISTANCE,
    return_events=False,
):
    true_vals = np.asarray(true_vals, dtype=np.float64)
    pred_vals = np.asarray(pred_vals, dtype=np.float64)
    dates = pd.to_datetime(dates).reset_index(drop=True)

    if len(true_vals) == 0:
        empty_metrics = {
            "Samples": 0,
            "HighFlowThreshold(Q95)": np.nan,
            "HighFlowSampleCount": 0,
            "HighFlowNSE": np.nan,
            "Top5RMSE": np.nan,
            "FloodSeasonPBIAS(%)": np.nan,
            "PeakThreshold(Q90)": np.nan,
            "PeakMinDistance(days)": int(min_peak_distance),
        }
        empty_metrics.update(summarize_event_metrics(pd.DataFrame()))
        if return_events:
            return empty_metrics, pd.DataFrame()
        return empty_metrics

    high_flow_threshold = float(
        high_flow_threshold
        if high_flow_threshold is not None
        else np.quantile(true_vals, FLOOD_HIGH_FLOW_QUANTILE)
    )
    event_peak_threshold = float(
        event_peak_threshold
        if event_peak_threshold is not None
        else np.quantile(true_vals, FLOOD_EVENT_QUANTILE)
    )

    high_flow_mask = true_vals >= high_flow_threshold
    high_flow_nse = (
        calculate_nse(true_vals[high_flow_mask], pred_vals[high_flow_mask])
        if np.any(high_flow_mask)
        else np.nan
    )
    top5_rmse = (
        float(np.sqrt(mean_squared_error(true_vals[high_flow_mask], pred_vals[high_flow_mask])))
        if np.any(high_flow_mask)
        else np.nan
    )

    metrics = calculate_metrics(true_vals, pred_vals)
    metrics.update(
        {
            "Samples": int(len(true_vals)),
            "HighFlowThreshold(Q95)": high_flow_threshold,
            "HighFlowSampleCount": int(np.sum(high_flow_mask)),
            "HighFlowNSE": high_flow_nse,
            "Top5RMSE": top5_rmse,
            "FloodSeasonPBIAS(%)": calculate_pbias(true_vals, pred_vals),
            "PeakThreshold(Q90)": event_peak_threshold,
            "PeakMinDistance(days)": int(min_peak_distance),
        }
    )

    events_df = detect_flood_events(
        true_vals,
        pred_vals,
        dates,
        horizon=horizon,
        peak_threshold=event_peak_threshold,
        min_peak_distance=min_peak_distance,
    )
    metrics.update(summarize_event_metrics(events_df))

    if return_events:
        return metrics, events_df
    return metrics


def _format_metric_value(value, digits=4):
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return "nan" if np.isnan(value) else f"{float(value):.{digits}f}"
    return str(value)


def _write_flood_metrics_report(path, title, metrics, years=None):
    years = years or []
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{title}\n")
        if years:
            handle.write(f"Years: {list(years)}\n")
        handle.write(
            f"Flood months: {list(FLOOD_MONTHS)} | High-flow threshold = Q95 | "
            f"Event threshold = Q90 | Min event spacing = {FLOOD_EVENT_MIN_DISTANCE} days\n\n"
        )
        sections = [
            (
                "Whole Wet-Season Metrics",
                ["NSE", "KGE", "R", "RMSE", "MAE", "MAPE(%)", "Bias", "FloodSeasonPBIAS(%)"],
            ),
            (
                "High-Flow Metrics",
                ["HighFlowThreshold(Q95)", "HighFlowSampleCount", "HighFlowNSE", "Top5RMSE"],
            ),
            (
                "Event Metrics",
                [
                    "PeakThreshold(Q90)",
                    "PeakMinDistance(days)",
                    "EventCount",
                    "MeanAbsTimeToPeakError(days)",
                    "MeanSignedTimeToPeakError(days)",
                    "MeanAbsEventPFE(%)",
                    "MeanSignedEventPFE(%)",
                ],
            ),
        ]
        for section_title, keys in sections:
            handle.write(f"{section_title}\n")
            for key in keys:
                handle.write(f"{key}: {_format_metric_value(metrics.get(key, np.nan))}\n")
            handle.write("\n")


def save_flood_analysis(
    output_dir,
    prediction_df,
    horizon,
    prefix="",
    dir_name=None,
    run_label=None,
):
    flood_dir_name = dir_name or prefixed_name(prefix, "flood_analysis")
    flood_dir = ensure_dir(Path(output_dir) / flood_dir_name)
    run_label = run_label or "Model"

    prediction_df = prediction_df.copy()
    prediction_df["Date"] = pd.to_datetime(prediction_df["Date"])
    flood_df = prediction_df[
        prediction_df["Date"].dt.month.isin(FLOOD_MONTHS)
        & prediction_df["Date"].dt.year.isin(FLOOD_YEARS)
    ].copy()

    if flood_df.empty:
        note_path = flood_dir / prefixed_name(prefix, f"flood_summary_H{horizon}d.txt")
        note_path.write_text(
            "No 2017-2020 flood-season samples were found for the configured months.\n",
            encoding="utf-8",
        )
        return flood_dir

    flood_df["Year"] = flood_df["Date"].dt.year
    flood_df["Month"] = flood_df["Date"].dt.month
    flood_df["Error"] = flood_df["Pred_Q_shigu"] - flood_df["True_Q_shigu"]
    flood_df["Abs_Error"] = flood_df["Error"].abs()
    flood_df["Rel_Error(%)"] = (
        flood_df["Error"] / (flood_df["True_Q_shigu"] + 1e-8) * 100
    )

    global_q95 = float(np.quantile(flood_df["True_Q_shigu"].values, FLOOD_HIGH_FLOW_QUANTILE))
    global_q90 = float(np.quantile(flood_df["True_Q_shigu"].values, FLOOD_EVENT_QUANTILE))
    flood_df["HighFlowFlag(Q95)"] = flood_df["True_Q_shigu"] >= global_q95
    flood_df.to_csv(
        flood_dir / prefixed_name(prefix, f"flood_season_predictions_H{horizon}d.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    yearly_rows = []
    event_frames = []
    for year in FLOOD_YEARS:
        year_df = flood_df[flood_df["Year"] == year].reset_index(drop=True)
        if year_df.empty:
            continue

        metrics, events_df = calculate_flood_metrics(
            year_df["True_Q_shigu"].values,
            year_df["Pred_Q_shigu"].values,
            year_df["Date"],
            horizon=horizon,
            high_flow_threshold=global_q95,
            event_peak_threshold=global_q90,
            return_events=True,
        )
        yearly_rows.append({"Scope": str(year), **metrics})
        if not events_df.empty:
            event_frames.append(events_df)
            events_df.to_csv(
                flood_dir / prefixed_name(prefix, f"flood_events_{year}_H{horizon}d.csv"),
                index=False,
                encoding="utf-8-sig",
            )

        year_df.to_csv(
            flood_dir / prefixed_name(prefix, f"flood_pred_{year}_H{horizon}d.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        _write_flood_metrics_report(
            flood_dir / prefixed_name(prefix, f"flood_metrics_{year}_H{horizon}d.txt"),
            title=f"{run_label} flood-season report for {year}",
            metrics=metrics,
            years=[year],
        )

        set_plot_style()
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(year_df["Date"], year_df["True_Q_shigu"], label="Observed", linewidth=1.4)
        ax.plot(year_df["Date"], year_df["Pred_Q_shigu"], label="Predicted", linewidth=1.4)
        if not events_df.empty:
            obs_dates = pd.to_datetime(events_df["ObsPeakDate"])
            pred_dates = pd.to_datetime(events_df["PredPeakDate"])
            ax.scatter(
                obs_dates,
                events_df["ObsPeakFlow"],
                color="black",
                marker="^",
                s=46,
                zorder=5,
                label="Observed peaks",
            )
            ax.scatter(
                pred_dates,
                events_df["PredPeakFlow"],
                color="red",
                marker="v",
                s=46,
                zorder=5,
                label="Predicted peaks",
            )
        ax.set_xlabel("Date")
        ax.set_ylabel("Runoff")
        ax.set_title(
            f"{run_label} {year} flood season (H={horizon}d, "
            f"HighFlowNSE={_format_metric_value(metrics['HighFlowNSE'])}, "
            f"Mean|PFE|={_format_metric_value(metrics['MeanAbsEventPFE(%)'], digits=2)}%)"
        )
        ax.legend()
        ax.grid(alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        per_year_pred_path = flood_dir / prefixed_name(prefix, f"flood_pred_{year}_H{horizon}d.png")
        per_year_legacy_path = flood_dir / prefixed_name(prefix, f"flood_plot_{year}_H{horizon}d.png")
        fig.savefig(per_year_pred_path, dpi=300)
        fig.savefig(per_year_legacy_path, dpi=300)
        plt.close(fig)

    all_events_df = (
        pd.concat(event_frames, ignore_index=True)
        if event_frames
        else pd.DataFrame(
            columns=[
                "EventID",
                "Year",
                "ObsPeakDate",
                "PredPeakDate",
                "ObsPeakFlow",
                "PredPeakFlow",
                "PeakFlowAbsError",
                "PeakFlowRelError(%)",
                "TimeToPeakError(days)",
                "MatchWindowStart",
                "MatchWindowEnd",
            ]
        )
    )
    all_events_df.to_csv(
        flood_dir / prefixed_name(prefix, f"flood_events_H{horizon}d.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    overall_metrics = calculate_metrics(
        flood_df["True_Q_shigu"].values,
        flood_df["Pred_Q_shigu"].values,
    )
    overall_metrics.update(
        {
            "Samples": int(len(flood_df)),
            "HighFlowThreshold(Q95)": global_q95,
            "HighFlowSampleCount": int(flood_df["HighFlowFlag(Q95)"].sum()),
            "HighFlowNSE": calculate_nse(
                flood_df.loc[flood_df["HighFlowFlag(Q95)"], "True_Q_shigu"].values,
                flood_df.loc[flood_df["HighFlowFlag(Q95)"], "Pred_Q_shigu"].values,
            ),
            "Top5RMSE": float(
                np.sqrt(
                    mean_squared_error(
                        flood_df.loc[flood_df["HighFlowFlag(Q95)"], "True_Q_shigu"].values,
                        flood_df.loc[flood_df["HighFlowFlag(Q95)"], "Pred_Q_shigu"].values,
                    )
                )
            ),
            "FloodSeasonPBIAS(%)": calculate_pbias(
                flood_df["True_Q_shigu"].values,
                flood_df["Pred_Q_shigu"].values,
            ),
            "PeakThreshold(Q90)": global_q90,
            "PeakMinDistance(days)": FLOOD_EVENT_MIN_DISTANCE,
        }
    )
    overall_metrics.update(summarize_event_metrics(all_events_df))

    summary_rows = [{"Scope": "Overall", **overall_metrics}] + yearly_rows
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        flood_dir / prefixed_name(prefix, f"high_flow_summary_H{horizon}d.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    _write_flood_metrics_report(
        flood_dir / prefixed_name(prefix, f"overall_flood_metrics_H{horizon}d.txt"),
        title=f"{run_label} flood-season overall report",
        metrics=overall_metrics,
        years=FLOOD_YEARS,
    )
    _write_flood_metrics_report(
        flood_dir / prefixed_name(prefix, f"flood_summary_H{horizon}d.txt"),
        title=f"{run_label} flood-season overall report",
        metrics=overall_metrics,
        years=FLOOD_YEARS,
    )

    set_plot_style()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    year_labels = [row["Scope"] for row in yearly_rows]
    metric_panels = [
        ("High-flow NSE", "HighFlowNSE"),
        ("Top-5% RMSE", "Top5RMSE"),
        ("Flood-season PBIAS (%)", "FloodSeasonPBIAS(%)"),
        ("Mean |TPE| (days)", "MeanAbsTimeToPeakError(days)"),
        ("Mean |event PFE| (%)", "MeanAbsEventPFE(%)"),
    ]
    for ax, (title, column_name) in zip(axes, metric_panels):
        values = [row.get(column_name, np.nan) for row in yearly_rows]
        ax.bar(year_labels, values, color="steelblue", alpha=0.85)
        ax.set_title(title)
        ax.grid(alpha=0.25, axis="y")
    axes[-1].axis("off")
    fig.suptitle(f"{run_label} flood-season high-flow summary (H={horizon}d)", fontsize=13)
    fig.tight_layout()
    summary_png = flood_dir / prefixed_name(prefix, f"flood_summary_H{horizon}d.png")
    summary_plot_png = flood_dir / prefixed_name(prefix, f"flood_summary_plot_H{horizon}d.png")
    fig.savefig(summary_png, dpi=300)
    fig.savefig(summary_plot_png, dpi=300)
    plt.close(fig)

    if not all_events_df.empty:
        set_plot_style()
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        ax.scatter(
            all_events_df["ObsPeakFlow"],
            all_events_df["PredPeakFlow"],
            color="firebrick",
            alpha=0.75,
            s=36,
        )
        max_val = (
            max(
                float(all_events_df["ObsPeakFlow"].max()),
                float(all_events_df["PredPeakFlow"].max()),
            )
            * 1.05
        )
        ax.plot([0, max_val], [0, max_val], "k--", linewidth=1.0)
        ax.set_xlabel("Observed event peak flow")
        ax.set_ylabel("Predicted event peak flow")
        ax.set_title(f"{run_label} event peaks (H={horizon}d)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(
            flood_dir / prefixed_name(prefix, f"peak_scatter_H{horizon}d.png"),
            dpi=300,
        )
        plt.close(fig)

    return flood_dir


def save_prediction_outputs(
    output_dir,
    prefix,
    run_label,
    horizon,
    time_test,
    y_test_true,
    y_test_pred,
    train_metrics,
    val_metrics,
    test_metrics,
    train_losses=None,
    val_losses=None,
    extra_sections=None,
):
    output_dir = ensure_dir(output_dir)
    prediction_df = build_prediction_frame(time_test, y_test_true, y_test_pred)
    prediction_df.to_csv(
        output_dir / prefixed_name(prefix, "prediction_test.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    write_metrics_file(
        output_dir / prefixed_name(prefix, "metrics.txt"),
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        extra_sections=extra_sections,
    )

    if train_losses is not None and val_losses is not None:
        plot_training_curves(
            output_dir / prefixed_name(prefix, "train_val_loss.png"),
            train_losses,
            val_losses,
            f"{run_label} Training Curve",
        )

    plot_prediction_series(
        output_dir / prefixed_name(prefix, "test_prediction.png"),
        time_test,
        y_test_true,
        y_test_pred,
        f"{run_label} Test Prediction (H={horizon}d, NSE={test_metrics['NSE']:.4f})",
    )
    plot_scatter(
        output_dir / prefixed_name(prefix, "test_scatter.png"),
        y_test_true,
        y_test_pred,
        f"{run_label} Test Scatter (R={test_metrics['R']:.4f})",
    )
    save_flood_analysis(
        output_dir,
        prediction_df,
        horizon=horizon,
        prefix=prefix,
        run_label=run_label,
    )
    return prediction_df


def choose_ablation_group(group_name=None):
    if group_name:
        if group_name not in VALID_GROUPS:
            raise ValueError(f"Unknown group: {group_name}")
        return group_name

    print("=" * 72)
    print("No --group argument was provided. Choose an ablation group:")
    print("=" * 72)
    for idx, key in enumerate(VALID_GROUPS, start=1):
        group = ABLATION_GROUPS[key]
        feature_count = len(group["upstream"]) + len(group["local"])
        print(f"{idx:>2}. {key:<20} ({feature_count} features) - {group['desc']}")
    print("=" * 72)

    while True:
        raw = input("Enter the group number or name: ").strip()
        if raw.isdigit():
            index = int(raw) - 1
            if 0 <= index < len(VALID_GROUPS):
                return VALID_GROUPS[index]
        elif raw in VALID_GROUPS:
            return raw
        print("Invalid input. Please try again.")


def get_group_feature_cols(group_name):
    group_cfg = ABLATION_GROUPS[group_name]
    return list(group_cfg["upstream"] + group_cfg["local"])


def get_group_metadata(group_name):
    group_cfg = ABLATION_GROUPS[group_name]
    return {
        "group_name": group_name,
        "description": group_cfg["desc"],
        "feature_cols": get_group_feature_cols(group_name),
        "upstream_cols": list(group_cfg["upstream"]),
        "local_cols": list(group_cfg["local"]),
    }


def summarize_split_sizes(splits):
    return {
        "train_samples": int(len(splits.y_train)),
        "val_samples": int(len(splits.y_val)),
        "test_samples": int(len(splits.y_test)),
    }


def select_required_feature_cols(group_name=None):
    if group_name is None:
        return list(FEATURE_COLS)
    return get_group_feature_cols(group_name)


def all_feature_cols():
    return list(UPSTREAM_COLS + LOCAL_COLS)
