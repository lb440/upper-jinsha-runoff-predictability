import copy
import json
import math
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset


warnings.filterwarnings("ignore")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = Path(os.environ.get(
    "RUNOFF_DATA_PATH", str(REPOSITORY_ROOT / "data" / "example_model_input_synthetic.csv")
))
OUTPUT_ROOT = Path(os.environ.get(
    'FIXED_SHAP_OUTPUT_ROOT', str(REPOSITORY_ROOT / 'outputs' / 'fixed_shap')
))
MODEL_ROOT = OUTPUT_ROOT / "fixed_architecture_models"

TIME_COL = "Date"
TARGET_COL = "Q_shigu"
FEATURE_COLS = ["Qz", "Pz", "Tz", "Ez", "Sz", "Pi", "Ti", "Ei", "Si"]
FAST_FEATURES = ["Qz", "Pz", "Pi"]
SLOW_SEASONAL_FEATURES = ["Sz", "Si", "Ez", "Ei", "Tz", "Ti"]

HORIZONS = [1, 3, 7, 15]
TRAIN_END_DATE = pd.Timestamp("2014-12-31")
VAL_END_DATE = pd.Timestamp("2016-12-31")
TEST_END_DATE = pd.Timestamp("2020-12-31")

FIXED_PARAMS = {
    "batch_size": 16,
    "d_model": 128,
    "nhead": 4,
    "lstm_hidden": 256,
    "fusion_hidden": 128,
    "lr": 0.0001,
    "dropout": 0.1,
    "dropout_fc": 0.1,
    "dim_feedforward": 256,
    "kim": 90,
    "random_seed": 222,
}

MAX_EPOCHS = 100
PATIENCE = 10
MIN_DELTA = 1e-5
WEIGHT_DECAY = 1e-4
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5
SCHEDULER_MIN_LR = 1e-6
N_BACKGROUND = int(os.environ.get("SHAP_N_BACKGROUND", "200"))
N_EXPLAIN = int(os.environ.get("SHAP_N_EXPLAIN", "400"))
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available() and os.environ.get("SHAP_DEVICE", "cuda").lower() != "cpu"
    else "cpu"
)


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False


def read_csv_auto(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk")


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


def restore_minmax_scaler(state):
    scaler = MinMaxScaler(feature_range=tuple(state["feature_range"]))
    scaler.min_ = np.asarray(state["min_"], dtype=np.float64)
    scaler.scale_ = np.asarray(state["scale_"], dtype=np.float64)
    scaler.data_min_ = np.asarray(state["data_min_"], dtype=np.float64)
    scaler.data_max_ = np.asarray(state["data_max_"], dtype=np.float64)
    scaler.data_range_ = np.asarray(state["data_range_"], dtype=np.float64)
    scaler.n_features_in_ = int(state["n_features_in_"])
    scaler.n_samples_seen_ = int(state.get("n_samples_seen_", 0))
    return scaler


def normalize_sequences(x_data, x_scalers):
    norm = np.zeros_like(x_data, dtype=np.float32)
    sample_count, sequence_length, _ = x_data.shape
    for idx, feature_name in enumerate(FEATURE_COLS):
        norm[:, :, idx] = x_scalers[feature_name].transform(
            x_data[:, :, idx].reshape(-1, 1)
        ).reshape(sample_count, sequence_length)
    return norm


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class HybridLSTMTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        lstm_hidden,
        d_model,
        nhead,
        dim_feedforward,
        fusion_hidden,
        dropout,
        dropout_fc,
        max_len,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.lstm_norm = nn.LayerNorm(lstm_hidden)
        self.input_proj = nn.Linear(lstm_hidden, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.predict_head = nn.Sequential(
            nn.Linear(d_model, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_fc),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_norm(lstm_out)
        trans_in = self.input_proj(lstm_out)
        trans_in = self.pos_encoder(trans_in)
        trans_out = self.transformer(trans_in)
        return self.predict_head(trans_out[:, -1, :])


def calculate_metrics(true_vals, pred_vals):
    true = np.asarray(true_vals, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred_vals, dtype=np.float64).reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    mae = float(mean_absolute_error(true, pred))
    r2 = float(r2_score(true, pred))
    r = float(np.corrcoef(true, pred)[0, 1])
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    nse = float(1 - ss_res / ss_tot) if ss_tot != 0 else float("nan")
    mape = float(np.mean(np.abs((true - pred) / (true + 1e-8))) * 100)
    bias = float(np.mean(pred - true))
    alpha_kge = np.std(pred) / (np.std(true) + 1e-8)
    beta_kge = np.mean(pred) / (np.mean(true) + 1e-8)
    kge = float(1 - np.sqrt((r - 1) ** 2 + (alpha_kge - 1) ** 2 + (beta_kge - 1) ** 2))
    return {
        "R": r,
        "NSE": nse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE(%)": mape,
        "Bias": bias,
        "KGE": kge,
        "R2": r2,
    }


def build_samples(data, horizon):
    kim = FIXED_PARAMS["kim"]
    data = data.dropna(subset=[TIME_COL, TARGET_COL] + FEATURE_COLS).reset_index(drop=True)
    data[TIME_COL] = pd.to_datetime(data[TIME_COL])
    data = data[
        (data[TIME_COL] >= "2006-01-01") & (data[TIME_COL] <= "2020-12-31")
    ].reset_index(drop=True)
    data[TARGET_COL] = data[TARGET_COL].clip(lower=0)
    for col in FEATURE_COLS:
        if "Q" in col:
            data[col] = data[col].clip(lower=0)

    time = data[TIME_COL]
    features = data[FEATURE_COLS].values.astype(np.float32)
    runoff = data[TARGET_COL].values.astype(np.float32)

    valid_samples = len(runoff) - kim - horizon + 1
    x_samples, y_samples, out_idx = [], [], []
    for i in range(valid_samples):
        forecast_time = i + kim - 1
        target_idx = forecast_time + horizon
        x_samples.append(features[i : i + kim])
        y_samples.append(runoff[target_idx])
        out_idx.append(target_idx)

    x_arr = np.asarray(x_samples, dtype=np.float32)
    y_arr = np.asarray(y_samples, dtype=np.float32).reshape(-1, 1)
    data_time = time.iloc[out_idx].reset_index(drop=True)

    train_idx = np.where(data_time <= TRAIN_END_DATE)[0]
    val_idx = np.where((data_time > TRAIN_END_DATE) & (data_time <= VAL_END_DATE))[0]
    test_idx = np.where((data_time > VAL_END_DATE) & (data_time <= TEST_END_DATE))[0]
    return x_arr, y_arr, data_time, train_idx, val_idx, test_idx


def create_model():
    return HybridLSTMTransformer(
        input_dim=len(FEATURE_COLS),
        lstm_hidden=FIXED_PARAMS["lstm_hidden"],
        d_model=FIXED_PARAMS["d_model"],
        nhead=FIXED_PARAMS["nhead"],
        dim_feedforward=FIXED_PARAMS["dim_feedforward"],
        fusion_hidden=FIXED_PARAMS["fusion_hidden"],
        dropout=FIXED_PARAMS["dropout"],
        dropout_fc=FIXED_PARAMS["dropout_fc"],
        max_len=FIXED_PARAMS["kim"],
    )


def prepare_split(data, horizon):
    x_all, y_all, data_time, train_idx, val_idx, test_idx = build_samples(data.copy(), horizon)
    x_train, x_val, x_test = x_all[train_idx], x_all[val_idx], x_all[test_idx]
    y_train, y_val, y_test = y_all[train_idx], y_all[val_idx], y_all[test_idx]

    x_scalers = {}
    for idx, feature_name in enumerate(FEATURE_COLS):
        scaler = MinMaxScaler()
        scaler.fit(x_train[:, :, idx].reshape(-1, 1))
        x_scalers[feature_name] = scaler

    x_train_norm = normalize_sequences(x_train, x_scalers)
    x_val_norm = normalize_sequences(x_val, x_scalers)
    x_test_norm = normalize_sequences(x_test, x_scalers)

    y_scaler = MinMaxScaler()
    y_train_norm = y_scaler.fit_transform(y_train)
    y_val_norm = y_scaler.transform(y_val)
    y_test_norm = y_scaler.transform(y_test)

    return {
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "x_train_norm": x_train_norm,
        "x_val_norm": x_val_norm,
        "x_test_norm": x_test_norm,
        "y_train_norm": y_train_norm,
        "y_val_norm": y_val_norm,
        "y_test_norm": y_test_norm,
        "x_scalers": x_scalers,
        "y_scaler": y_scaler,
        "data_time": data_time,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }


def evaluate_model(model, split):
    model.eval()
    y_scaler = split["y_scaler"]
    with torch.no_grad():
        pred_train_norm = model(torch.FloatTensor(split["x_train_norm"]).to(DEVICE)).cpu().numpy()
        pred_val_norm = model(torch.FloatTensor(split["x_val_norm"]).to(DEVICE)).cpu().numpy()
        pred_test_norm = model(torch.FloatTensor(split["x_test_norm"]).to(DEVICE)).cpu().numpy()
    pred_train = np.clip(y_scaler.inverse_transform(pred_train_norm).reshape(-1), 0, None)
    pred_val = np.clip(y_scaler.inverse_transform(pred_val_norm).reshape(-1), 0, None)
    pred_test = np.clip(y_scaler.inverse_transform(pred_test_norm).reshape(-1), 0, None)
    return {
        "train_pred": pred_train,
        "val_pred": pred_val,
        "test_pred": pred_test,
        "train_metrics": calculate_metrics(split["y_train"].reshape(-1), pred_train),
        "val_metrics": calculate_metrics(split["y_val"].reshape(-1), pred_val),
        "test_metrics": calculate_metrics(split["y_test"].reshape(-1), pred_test),
    }


def save_checkpoint(horizon, model, split, evaluation, best_val_loss):
    model_dir = MODEL_ROOT / ("L-T-%d" % horizon)
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / ("best_lstm_transformer_fixed_H%dd.pth" % horizon)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_type": "HybridLSTMTransformer_Fixed_Architecture",
        "input_dim": len(FEATURE_COLS),
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "horizon": horizon,
        "kim": FIXED_PARAMS["kim"],
        "random_seed": FIXED_PARAMS["random_seed"],
        "batch_size": FIXED_PARAMS["batch_size"],
        "learning_rate": FIXED_PARAMS["lr"],
        "d_model": FIXED_PARAMS["d_model"],
        "nhead": FIXED_PARAMS["nhead"],
        "lstm_hidden": FIXED_PARAMS["lstm_hidden"],
        "fusion_hidden": FIXED_PARAMS["fusion_hidden"],
        "dim_feedforward": FIXED_PARAMS["dim_feedforward"],
        "dropout": FIXED_PARAMS["dropout"],
        "dropout_fc": FIXED_PARAMS["dropout_fc"],
        "x_scalers": {
            name: serialize_minmax_scaler(split["x_scalers"][name]) for name in FEATURE_COLS
        },
        "y_scaler": serialize_minmax_scaler(split["y_scaler"]),
        "train_metrics": evaluation["train_metrics"],
        "val_metrics": evaluation["val_metrics"],
        "test_metrics": evaluation["test_metrics"],
        "best_val_loss": float(best_val_loss),
    }
    with open(checkpoint_path, "wb") as f:
        torch.save(checkpoint, f)
    return checkpoint_path


def load_checkpoint(checkpoint_path):
    with open(checkpoint_path, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")
    model = create_model().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def train_fixed_model(data, horizon):
    checkpoint_path = MODEL_ROOT / ("L-T-%d" % horizon) / ("best_lstm_transformer_fixed_H%dd.pth" % horizon)
    split = prepare_split(data, horizon)
    if checkpoint_path.exists() and os.environ.get("FORCE_RETRAIN", "0") != "1":
        model, checkpoint = load_checkpoint(checkpoint_path)
        x_scalers = {name: restore_minmax_scaler(checkpoint["x_scalers"][name]) for name in FEATURE_COLS}
        split["x_scalers"] = x_scalers
        split["y_scaler"] = restore_minmax_scaler(checkpoint["y_scaler"])
        split["x_train_norm"] = normalize_sequences(split["x_train"], x_scalers)
        split["x_val_norm"] = normalize_sequences(split["x_val"], x_scalers)
        split["x_test_norm"] = normalize_sequences(split["x_test"], x_scalers)
        evaluation = evaluate_model(model, split)
        return model, split, checkpoint_path, evaluation, checkpoint.get("best_val_loss", np.nan)

    seed_everything(FIXED_PARAMS["random_seed"])
    model = create_model().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=FIXED_PARAMS["lr"],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=SCHEDULER_MIN_LR,
        verbose=False,
    )

    x_train_tensor = torch.FloatTensor(split["x_train_norm"])
    y_train_tensor = torch.FloatTensor(split["y_train_norm"])
    train_loader = DataLoader(
        TensorDataset(x_train_tensor, y_train_tensor),
        batch_size=FIXED_PARAMS["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(FIXED_PARAMS["random_seed"]),
    )
    x_val_tensor = torch.FloatTensor(split["x_val_norm"]).to(DEVICE)
    y_val_tensor = torch.FloatTensor(split["y_val_norm"]).to(DEVICE)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item() * x_batch.shape[0]

        train_loss = running_loss / split["x_train"].shape[0]
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(x_val_tensor), y_val_tensor).item()
        scheduler.step(val_loss)

        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        print(
            "H%dd epoch %03d train=%.6f val=%.6f best=%.6f"
            % (horizon, epoch + 1, train_loss, val_loss, best_val_loss),
            flush=True,
        )
        if no_improve >= PATIENCE:
            break

    model.load_state_dict(best_state)
    model.eval()
    evaluation = evaluate_model(model, split)
    checkpoint_path = save_checkpoint(horizon, model, split, evaluation, best_val_loss)
    return model, split, checkpoint_path, evaluation, best_val_loss


def save_prediction_and_metrics(horizon, split, evaluation, checkpoint_path, best_val_loss):
    h_dir = OUTPUT_ROOT / ("H%dd" % horizon)
    h_dir.mkdir(parents=True, exist_ok=True)
    time_test = split["data_time"].iloc[split["test_idx"]].reset_index(drop=True)
    prediction_df = pd.DataFrame(
        {
            "Date": time_test,
            "Observed": split["y_test"].reshape(-1),
            "Predicted": evaluation["test_pred"],
            "Error": evaluation["test_pred"] - split["y_test"].reshape(-1),
        }
    )
    prediction_df.to_csv(h_dir / ("prediction_test_H%dd.csv" % horizon), index=False, encoding="utf-8-sig")
    metadata = {
        "data_path": str(DATA_PATH),
        "checkpoint_path": str(checkpoint_path),
        "horizon": horizon,
        "fixed_architecture": FIXED_PARAMS,
        "train_end_date": str(TRAIN_END_DATE.date()),
        "val_end_date": str(VAL_END_DATE.date()),
        "test_end_date": str(TEST_END_DATE.date()),
        "train_samples": int(split["x_train"].shape[0]),
        "validation_samples": int(split["x_val"].shape[0]),
        "test_samples": int(split["x_test"].shape[0]),
        "best_val_loss": float(best_val_loss),
        "train_metrics": evaluation["train_metrics"],
        "val_metrics": evaluation["val_metrics"],
        "test_metrics": evaluation["test_metrics"],
    }
    with open(h_dir / ("fixed_model_metadata_H%dd.json" % horizon), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def run_shap(horizon, model, split, checkpoint_path, evaluation):
    h_dir = OUTPUT_ROOT / ("H%dd" % horizon)
    h_dir.mkdir(parents=True, exist_ok=True)
    background_data = torch.FloatTensor(split["x_train_norm"][:N_BACKGROUND]).to(DEVICE)
    x_explain_np = split["x_test_norm"][:N_EXPLAIN]
    x_explain = torch.FloatTensor(x_explain_np).to(DEVICE)

    print(
        "H%dd SHAP on %s: background=%d explain=%d"
        % (horizon, DEVICE, background_data.shape[0], x_explain.shape[0]),
        flush=True,
    )
    model.eval()
    explainer = shap.GradientExplainer(model, background_data)
    shap_values = explainer.shap_values(x_explain)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 4:
        shap_values = shap_values[..., 0]
    if shap_values.ndim != 3:
        raise ValueError("Unexpected SHAP shape for H%dd: %s" % (horizon, shap_values.shape))

    lag_feature_abs = np.mean(np.abs(shap_values), axis=0)
    feature_lag_abs = lag_feature_abs[::-1, :].T
    lag_cols = [str(i) for i in range(FIXED_PARAMS["kim"])]
    pd.DataFrame(feature_lag_abs, index=FEATURE_COLS, columns=lag_cols).to_csv(
        h_dir / ("SHAP_feature_lag_mean_abs_matrix_H%dd.csv" % horizon),
        encoding="utf-8-sig",
    )

    shap_values_agg = np.mean(shap_values, axis=1)
    shap_importance = np.mean(np.abs(shap_values_agg), axis=0)
    pd.DataFrame({"Feature": FEATURE_COLS, "Mean_abs_SHAP": shap_importance}).sort_values(
        "Mean_abs_SHAP", ascending=False
    ).to_csv(h_dir / ("shap_feature_importance_H%dd.csv" % horizon), index=False, encoding="utf-8-sig")

    x_explain_agg = np.mean(x_explain_np, axis=1)
    np.savez(
        h_dir / ("shap_results_H%dd.npz" % horizon),
        shap_values_raw=shap_values,
        shap_values_agg=shap_values_agg,
        shap_importance=shap_importance,
        feature_names=np.asarray(FEATURE_COLS),
        feature_lag_mean_abs_shap=feature_lag_abs,
        lag_labels=np.asarray(lag_cols),
        x_explain_agg=x_explain_agg,
    )

    shap_metadata = {
        "data_path": str(DATA_PATH),
        "checkpoint_path": str(checkpoint_path),
        "horizon": horizon,
        "kim": FIXED_PARAMS["kim"],
        "feature_cols": FEATURE_COLS,
        "random_seed": FIXED_PARAMS["random_seed"],
        "fixed_architecture": FIXED_PARAMS,
        "explainer": "shap.GradientExplainer",
        "background_dataset": "first training samples after chronological split and training-set normalization",
        "background_samples": int(background_data.shape[0]),
        "explained_dataset": "first test samples after chronological split and training-set normalization",
        "explained_test_samples": int(x_explain.shape[0]),
        "input_tensor_shape": list(x_explain.shape),
        "shap_values_shape": list(shap_values.shape),
        "device": str(DEVICE),
        "test_metrics": evaluation["test_metrics"],
    }
    with open(h_dir / ("shap_run_metadata_H%dd.json" % horizon), "w", encoding="utf-8") as f:
        json.dump(shap_metadata, f, indent=2)

    importance_by_feature = dict(zip(FEATURE_COLS, shap_importance))
    fast_raw = float(sum(importance_by_feature[f] for f in FAST_FEATURES))
    slow_raw = float(sum(importance_by_feature[f] for f in SLOW_SEASONAL_FEATURES))
    total_raw = fast_raw + slow_raw
    return {
        "Lead time (d)": horizon,
        "Fast-response group": fast_raw / total_raw,
        "Slow/seasonal group": slow_raw / total_raw,
        "Fast-response raw Mean_abs_SHAP sum": fast_raw,
        "Slow/seasonal raw Mean_abs_SHAP sum": slow_raw,
        "Total raw Mean_abs_SHAP sum": total_raw,
    }


def configure_old_style():
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 13,
        }
    )


def normalize_columns(values):
    values = np.asarray(values, dtype=np.float64)
    vmin = np.nanmin(values, axis=0)
    vmax = np.nanmax(values, axis=0)
    denom = vmax - vmin
    denom[denom == 0] = 1.0
    return (values - vmin) / denom


def plot_beeswarm_panel(ax, horizon, panel_label):
    h_dir = OUTPUT_ROOT / ("H%dd" % horizon)
    result = np.load(h_dir / ("shap_results_H%dd.npz" % horizon), allow_pickle=True)
    shap_values = np.asarray(result["shap_values_agg"], dtype=np.float64)
    feature_values = normalize_columns(np.asarray(result["x_explain_agg"], dtype=np.float64))
    feature_names = [str(x) for x in result["feature_names"]]
    importance = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(importance)[::-1]
    ordered_features = [feature_names[i] for i in order]
    y_positions = np.arange(len(ordered_features))

    ax_top = ax.twiny()
    ax_top.barh(
        y_positions,
        importance[order],
        height=0.75,
        color="#C9E5FA",
        edgecolor="none",
        zorder=0,
    )
    ax_top.set_xlim(0, np.nanmax(importance) * 1.08)
    ax_top.set_xlabel("Mean Shapley Value (Feature Importance)", labelpad=4)
    ax_top.tick_params(axis="x", direction="out", labelsize=11, pad=1)
    ax_top.spines["right"].set_visible(False)
    ax_top.spines["bottom"].set_visible(False)

    cmap = LinearSegmentedColormap.from_list("shap_feature_value", ["#1585E5", "#8552CC", "#F0065A"])
    for row, feature_idx in enumerate(order):
        x = shap_values[:, feature_idx]
        c = feature_values[:, feature_idx]
        jitter = 0.16 * np.sin(np.arange(x.shape[0], dtype=np.float64) * 2.399 + row)
        y = np.full_like(x, y_positions[row], dtype=np.float64) + jitter
        ax.scatter(x, y, c=c, cmap=cmap, s=7, alpha=0.85, linewidths=0, rasterized=True)

    ax.axvline(0, color="#777777", linewidth=0.9)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(ordered_features, fontsize=13)
    ax.invert_yaxis()
    ax.set_xlabel("Shapley Value Contribution (Bee Swarm)", fontsize=13)
    ax.tick_params(axis="x", labelsize=11)
    ax.grid(axis="y", color="#EAEAEA", linestyle=":", linewidth=0.5)
    ax.text(-0.16, 1.08, panel_label, transform=ax.transAxes, fontsize=15, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.05)
    cbar.set_label("Feature Value", fontsize=12)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])
    cbar.ax.tick_params(labelsize=11)


def plot_lag_panel(ax, horizon, panel_label):
    h_dir = OUTPUT_ROOT / ("H%dd" % horizon)
    result = np.load(h_dir / ("shap_results_H%dd.npz" % horizon), allow_pickle=True)
    matrix = np.asarray(result["feature_lag_mean_abs_shap"], dtype=np.float64)
    feature_names = [str(x) for x in result["feature_names"]]
    cmap = LinearSegmentedColormap.from_list("shap_abs_lightblue_blue", ["#EAF7FF", "#B7DCF4", "#4AB0F5", "#008BFB"])
    vmax = float(np.nanpercentile(matrix, 99))

    im = ax.imshow(matrix, cmap=cmap, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax)
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=12)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 89])
    ax.set_xticklabels(["0", "10", "20", "30", "40", "50", "60", "70", "80", "89"], fontsize=11)
    ax.set_xlabel("Lag days", fontsize=13)
    ax.set_title("Lag-specific SHAP importance", fontsize=13)
    ax.text(-0.18, 1.08, panel_label, transform=ax.transAxes, fontsize=15, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.012)
    cbar.set_label("Mean |SHAP value|", fontsize=12)
    cbar.ax.tick_params(labelsize=11)


def plot_four_lead_figures():
    configure_old_style()
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.2), dpi=600)
    for ax, horizon, label in zip(axes.flat, HORIZONS, panel_labels):
        plot_beeswarm_panel(ax, horizon, label)
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.92, wspace=0.38, hspace=0.54)
    fig.savefig(OUTPUT_ROOT / "SHAP_beeswarm_feature_importance_1_3_7_15d.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT_ROOT / "SHAP_beeswarm_feature_importance_1_3_7_15d.tif", dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT_ROOT / "SHAP_beeswarm_feature_importance_1_3_7_15d.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.0), dpi=600)
    for ax, horizon, label in zip(axes.flat, HORIZONS, panel_labels):
        plot_lag_panel(ax, horizon, label)
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.92, wspace=0.35, hspace=0.55)
    fig.savefig(OUTPUT_ROOT / "SHAP_lag_specific_importance_1_3_7_15d.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT_ROOT / "SHAP_lag_specific_importance_1_3_7_15d.tif", dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT_ROOT / "SHAP_lag_specific_importance_1_3_7_15d.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_group_contributions(group_df):
    configure_old_style()
    plot_df = group_df.sort_values("Lead time (d)")
    x = np.arange(len(plot_df))
    fast_pct = plot_df["Fast-response group"].values * 100
    slow_pct = plot_df["Slow/seasonal group"].values * 100

    fig, ax = plt.subplots(figsize=(6.8, 5.6), dpi=600)
    ax.bar(x, fast_pct, width=0.72, color="#f26063", label="Fast-response (Qz, Pz, Pi)")
    ax.bar(
        x,
        slow_pct,
        width=0.72,
        bottom=fast_pct,
        color="#4a8cdf",
        label="Slow/seasonal (Sz, Si, Ez, Ei, Tz, Ti)",
    )
    for i, (fast_value, slow_value) in enumerate(zip(fast_pct, slow_pct)):
        ax.text(i, fast_value / 2, "%.0f%%" % fast_value, ha="center", va="center", fontsize=15)
        ax.text(
            i,
            fast_value + slow_value / 2,
            "%.0f%%" % slow_value,
            ha="center",
            va="center",
            fontsize=15,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Lead time (d)"].astype(str).tolist(), fontsize=14)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.tick_params(axis="y", labelsize=14)
    ax.set_xlabel("Lead time (d)", fontsize=16)
    ax.set_ylabel("Normalized mean |SHAP| contribution (%)", fontsize=16)
    handles, labels = ax.get_legend_handles_labels()
    fig.suptitle("Predictor-group contributions", fontsize=17, y=0.98)
    fig.legend(handles, labels, frameon=False, fontsize=11, loc="upper center", bbox_to_anchor=(0.5, 0.925))
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.14, top=0.74)
    fig.savefig(OUTPUT_ROOT / "Fig9_predictor_group_contributions_fixed.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT_ROOT / "Fig9_predictor_group_contributions_fixed.tif", dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT_ROOT / "Fig9_predictor_group_contributions_fixed.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    seed_everything(FIXED_PARAMS["random_seed"])
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    data = read_csv_auto(DATA_PATH)
    data.columns = data.columns.str.strip()

    print("Output root: %s" % OUTPUT_ROOT, flush=True)
    print("Device: %s" % DEVICE, flush=True)
    print("Fixed params: %s" % json.dumps(FIXED_PARAMS), flush=True)

    rows = []
    for horizon in HORIZONS:
        print("\n===== H%dd fixed-architecture model =====" % horizon, flush=True)
        model, split, checkpoint_path, evaluation, best_val_loss = train_fixed_model(data, horizon)
        save_prediction_and_metrics(horizon, split, evaluation, checkpoint_path, best_val_loss)
        rows.append(run_shap(horizon, model, split, checkpoint_path, evaluation))
        print(
            "H%dd done: test NSE=%.4f RMSE=%.3f"
            % (horizon, evaluation["test_metrics"]["NSE"], evaluation["test_metrics"]["RMSE"]),
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    group_df = pd.DataFrame(rows).sort_values("Lead time (d)")
    group_df.to_csv(OUTPUT_ROOT / "fast_slow_shap_contribution_fixed.csv", index=False, encoding="utf-8-sig")
    plot_group_contributions(group_df)
    plot_four_lead_figures()
    print("\nAll fixed-architecture SHAP outputs saved to: %s" % OUTPUT_ROOT, flush=True)


if __name__ == "__main__":
    main()
