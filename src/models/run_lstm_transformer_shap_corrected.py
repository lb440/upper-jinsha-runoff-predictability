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
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = os.environ.get('RUNOFF_DATA_PATH', str(REPOSITORY_ROOT / 'data' / 'example_model_input_synthetic.csv'))
MODEL_ROOT = os.environ.get('LSTM_TRANSFORMER_MODEL_ROOT', 'config/optimized_parameters')
OUTPUT_ROOT = os.environ.get('SHAP_OUTPUT_ROOT', 'outputs/optimized_shap')

TIME_COL = "Date"
TARGET_COL = "Q_shigu"
HORIZONS = [1, 3, 7, 15]
TRAIN_END_DATE = pd.Timestamp("2014-12-31")
VAL_END_DATE = pd.Timestamp("2016-12-31")
TEST_END_DATE = pd.Timestamp("2020-12-31")
RANDOM_SEED = 222
N_BACKGROUND = int(os.environ.get("SHAP_N_BACKGROUND", "200"))
N_EXPLAIN = int(os.environ.get("SHAP_N_EXPLAIN", "400"))
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available() and os.environ.get("SHAP_DEVICE", "cuda").lower() != "cpu"
    else "cpu"
)

FAST_FEATURES = ["Qz", "Pz", "Pi"]
SLOW_SEASONAL_FEATURES = ["Sz", "Si", "Ez", "Ei", "Tz", "Ti"]


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


def normalize_sequences(x_data, feature_cols, x_scalers):
    norm = np.zeros_like(x_data, dtype=np.float32)
    sample_count, sequence_length, _ = x_data.shape

    for idx, feature_name in enumerate(feature_cols):
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
    true = np.asarray(true_vals)
    pred = np.asarray(pred_vals)
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    r = np.corrcoef(true, pred)[0, 1]
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    nse = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    alpha_kge = np.std(pred) / (np.std(true) + 1e-8)
    beta_kge = np.mean(pred) / (np.mean(true) + 1e-8)
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha_kge - 1) ** 2 + (beta_kge - 1) ** 2)
    return {
        "R": float(r),
        "NSE": float(nse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "KGE": float(kge),
        "R2": float(r2),
    }


def build_samples(data, feature_cols, kim, horizon):
    data = data.dropna(subset=[TIME_COL, TARGET_COL] + feature_cols).reset_index(drop=True)
    data[TIME_COL] = pd.to_datetime(data[TIME_COL])
    data = data[
        (data[TIME_COL] >= "2006-01-01") & (data[TIME_COL] <= "2020-12-31")
    ].reset_index(drop=True)

    data[TARGET_COL] = data[TARGET_COL].clip(lower=0)
    for col in feature_cols:
        if "Q" in col:
            data[col] = data[col].clip(lower=0)

    time = data[TIME_COL]
    features = data[feature_cols].values.astype(np.float32)
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


def load_checkpoint(horizon):
    checkpoint_path = os.path.join(
        MODEL_ROOT,
        f"L-T-{horizon}",
        f"best_lstm_transformer_no_lag_{horizon}d.pth",
    )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint_path, checkpoint


def run_one_horizon(data, horizon):
    checkpoint_path, checkpoint = load_checkpoint(horizon)
    feature_cols = list(checkpoint["feature_cols"])
    kim = int(checkpoint["kim"])
    output_dir = os.path.join(OUTPUT_ROOT, f"H{horizon}d")
    os.makedirs(output_dir, exist_ok=True)

    x_all, y_all, data_time, train_idx, val_idx, test_idx = build_samples(
        data.copy(), feature_cols, kim, horizon
    )
    x_train, x_val, x_test = x_all[train_idx], x_all[val_idx], x_all[test_idx]
    y_train, y_val, y_test = y_all[train_idx], y_all[val_idx], y_all[test_idx]

    x_scalers = {
        name: restore_minmax_scaler(checkpoint["x_scalers"][name])
        for name in feature_cols
    }
    y_scaler = restore_minmax_scaler(checkpoint["y_scaler"])

    x_train_norm = normalize_sequences(x_train, feature_cols, x_scalers)
    x_val_norm = normalize_sequences(x_val, feature_cols, x_scalers)
    x_test_norm = normalize_sequences(x_test, feature_cols, x_scalers)

    model = HybridLSTMTransformer(
        input_dim=len(feature_cols),
        lstm_hidden=int(checkpoint["lstm_hidden"]),
        d_model=int(checkpoint["d_model"]),
        nhead=int(checkpoint["nhead"]),
        dim_feedforward=int(checkpoint.get("dim_feedforward", int(checkpoint["d_model"]) * 2)),
        fusion_hidden=int(checkpoint["fusion_hidden"]),
        dropout=float(checkpoint["dropout"]),
        dropout_fc=float(checkpoint["dropout_fc"]),
        max_len=kim,
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        pred_test_norm = model(torch.FloatTensor(x_test_norm).to(DEVICE)).cpu().numpy()
    y_test_pred = y_scaler.inverse_transform(pred_test_norm).flatten()
    y_test_pred = np.clip(y_test_pred, 0, None)
    y_test_true = y_test.flatten()
    metrics = calculate_metrics(y_test_true, y_test_pred)

    pd.DataFrame(
        {
            "Date": data_time.iloc[test_idx].reset_index(drop=True),
            "Observed": y_test_true,
            "Predicted": y_test_pred,
            "Error": y_test_pred - y_test_true,
        }
    ).to_csv(
        os.path.join(output_dir, f"prediction_test_H{horizon}d.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    with open(os.path.join(output_dir, f"shap_run_metadata_H{horizon}d.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "data_path": DATA_PATH,
                "checkpoint_path": checkpoint_path,
                "horizon": horizon,
                "kim": kim,
                "feature_cols": feature_cols,
                "random_seed": RANDOM_SEED,
                "background_samples": min(N_BACKGROUND, x_train_norm.shape[0]),
                "explained_test_samples": min(N_EXPLAIN, x_test_norm.shape[0]),
                "device": str(DEVICE),
                "test_metrics": metrics,
            },
            f,
            indent=2,
        )

    background_data = torch.FloatTensor(x_train_norm[:N_BACKGROUND]).to(DEVICE)
    x_explain = torch.FloatTensor(x_test_norm[:N_EXPLAIN]).to(DEVICE)

    print(
        f"H{horizon}d: running SHAP on {DEVICE} with "
        f"{background_data.shape[0]} background and {x_explain.shape[0]} test samples",
        flush=True,
    )
    explainer = shap.GradientExplainer(model, background_data)
    shap_values = explainer.shap_values(x_explain)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 4:
        shap_values = shap_values[..., 0]
    if shap_values.ndim != 3:
        raise ValueError(f"Unexpected SHAP shape for H{horizon}d: {shap_values.shape}")

    lag_feature_abs = np.mean(np.abs(shap_values), axis=0)
    feature_lag_abs = lag_feature_abs[::-1, :].T
    lag_cols = [str(i) for i in range(kim)]
    pd.DataFrame(feature_lag_abs, index=feature_cols, columns=lag_cols).to_csv(
        os.path.join(output_dir, f"SHAP_feature_lag_mean_abs_matrix_H{horizon}d.csv"),
        encoding="utf-8-sig",
    )

    shap_values_agg = np.mean(shap_values, axis=1)
    shap_importance = np.mean(np.abs(shap_values_agg), axis=0)
    shap_metrics = pd.DataFrame(
        {"Feature": feature_cols, "Mean_abs_SHAP": shap_importance}
    ).sort_values("Mean_abs_SHAP", ascending=False)
    shap_metrics.to_csv(
        os.path.join(output_dir, f"shap_feature_importance_H{horizon}d.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    np.savez(
        os.path.join(output_dir, f"shap_results_H{horizon}d.npz"),
        shap_values_raw=shap_values,
        shap_values_agg=shap_values_agg,
        shap_importance=shap_importance,
        feature_names=np.asarray(feature_cols),
        feature_lag_mean_abs_shap=feature_lag_abs,
        lag_labels=np.asarray(lag_cols),
    )

    importance_by_feature = dict(zip(feature_cols, shap_importance))
    fast_raw = float(sum(importance_by_feature[f] for f in FAST_FEATURES))
    slow_raw = float(sum(importance_by_feature[f] for f in SLOW_SEASONAL_FEATURES))
    total_raw = fast_raw + slow_raw

    print(
        f"H{horizon}d: NSE={metrics['NSE']:.4f}, fast={fast_raw / total_raw:.4f}, "
        f"slow={slow_raw / total_raw:.4f}",
        flush=True,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "Lead time (d)": horizon,
        "Fast-response group": fast_raw / total_raw,
        "Slow/seasonal group": slow_raw / total_raw,
        "Fast-response raw Mean_abs_SHAP sum": fast_raw,
        "Slow/seasonal raw Mean_abs_SHAP sum": slow_raw,
        "Total raw Mean_abs_SHAP sum": total_raw,
    }


def plot_group_contributions(group_df):
    plot_df = group_df.sort_values("Lead time (d)")
    x = np.arange(len(plot_df))
    fast_pct = plot_df["Fast-response group"].values * 100
    slow_pct = plot_df["Slow/seasonal group"].values * 100

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
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
        ax.text(i, fast_value / 2, f"{fast_value:.0f}%", ha="center", va="center", fontsize=15)
        ax.text(
            i,
            fast_value + slow_value / 2,
            f"{slow_value:.0f}%",
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
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=11,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.14, top=0.74)

    png_path = os.path.join(OUTPUT_ROOT, "Fig9_predictor_group_contributions_corrected.png")
    tif_path = os.path.join(OUTPUT_ROOT, "Fig9_predictor_group_contributions_corrected.tif")
    pdf_path = os.path.join(OUTPUT_ROOT, "Fig9_predictor_group_contributions_corrected.pdf")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(tif_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, tif_path, pdf_path


def main():
    seed_everything(RANDOM_SEED)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    if os.environ.get("PLOT_ONLY", "0") == "1":
        group_csv = os.path.join(OUTPUT_ROOT, "fast_slow_shap_contribution_corrected.csv")
        group_df = pd.read_csv(group_csv)
        png_path, tif_path, pdf_path = plot_group_contributions(group_df)
        print(f"Figure PNG: {png_path}")
        print(f"Figure TIF: {tif_path}")
        print(f"Figure PDF: {pdf_path}")
        return

    data = read_csv_auto(DATA_PATH)
    data.columns = data.columns.str.strip()

    print(
        f"Running corrected LSTM-Transformer SHAP on {DEVICE}; "
        f"background={N_BACKGROUND}, explain={N_EXPLAIN}",
        flush=True,
    )

    rows = []
    for horizon in HORIZONS:
        rows.append(run_one_horizon(data, horizon))

    group_df = pd.DataFrame(rows).sort_values("Lead time (d)")
    group_csv = os.path.join(OUTPUT_ROOT, "fast_slow_shap_contribution_corrected.csv")
    group_df.to_csv(group_csv, index=False, encoding="utf-8-sig")
    png_path, tif_path, pdf_path = plot_group_contributions(group_df)

    print(f"Group contribution CSV: {group_csv}")
    print(f"Figure PNG: {png_path}")
    print(f"Figure TIF: {tif_path}")
    print(f"Figure PDF: {pdf_path}")


if __name__ == "__main__":
    main()
