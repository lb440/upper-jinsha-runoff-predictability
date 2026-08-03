import argparse
import copy
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset


TIME_COL = "Date"
TARGET_COL = "Q_shigu"
FEATURE_COLS = ["Qz", "Pz", "Tz", "Ez", "Sz", "Pi", "Ti", "Ei", "Si"]
TRAIN_END_DATE = pd.Timestamp("2014-12-31")
VAL_END_DATE = pd.Timestamp("2016-12-31")
TEST_END_DATE = pd.Timestamp("2020-12-31")
WET_MONTHS = (6, 7, 8, 9, 10)
WET_YEARS = (2017, 2018, 2019, 2020)

COMBINATIONS = {
    "C1_Qz_only": {
        "code": "C1",
        "label": "Qz only",
        "features": ["Qz"],
        "description": "Upstream discharge only.",
    },
    "C2_Qz_Pz_Pi": {
        "code": "C2",
        "label": "Qz + Pz + Pi",
        "features": ["Qz", "Pz", "Pi"],
        "description": "Core fast-response predictors.",
    },
    "C3_Core_plus_T": {
        "code": "C3",
        "label": "Qz + Pz + Pi + Tz + Ti",
        "features": ["Qz", "Pz", "Tz", "Pi", "Ti"],
        "description": "Core predictors plus temperature.",
    },
    "C4_Core_plus_E": {
        "code": "C4",
        "label": "Qz + Pz + Pi + Ez + Ei",
        "features": ["Qz", "Pz", "Ez", "Pi", "Ei"],
        "description": "Core predictors plus evaporation.",
    },
    "C5_Core_plus_S": {
        "code": "C5",
        "label": "Qz + Pz + Pi + Sz + Si",
        "features": ["Qz", "Pz", "Sz", "Pi", "Si"],
        "description": "Core predictors plus snow depth.",
    },
    "C6_Core_plus_T_E": {
        "code": "C6",
        "label": "Qz + Pz + Pi + Tz + Ti + Ez + Ei",
        "features": ["Qz", "Pz", "Tz", "Ez", "Pi", "Ti", "Ei"],
        "description": "Core predictors plus temperature and evaporation.",
    },
    "C7_Full": {
        "code": "C7",
        "label": "Full predictors",
        "features": FEATURE_COLS,
        "description": "All nine predictors.",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LSTM-Transformer factor-combination experiments for H=1/3/7/15."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=os.environ.get("RUNOFF_DATA_PATH", "data/example_model_input_synthetic.csv"),
    )
    parser.add_argument(
        "--param-root",
        type=str,
        default="config/optimized_parameters",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/factor_combinations",
    )
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 7, 15])
    parser.add_argument("--combinations", nargs="+", type=str, default=list(COMBINATIONS))
    parser.add_argument("--kim", type=int, default=90)
    parser.add_argument("--random-seed", type=int, default=222)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-6)
    parser.add_argument("--compute-device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--force", action="store_true", help="Rerun completed experiments.")
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available.")
    return torch.device(requested)


def read_data(data_path):
    try:
        data = pd.read_csv(data_path, encoding="utf-8")
    except UnicodeDecodeError:
        data = pd.read_csv(data_path, encoding="gbk")
    data.columns = data.columns.str.strip()
    required_cols = [TIME_COL, TARGET_COL] + FEATURE_COLS
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        raise ValueError(f"Missing columns in data file: {missing}")
    data = data.dropna(subset=required_cols).reset_index(drop=True)
    data[TIME_COL] = pd.to_datetime(data[TIME_COL])
    data = data[
        (data[TIME_COL] >= "2006-01-01")
        & (data[TIME_COL] <= str(TEST_END_DATE.date()))
    ].reset_index(drop=True)
    data[TARGET_COL] = data[TARGET_COL].clip(lower=0)
    data["Qz"] = data["Qz"].clip(lower=0)
    return data


def get_split_indices(data_time):
    train_mask = data_time <= TRAIN_END_DATE
    val_mask = (data_time > TRAIN_END_DATE) & (data_time <= VAL_END_DATE)
    test_mask = (data_time > VAL_END_DATE) & (data_time <= TEST_END_DATE)
    return np.where(train_mask)[0], np.where(val_mask)[0], np.where(test_mask)[0]


def build_supervised(data, feature_cols, kim, horizon):
    features = data[feature_cols].values.astype(np.float32)
    runoff = data[TARGET_COL].values.astype(np.float32)
    time = data[TIME_COL]
    valid_samples = len(runoff) - kim - horizon + 1
    if valid_samples <= 0:
        raise ValueError("No valid samples for the configured kim/horizon.")
    x_values, y_values, out_idx = [], [], []
    for start in range(valid_samples):
        target_idx = start + kim + horizon - 1
        x_values.append(features[start : start + kim])
        y_values.append(runoff[target_idx])
        out_idx.append(target_idx)
    x_values = np.asarray(x_values, dtype=np.float32)
    y_values = np.asarray(y_values, dtype=np.float32).reshape(-1, 1)
    data_time = time.iloc[out_idx].reset_index(drop=True)
    train_idx, val_idx, test_idx = get_split_indices(data_time)
    return {
        "X_train": x_values[train_idx],
        "X_val": x_values[val_idx],
        "X_test": x_values[test_idx],
        "y_train": y_values[train_idx],
        "y_val": y_values[val_idx],
        "y_test": y_values[test_idx],
        "time_train": data_time.iloc[train_idx].reset_index(drop=True),
        "time_val": data_time.iloc[val_idx].reset_index(drop=True),
        "time_test": data_time.iloc[test_idx].reset_index(drop=True),
    }


def scale_splits(splits, feature_cols, kim):
    feature_dim = len(feature_cols)
    x_train_norm = np.zeros_like(splits["X_train"])
    x_val_norm = np.zeros_like(splits["X_val"])
    x_test_norm = np.zeros_like(splits["X_test"])
    x_scalers = {}
    for idx, feature in enumerate(feature_cols):
        scaler = MinMaxScaler()
        scaler.fit(splits["X_train"][:, :, idx].reshape(-1, 1))
        x_train_norm[:, :, idx] = scaler.transform(
            splits["X_train"][:, :, idx].reshape(-1, 1)
        ).reshape(splits["X_train"].shape[0], kim)
        x_val_norm[:, :, idx] = scaler.transform(
            splits["X_val"][:, :, idx].reshape(-1, 1)
        ).reshape(splits["X_val"].shape[0], kim)
        x_test_norm[:, :, idx] = scaler.transform(
            splits["X_test"][:, :, idx].reshape(-1, 1)
        ).reshape(splits["X_test"].shape[0], kim)
        x_scalers[feature] = scaler
    y_scaler = MinMaxScaler()
    y_train_norm = y_scaler.fit_transform(splits["y_train"])
    y_val_norm = y_scaler.transform(splits["y_val"])
    y_test_norm = y_scaler.transform(splits["y_test"])
    return {
        "X_train": x_train_norm,
        "X_val": x_val_norm,
        "X_test": x_test_norm,
        "y_train": y_train_norm,
        "y_val": y_val_norm,
        "y_test": y_test_norm,
        "x_scalers": x_scalers,
        "y_scaler": y_scaler,
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
    return float(1 - np.sum((true_vals - pred_vals) ** 2) / denominator)


def calculate_kge(true_vals, pred_vals):
    true_vals = np.asarray(true_vals, dtype=np.float64)
    pred_vals = np.asarray(pred_vals, dtype=np.float64)
    corr = safe_corrcoef(true_vals, pred_vals)
    if np.isnan(corr):
        return np.nan
    alpha = np.std(pred_vals) / (np.std(true_vals) + 1e-8)
    beta = np.mean(pred_vals) / (np.mean(true_vals) + 1e-8)
    return float(1 - np.sqrt((corr - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def calculate_pbias(true_vals, pred_vals):
    true_vals = np.asarray(true_vals, dtype=np.float64)
    pred_vals = np.asarray(pred_vals, dtype=np.float64)
    denominator = np.sum(true_vals)
    if np.isclose(denominator, 0):
        return np.nan
    return float(100 * np.sum(pred_vals - true_vals) / denominator)


def calculate_metrics(true_vals, pred_vals):
    true_vals = np.asarray(true_vals, dtype=np.float64).reshape(-1)
    pred_vals = np.asarray(pred_vals, dtype=np.float64).reshape(-1)
    return {
        "R": safe_corrcoef(true_vals, pred_vals),
        "NSE": calculate_nse(true_vals, pred_vals),
        "RMSE": float(np.sqrt(mean_squared_error(true_vals, pred_vals))),
        "MAE": float(mean_absolute_error(true_vals, pred_vals)),
        "MAPE(%)": float(np.mean(np.abs((pred_vals - true_vals) / (true_vals + 1e-8))) * 100),
        "Bias": float(np.mean(pred_vals - true_vals)),
        "PBIAS(%)": calculate_pbias(true_vals, pred_vals),
        "KGE": calculate_kge(true_vals, pred_vals),
        "R2": float(r2_score(true_vals, pred_vals)),
    }


def serialize_scaler(scaler):
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
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


def load_params(param_root, horizon):
    param_path = Path(param_root) / f"L-T-BO-{horizon}" / "no_lag_best_params.json"
    with param_path.open("r", encoding="utf-8") as handle:
        return param_path, json.load(handle)


def predict(model, x_tensor, y_scaler):
    model.eval()
    with torch.no_grad():
        pred_norm = model(x_tensor).detach().cpu().numpy()
    return y_scaler.inverse_transform(pred_norm).reshape(-1)


def run_one(data, combo_name, combo_meta, horizon, args, device):
    feature_cols = [feature for feature in FEATURE_COLS if feature in combo_meta["features"]]
    output_dir = Path(args.output_root) / f"H{horizon}" / combo_name
    metrics_json = output_dir / "metrics.json"
    if metrics_json.exists() and not args.force:
        with metrics_json.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload["summary_row"]

    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.random_seed)
    param_path, params = load_params(args.param_root, horizon)

    splits = build_supervised(data, feature_cols, args.kim, horizon)
    scaled = scale_splits(splits, feature_cols, args.kim)

    tensors = {
        key: torch.FloatTensor(value).to(device)
        for key, value in scaled.items()
        if key in {"X_train", "X_val", "X_test", "y_train", "y_val", "y_test"}
    }
    train_loader = DataLoader(
        TensorDataset(tensors["X_train"], tensors["y_train"]),
        batch_size=int(params["batch_size"]),
        shuffle=True,
        generator=torch.Generator(device="cpu").manual_seed(args.random_seed),
    )

    model = HybridLSTMTransformer(
        input_dim=len(feature_cols),
        lstm_hidden=int(params["lstm_hidden"]),
        d_model=int(params["d_model"]),
        nhead=int(params["nhead"]),
        dim_feedforward=int(params["d_model"]) * 2,
        fusion_hidden=int(params["fusion_hidden"]),
        dropout=float(params.get("dropout", 0.1)),
        dropout_fc=float(params.get("dropout_fc", 0.1)),
        max_len=args.kim,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(params["lr"]),
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.scheduler_min_lr,
        verbose=False,
    )

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(args.max_epochs):
        model.train()
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item() * x_batch.size(0)
        train_loss = running_loss / len(tensors["X_train"])
        train_losses.append(float(train_loss))

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(tensors["X_val"]), tensors["y_val"]).item()
        val_losses.append(float(val_loss))
        scheduler.step(val_loss)

        if val_loss < best_val_loss - args.min_delta:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= args.patience:
            break

    model.load_state_dict(best_state)

    y_train_true = splits["y_train"].reshape(-1)
    y_val_true = splits["y_val"].reshape(-1)
    y_test_true = splits["y_test"].reshape(-1)
    y_train_pred = predict(model, tensors["X_train"], scaled["y_scaler"])
    y_val_pred = predict(model, tensors["X_val"], scaled["y_scaler"])
    y_test_pred = predict(model, tensors["X_test"], scaled["y_scaler"])

    train_metrics = calculate_metrics(y_train_true, y_train_pred)
    val_metrics = calculate_metrics(y_val_true, y_val_pred)
    test_metrics = calculate_metrics(y_test_true, y_test_pred)

    prediction_df = pd.DataFrame(
        {
            "Date": splits["time_test"],
            "True_Q_shigu": y_test_true,
            "Pred_Q_shigu": y_test_pred,
        }
    )
    prediction_df["Error"] = prediction_df["Pred_Q_shigu"] - prediction_df["True_Q_shigu"]
    prediction_df["Absolute error"] = prediction_df["Error"].abs()
    prediction_df["Year"] = pd.to_datetime(prediction_df["Date"]).dt.year
    prediction_df["Month"] = pd.to_datetime(prediction_df["Date"]).dt.month
    prediction_df.to_csv(output_dir / "prediction_test.csv", index=False, encoding="utf-8-sig")

    wet_df = prediction_df[
        prediction_df["Month"].isin(WET_MONTHS) & prediction_df["Year"].isin(WET_YEARS)
    ].copy()
    wet_metrics = calculate_metrics(
        wet_df["True_Q_shigu"].to_numpy(dtype=float),
        wet_df["Pred_Q_shigu"].to_numpy(dtype=float),
    )

    torch.save(model.state_dict(), output_dir / "best_model_state.pt")
    scaler_payload = {
        "x_scalers": {feature: serialize_scaler(scaler) for feature, scaler in scaled["x_scalers"].items()},
        "y_scaler": serialize_scaler(scaled["y_scaler"]),
    }
    with (output_dir / "scalers.json").open("w", encoding="utf-8") as handle:
        json.dump(scaler_payload, handle, indent=2, ensure_ascii=False)

    summary_row = {
        "Model": "LSTM-Transformer",
        "Lead time (d)": horizon,
        "Code": combo_meta["code"],
        "Combination": combo_name,
        "Label": combo_meta["label"],
        "Description": combo_meta["description"],
        "Feature count": len(feature_cols),
        "Features": ", ".join(feature_cols),
        "Test_R": test_metrics["R"],
        "Test_NSE": test_metrics["NSE"],
        "Test_RMSE": test_metrics["RMSE"],
        "Test_MAE": test_metrics["MAE"],
        "Test_PBIAS(%)": test_metrics["PBIAS(%)"],
        "Test_KGE": test_metrics["KGE"],
        "Wet_R": wet_metrics["R"],
        "Wet_NSE": wet_metrics["NSE"],
        "Wet_RMSE": wet_metrics["RMSE"],
        "Wet_MAE": wet_metrics["MAE"],
        "Wet_PBIAS(%)": wet_metrics["PBIAS(%)"],
        "Wet_KGE": wet_metrics["KGE"],
        "Train_NSE": train_metrics["NSE"],
        "Val_NSE": val_metrics["NSE"],
        "Best val loss": best_val_loss,
        "Epochs run": len(train_losses),
        "Random seed": args.random_seed,
        "Param path": str(param_path),
        "Output dir": str(output_dir),
    }

    metadata = {
        "summary_row": summary_row,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "wet_metrics": wet_metrics,
        "params": params,
        "feature_cols": feature_cols,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "sample_counts": {
            "train": int(len(y_train_true)),
            "validation": int(len(y_val_true)),
            "test": int(len(y_test_true)),
            "wet_test": int(len(wet_df)),
        },
        "data_path": str(args.data_path),
    }
    with metrics_json.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    return summary_row


def write_summary_outputs(rows, output_root):
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    code_order = {meta["code"]: i for i, meta in enumerate(COMBINATIONS.values())}
    df["CodeOrder"] = df["Code"].map(code_order)
    df = df.sort_values(["Lead time (d)", "CodeOrder"]).drop(columns=["CodeOrder"]).reset_index(drop=True)
    df.to_csv(root / "lstm_transformer_factor_combination_metrics_long.csv", index=False, encoding="utf-8-sig")

    baseline = df[df["Code"] == "C2"][["Lead time (d)", "Test_NSE", "Wet_NSE"]].rename(
        columns={"Test_NSE": "Baseline_Test_NSE", "Wet_NSE": "Baseline_Wet_NSE"}
    )
    delta_df = df.merge(baseline, on="Lead time (d)", how="left")
    delta_df["Delta_Test_NSE_vs_C2"] = delta_df["Test_NSE"] - delta_df["Baseline_Test_NSE"]
    delta_df["Delta_Wet_NSE_vs_C2"] = delta_df["Wet_NSE"] - delta_df["Baseline_Wet_NSE"]
    delta_df.to_csv(
        root / "lstm_transformer_factor_combination_metrics_with_delta.csv",
        index=False,
        encoding="utf-8-sig",
    )

    for metric in ("Test_NSE", "Delta_Test_NSE_vs_C2", "Wet_NSE", "Delta_Wet_NSE_vs_C2"):
        matrix = delta_df.pivot(index="Label", columns="Lead time (d)", values=metric)
        ordered_labels = [COMBINATIONS[name]["label"] for name in COMBINATIONS]
        matrix = matrix.reindex(index=ordered_labels, columns=[1, 3, 7, 15])
        matrix.to_csv(root / f"{metric}_matrix.csv", encoding="utf-8-sig")
    return df, delta_df


def main():
    args = parse_args()
    unknown_combos = [combo for combo in args.combinations if combo not in COMBINATIONS]
    if unknown_combos:
        raise ValueError(f"Unknown combinations: {unknown_combos}")
    set_seed(args.random_seed)
    device = resolve_device(args.compute_device)
    print(f"Using device: {device}")
    print(f"Data path: {args.data_path}")
    print(f"Output root: {args.output_root}")
    data = read_data(args.data_path)
    print(f"Rows loaded: {len(data)} | {data[TIME_COL].iloc[0].date()} to {data[TIME_COL].iloc[-1].date()}")

    rows = []
    total = len(args.horizons) * len(args.combinations)
    counter = 0
    for horizon in args.horizons:
        for combo_name in args.combinations:
            counter += 1
            combo_meta = COMBINATIONS[combo_name]
            print(
                f"[{counter:02d}/{total:02d}] H={horizon} | "
                f"{combo_meta['code']} {combo_meta['label']} | features={combo_meta['features']}",
                flush=True,
            )
            row = run_one(data, combo_name, combo_meta, horizon, args, device)
            print(
                f"    Test_NSE={row['Test_NSE']:.4f} | Wet_NSE={row['Wet_NSE']:.4f} | "
                f"epochs={row['Epochs run']}",
                flush=True,
            )
            rows.append(row)
            write_summary_outputs(rows, args.output_root)
    write_summary_outputs(rows, args.output_root)
    print("Completed LSTM-Transformer factor-combination experiments.")


if __name__ == "__main__":
    main()
