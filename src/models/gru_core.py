import copy

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from baseline_shared_utils import (
    FEATURE_COLS,
    build_supervised_splits,
    calculate_metrics,
    calculate_nse,
    ensure_dir,
    plot_optimization_history,
    prefixed_name,
    save_json,
    save_prediction_outputs,
    serialize_minmax_scaler,
    summarize_split_sizes,
    scale_sequence_splits,
)

FIXED_DROPOUT = 0.1


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        output, _ = self.gru(x)
        last = output[:, -1, :]
        last = self.dropout(last)
        return self.fc(last)


def build_tensors(scaled_splits, device):
    return {
        "X_train": torch.from_numpy(scaled_splits["X_train"]).float(),
        "y_train": torch.from_numpy(scaled_splits["y_train"]).float(),
        "X_val": torch.from_numpy(scaled_splits["X_val"]).float().to(device),
        "y_val": torch.from_numpy(scaled_splits["y_val"]).float().to(device),
        "X_test": torch.from_numpy(scaled_splits["X_test"]).float().to(device),
        "y_test": torch.from_numpy(scaled_splits["y_test"]).float().to(device),
    }


def train_gru_model(
    tensors,
    input_dim,
    gru_hidden,
    batch_size,
    dropout,
    lr,
    random_seed,
    device,
    max_epochs=100,
    patience=10,
    min_delta=1e-4,
    weight_decay=1e-4,
    scheduler_factor=0.5,
    scheduler_patience=5,
    scheduler_min_lr=1e-6,
):
    seed_torch(random_seed)
    train_loader = DataLoader(
        TensorDataset(tensors["X_train"], tensors["y_train"]),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(random_seed),
    )

    model = GRUModel(input_dim=input_dim, hidden_dim=gru_hidden, dropout=dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=scheduler_min_lr,
        verbose=False,
    )

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    train_losses = []
    val_losses = []
    no_improve = 0

    for _ in range(max_epochs):
        model.train()
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch)
            loss = criterion(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item() * x_batch.shape[0]

        train_loss = running_loss / len(tensors["X_train"])

        model.eval()
        with torch.no_grad():
            val_pred = model(tensors["X_val"])
            val_loss = criterion(val_pred, tensors["y_val"]).item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, train_losses, val_losses, best_val_loss


def predict_gru(model, x_tensor, y_scaler, device):
    with torch.no_grad():
        pred_norm = model(x_tensor.to(device)).cpu().numpy()
    pred = y_scaler.inverse_transform(pred_norm).flatten()
    return pred.clip(min=0.0)


def run_gru_experiment(
    kim,
    horizon,
    output_dir,
    params,
    feature_cols=None,
    prefix="",
    group_name=None,
    random_seed=222,
    max_epochs=100,
    patience=10,
    min_delta=1e-4,
    weight_decay=1e-4,
    scheduler_factor=0.5,
    scheduler_patience=5,
    scheduler_min_lr=1e-6,
):
    feature_cols = list(feature_cols or FEATURE_COLS)
    splits = build_supervised_splits(feature_cols, kim=kim, horizon=horizon)
    scaled = scale_sequence_splits(splits)
    device = resolve_device()
    tensors = build_tensors(scaled, device=device)
    output_dir = ensure_dir(output_dir)

    model, train_losses, val_losses, _ = train_gru_model(
        tensors=tensors,
        input_dim=len(feature_cols),
        gru_hidden=int(params["gru_hidden"]),
        batch_size=int(params["batch_size"]),
        dropout=float(params.get("dropout", FIXED_DROPOUT)),
        lr=float(params["lr"]),
        random_seed=random_seed,
        device=device,
        max_epochs=max_epochs,
        patience=patience,
        min_delta=min_delta,
        weight_decay=weight_decay,
        scheduler_factor=scheduler_factor,
        scheduler_patience=scheduler_patience,
        scheduler_min_lr=scheduler_min_lr,
    )

    train_true = splits.y_train.flatten()
    val_true = splits.y_val.flatten()
    test_true = splits.y_test.flatten()
    train_pred = predict_gru(
        model,
        torch.from_numpy(scaled["X_train"]).float(),
        scaled["y_scaler"],
        device=device,
    )
    val_pred = predict_gru(
        model,
        torch.from_numpy(scaled["X_val"]).float(),
        scaled["y_scaler"],
        device=device,
    )
    test_pred = predict_gru(
        model,
        tensors["X_test"].cpu(),
        scaled["y_scaler"],
        device=device,
    )

    train_metrics = calculate_metrics(train_true, train_pred)
    val_metrics = calculate_metrics(val_true, val_pred)
    test_metrics = calculate_metrics(test_true, test_pred)

    metadata = {
        "model_name": "GRU",
        "group_name": group_name,
        "kim": kim,
        "horizon": horizon,
        "feature_cols": feature_cols,
        "params": params,
        "random_seed": random_seed,
        "device": str(device),
        "max_epochs": max_epochs,
        "patience": patience,
        "min_delta": min_delta,
        "weight_decay": weight_decay,
        "scheduler_factor": scheduler_factor,
        "scheduler_patience": scheduler_patience,
        "scheduler_min_lr": scheduler_min_lr,
    }
    metadata.update(summarize_split_sizes(splits))

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": len(feature_cols),
            "hidden_dim": int(params["gru_hidden"]),
            "dropout": float(params.get("dropout", FIXED_DROPOUT)),
            "kim": kim,
            "horizon": horizon,
            "feature_cols": feature_cols,
            "target_col": "Q_shigu",
            "x_scalers": {
                name: serialize_minmax_scaler(scaler)
                for name, scaler in scaled["x_scalers"].items()
            },
            "y_scaler": serialize_minmax_scaler(scaled["y_scaler"]),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "metadata": metadata,
        },
        output_dir / prefixed_name(prefix, "best_gru_model.pth"),
    )
    save_json(output_dir / prefixed_name(prefix, "gru_metadata.json"), metadata)
    save_prediction_outputs(
        output_dir=output_dir,
        prefix=prefix,
        run_label="GRU",
        horizon=horizon,
        time_test=splits.time_test,
        y_test_true=test_true,
        y_test_pred=test_pred,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        train_losses=train_losses,
        val_losses=val_losses,
        extra_sections=[("Model Metadata", metadata)],
    )

    return {
        "metadata": metadata,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def run_gru_bo(
    kim,
    horizon,
    output_dir,
    feature_cols=None,
    n_trials=30,
    random_seed=222,
    max_epochs=100,
    patience=10,
    min_delta=1e-4,
    weight_decay=1e-4,
    scheduler_factor=0.5,
    scheduler_patience=5,
    scheduler_min_lr=1e-6,
):
    feature_cols = list(feature_cols or FEATURE_COLS)
    splits = build_supervised_splits(feature_cols, kim=kim, horizon=horizon)
    scaled = scale_sequence_splits(splits)
    device = resolve_device()
    tensors = build_tensors(scaled, device=device)
    output_dir = ensure_dir(output_dir)

    def objective(trial):
        params = {
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
            "gru_hidden": trial.suggest_categorical("gru_hidden", [32, 64, 128, 256]),
            "dropout": FIXED_DROPOUT,
            "lr": trial.suggest_categorical("lr", [1e-4, 3e-4, 5e-4, 1e-3]),
        }
        model, _, _, _ = train_gru_model(
            tensors=tensors,
            input_dim=len(feature_cols),
            gru_hidden=params["gru_hidden"],
            batch_size=params["batch_size"],
            dropout=params["dropout"],
            lr=params["lr"],
            random_seed=random_seed,
            device=device,
            max_epochs=max_epochs,
            patience=patience,
            min_delta=min_delta,
            weight_decay=weight_decay,
            scheduler_factor=scheduler_factor,
            scheduler_patience=scheduler_patience,
            scheduler_min_lr=scheduler_min_lr,
        )
        val_pred = predict_gru(model, tensors["X_val"].cpu(), scaled["y_scaler"], device=device)
        return calculate_nse(splits.y_val.flatten(), val_pred)

    sampler = TPESampler(seed=random_seed)
    pruner = MedianPruner()
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_trial = study.best_trial
    params_json = dict(best_trial.params)
    params_json.update(
        {
            "best_nse": float(best_trial.value),
            "kim": kim,
            "horizon": horizon,
            "n_trials": n_trials,
            "feature_cols": feature_cols,
            "random_seed": random_seed,
            "max_epochs": max_epochs,
            "patience": patience,
            "min_delta": min_delta,
            "weight_decay": weight_decay,
            "scheduler_factor": scheduler_factor,
            "scheduler_patience": scheduler_patience,
            "scheduler_min_lr": scheduler_min_lr,
            "device": str(device),
            "dropout": FIXED_DROPOUT,
            "dropout_search_mode": "fixed",
        }
    )
    params_json.update(summarize_split_sizes(splits))

    save_json(output_dir / "gru_best_params.json", params_json)
    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_dir / "optuna_trials.csv", index=False, encoding="utf-8-sig")
    plot_optimization_history(
        output_dir / "optimization_history.png",
        trials_df=trials_df,
        title="GRU Validation NSE",
        best_value=best_trial.value,
    )
    return params_json
