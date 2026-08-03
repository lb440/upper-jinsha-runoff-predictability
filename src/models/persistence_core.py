import pandas as pd
import torch

from baseline_shared_utils import (
    FEATURE_COLS,
    build_supervised_splits,
    calculate_metrics,
    ensure_dir,
    plot_optimization_history,
    prefixed_name,
    save_json,
    save_prediction_outputs,
    summarize_split_sizes,
)


def resolve_device(requested_device="auto"):
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_device)


def _predict_from_history(history, device):
    history_tensor = torch.from_numpy(history).float().to(device)
    return history_tensor[:, -1].detach().cpu().numpy().astype(float)


def run_persistence_experiment(
    kim,
    horizon,
    output_dir,
    prefix="",
    group_name=None,
    compute_device="auto",
):
    splits = build_supervised_splits(FEATURE_COLS, kim=kim, horizon=horizon)
    output_dir = ensure_dir(output_dir)
    device = resolve_device(compute_device)

    train_pred = _predict_from_history(splits.history_train, device)
    val_pred = _predict_from_history(splits.history_val, device)
    test_pred = _predict_from_history(splits.history_test, device)

    train_true = splits.y_train.flatten()
    val_true = splits.y_val.flatten()
    test_true = splits.y_test.flatten()

    train_metrics = calculate_metrics(train_true, train_pred)
    val_metrics = calculate_metrics(val_true, val_pred)
    test_metrics = calculate_metrics(test_true, test_pred)

    metadata = {
        "model_name": "Persistence",
        "method": "Persistence baseline",
        "uses_target_history_only": True,
        "group_name": group_name,
        "kim": kim,
        "horizon": horizon,
        "compute_device": str(device),
    }
    metadata.update(summarize_split_sizes(splits))
    if group_name:
        metadata["note"] = (
            "Ablation group is recorded for compatibility only; "
            "persistence uses target history only and therefore does not change across groups."
        )

    save_json(
        output_dir / prefixed_name(prefix, "persistence_metadata.json"),
        metadata,
    )
    save_prediction_outputs(
        output_dir=output_dir,
        prefix=prefix,
        run_label="Persistence",
        horizon=horizon,
        time_test=splits.time_test,
        y_test_true=test_true,
        y_test_pred=test_pred,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        extra_sections=[("Model Metadata", metadata)],
    )

    return {
        "metadata": metadata,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def run_persistence_bo(kim, horizon, output_dir, compute_device="auto"):
    result = run_persistence_experiment(
        kim=kim,
        horizon=horizon,
        output_dir=output_dir,
        prefix="",
        group_name=None,
        compute_device=compute_device,
    )
    output_dir = ensure_dir(output_dir)

    params = {
        "model_name": "Persistence",
        "has_tunable_hyperparameters": False,
        "best_nse": result["val_metrics"]["NSE"],
        "kim": kim,
        "horizon": horizon,
        "compute_device": result["metadata"]["compute_device"],
    }
    save_json(output_dir / "persistence_best_params.json", params)

    trials_df = pd.DataFrame(
        [
            {
                "number": 0,
                "value": result["val_metrics"]["NSE"],
                "params": "{}",
                "state": "COMPLETE",
            }
        ]
    )
    trials_df.to_csv(output_dir / "optuna_trials.csv", index=False, encoding="utf-8-sig")
    plot_optimization_history(
        output_dir / "optimization_history.png",
        trials_df=trials_df,
        title="Persistence Validation NSE",
        best_value=result["val_metrics"]["NSE"],
    )
    return params
