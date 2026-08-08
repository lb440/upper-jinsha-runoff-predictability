"""Run the fixed-architecture predictor-deletion experiment at 7 and 15 days.

Set RUNOFF_DATA_PATH to the corrected daily modelling CSV. The experiment uses
the same architecture, training settings, and random seed for every input
configuration; only the predictor columns change.
"""

import importlib.util
import json
import os
from pathlib import Path

import pandas as pd
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CORE_SCRIPT = Path(__file__).with_name("run_fixed_lstm_transformer_shap_analysis.py")
DATA_PATH = Path(
    os.environ.get(
        "RUNOFF_DATA_PATH",
        str(REPOSITORY_ROOT / "data" / "example_model_input_synthetic.csv"),
    )
)
OUTPUT_ROOT = Path(
    os.environ.get(
        "ABLATION_OUTPUT_ROOT",
        str(REPOSITORY_ROOT / "outputs" / "fixed_architecture_ablation_H7_H15"),
    )
)

HORIZONS = (7, 15)
COMBINATIONS = {
    "Full": ["Qz", "Pz", "Tz", "Ez", "Sz", "Pi", "Ti", "Ei", "Si"],
    "Fast_only": ["Qz", "Pz", "Pi"],
    "No_snow": ["Qz", "Pz", "Tz", "Ez", "Pi", "Ti", "Ei"],
    "No_thermal_evaporative": ["Qz", "Pz", "Sz", "Pi", "Si"],
}


def load_core_module():
    spec = importlib.util.spec_from_file_location("fixed_lstm_transformer_core", CORE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_one(core, horizon, label, features, data):
    run_root = OUTPUT_ROOT / f"H{horizon}d" / label
    core.FEATURE_COLS = list(features)
    core.OUTPUT_ROOT = run_root
    core.MODEL_ROOT = run_root / "fixed_architecture_models"
    core.DATA_PATH = DATA_PATH
    core.seed_everything(core.FIXED_PARAMS["random_seed"])
    core.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    core.MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    model, split, checkpoint_path, evaluation, best_val_loss = core.train_fixed_model(
        data.copy(), horizon
    )
    core.save_prediction_and_metrics(
        horizon, split, evaluation, checkpoint_path, best_val_loss
    )
    test = evaluation["test_metrics"]
    metadata = {
        "horizon": horizon,
        "input_configuration": label,
        "feature_columns": features,
        "data_path": str(DATA_PATH),
        "fixed_architecture": core.FIXED_PARAMS,
        "max_epochs": core.MAX_EPOCHS,
        "patience": core.PATIENCE,
        "weight_decay": core.WEIGHT_DECAY,
        "test_metrics": test,
    }
    with open(run_root / "run_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return {
        "Lead time (d)": horizon,
        "Input configuration": label,
        "Predictors": ", ".join(features),
        "Number of predictors": len(features),
        "NSE": test["NSE"],
        "KGE": test["KGE"],
        "RMSE (m3/s)": test["RMSE"],
        "MAE (m3/s)": test["MAE"],
        "Best validation loss": best_val_loss,
        "Train samples": split["x_train"].shape[0],
        "Validation samples": split["x_val"].shape[0],
        "Test samples": split["x_test"].shape[0],
        "Checkpoint": str(checkpoint_path),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {DATA_PATH}. Set RUNOFF_DATA_PATH to the corrected daily CSV."
        )
    core = load_core_module()
    core.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = core.read_csv_auto(DATA_PATH)
    data.columns = data.columns.str.strip()

    rows = []
    for horizon in HORIZONS:
        for label, features in COMBINATIONS.items():
            print(f"Running H{horizon}d / {label}: {features}", flush=True)
            rows.append(run_one(core, horizon, label, features, data))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    results = pd.DataFrame(rows)
    full = results.loc[
        results["Input configuration"] == "Full", ["Lead time (d)", "NSE", "KGE"]
    ].rename(columns={"NSE": "Full NSE", "KGE": "Full KGE"})
    results = results.merge(full, on="Lead time (d)", how="left")
    results["Delta NSE vs Full"] = results["NSE"] - results["Full NSE"]
    results["Delta KGE vs Full"] = results["KGE"] - results["Full KGE"]
    results = results.sort_values(["Lead time (d)", "Input configuration"])
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results.to_csv(
        OUTPUT_ROOT / "fixed_architecture_ablation_H7_H15_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    results.to_excel(
        OUTPUT_ROOT / "fixed_architecture_ablation_H7_H15_metrics.xlsx", index=False
    )
    print(results.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
