import argparse
import json
from pathlib import Path

import pandas as pd

from baseline_shared_utils import calculate_metrics


SUPPORTED_HORIZONS = (1, 3, 7, 15)
MODEL_ORDER = {
    "Persistence": 1,
    "AR": 2,
    "XGBoost": 3,
    "BP": 4,
    "MLP": 5,
    "GRU": 6,
    "LSTM": 7,
    "Transformer": 8,
    "LSTM-Transformer": 9,
}
MODEL_SPECS = [
    {
        "model": "Persistence",
        "family": "Statistical",
        "dir_template": "C:/Persistence-{horizon}",
        "prediction_template": "prediction_test.csv",
    },
    {
        "model": "AR",
        "family": "Statistical",
        "dir_template": "C:/AR-{horizon}",
        "prediction_template": "prediction_test.csv",
    },
    {
        "model": "XGBoost",
        "family": "Classical ML",
        "dir_template": "C:/XGBoost-{horizon}",
        "prediction_template": "prediction_test.csv",
    },
    {
        "model": "BP",
        "family": "DL",
        "dir_template": "C:/BP-{horizon}",
        "prediction_template": "prediction_test.csv",
    },
    {
        "model": "MLP",
        "family": "DL",
        "dir_template": "C:/MLP-{horizon}",
        "prediction_template": "prediction_test.csv",
    },
    {
        "model": "GRU",
        "family": "DL",
        "dir_template": "C:/GRU-{horizon}",
        "prediction_template": "prediction_test.csv",
    },
    {
        "model": "LSTM",
        "family": "DL",
        "dir_template": "C:/LSTM-{horizon}",
        "prediction_template": "prediction_test.csv",
    },
    {
        "model": "Transformer",
        "family": "DL",
        "dir_template": "C:/Transformer-{horizon}",
        "prediction_template": "prediction_test.csv",
    },
    {
        "model": "LSTM-Transformer",
        "family": "DL",
        "dir_template": "C:/L-T-{horizon}",
        "prediction_template": "prediction_test_{horizon}d_no_lag.csv",
    },
]
METRIC_COLUMNS = ["R", "NSE", "RMSE", "MAE", "MAPE(%)", "Bias", "KGE", "R2"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recompute the main comparison table from the formal prediction CSV files."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "comparison_tables_seed222"),
    )
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def load_metadata(main_dir):
    for pattern in ("*metadata.json", "*.json"):
        for candidate in sorted(main_dir.glob(pattern)):
            try:
                with candidate.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if isinstance(payload, dict):
                    payload["_metadata_file"] = str(candidate)
                    return payload
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                continue
    return {}


def extract_backend(metadata):
    if "resolved_backend" in metadata:
        return metadata["resolved_backend"]
    params = metadata.get("params")
    if isinstance(params, dict):
        return params.get("resolved_backend") or params.get("requested_backend") or ""
    return metadata.get("backend", "")


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


def collect_rows():
    rows = []
    missing = []
    for spec in MODEL_SPECS:
        for horizon in SUPPORTED_HORIZONS:
            main_dir = Path(spec["dir_template"].format(horizon=horizon))
            prediction_file = main_dir / spec["prediction_template"].format(horizon=horizon)
            if not prediction_file.exists():
                missing.append(str(prediction_file))
                continue

            prediction_df = pd.read_csv(prediction_file)
            metrics = calculate_metrics(
                prediction_df["True_Q_shigu"].values,
                prediction_df["Pred_Q_shigu"].values,
            )
            metadata = load_metadata(main_dir)
            row = {
                "Family": spec["family"],
                "Model": spec["model"],
                "LeadTime(d)": horizon,
                "ComputeDevice": extract_compute_device(metadata),
                "Backend": extract_backend(metadata),
                "SourceDir": str(main_dir),
                "PredictionFile": str(prediction_file),
                "MetadataFile": metadata.get("_metadata_file", ""),
            }
            row.update({metric: metrics[metric] for metric in METRIC_COLUMNS})
            rows.append(row)
    return rows, missing


def build_wide_table(long_df):
    rows = []
    for model in long_df["Model"].drop_duplicates():
        model_df = long_df[long_df["Model"] == model].sort_values("LeadTime(d)")
        first_row = model_df.iloc[0]
        wide_row = {
            "Family": first_row["Family"],
            "Model": model,
            "ComputeDevice": ",".join(sorted(set(model_df["ComputeDevice"].dropna().astype(str)))),
        }
        for _, row in model_df.iterrows():
            horizon = int(row["LeadTime(d)"])
            for metric in METRIC_COLUMNS:
                wide_row[f"{metric}_H{horizon}"] = row[metric]
        rows.append(wide_row)
    return pd.DataFrame(rows).sort_values("Model", key=lambda s: s.map(MODEL_ORDER)).reset_index(drop=True)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, missing = collect_rows()
    if missing and not args.allow_missing:
        print("Missing formal prediction files:")
        for path in missing:
            print(path)
        raise SystemExit(1)

    long_df = pd.DataFrame(rows)
    if long_df.empty:
        print("No formal main prediction files were found.")
        raise SystemExit(1)

    long_df["ModelOrder"] = long_df["Model"].map(MODEL_ORDER)
    long_df = long_df.sort_values(["ModelOrder", "LeadTime(d)"]).reset_index(drop=True)
    wide_df = build_wide_table(long_df)

    long_csv = output_dir / "main_results_long.csv"
    wide_csv = output_dir / "main_results_wide.csv"
    long_xlsx = output_dir / "main_results_long.xlsx"
    wide_xlsx = output_dir / "main_results_wide.xlsx"
    workbook = output_dir / "fair_comparison_tables.xlsx"

    long_df.to_csv(long_csv, index=False, encoding="utf-8-sig")
    wide_df.to_csv(wide_csv, index=False, encoding="utf-8-sig")
    long_df.to_excel(long_xlsx, index=False)
    wide_df.to_excel(wide_xlsx, index=False)

    table5_path = output_dir / "table5_overall_latest.csv"
    yearly_high_flow_path = output_dir / "high_flow_yearly_latest.csv"

    with pd.ExcelWriter(workbook) as writer:
        long_df.to_excel(writer, sheet_name="main_long", index=False)
        wide_df.to_excel(writer, sheet_name="main_wide", index=False)
        if table5_path.exists():
            pd.read_csv(table5_path).to_excel(writer, sheet_name="high_flow_overall", index=False)
        if yearly_high_flow_path.exists():
            pd.read_csv(yearly_high_flow_path).to_excel(writer, sheet_name="high_flow_yearly", index=False)

    print(f"Rows exported: {len(long_df)}")
    print(f"Models exported: {len(wide_df)}")
    print(f"main_results_long: {long_csv}")
    print(f"main_results_wide: {wide_csv}")
    print(f"workbook: {workbook}")
    if missing:
        print(f"Missing files skipped: {len(missing)}")


if __name__ == "__main__":
    main()
