import argparse

from baseline_shared_utils import FEATURE_COLS, default_bo_output_dir, default_model_output_dir, load_best_params
from model_shared_config import get_sequence_config
from xgboost_core import run_xgboost_experiment


def main():
    default_kim, default_horizon = get_sequence_config("xgboost")
    parser = argparse.ArgumentParser(description="Run the XGBoost baseline.")
    parser.add_argument("--kim", type=int, default=default_kim)
    parser.add_argument("--horizon", type=int, default=default_horizon)
    parser.add_argument("--backend", choices=["auto", "xgboost", "hist_gbdt"], default="auto")
    parser.add_argument("--param-path", type=str, default=None)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--random-seed", type=int, default=222)
    parser.add_argument("--compute-device", choices=["auto", "cuda", "cpu"], default="cuda")
    args = parser.parse_args()

    param_path = args.param_path or (
        f"{default_bo_output_dir('XGBoost', args.horizon)}/xgboost_best_params.json"
    )
    params = load_best_params(param_path)
    if args.backend != "auto":
        params["resolved_backend"] = args.backend
    if args.compute_device != "auto":
        params["resolved_compute_device"] = args.compute_device
    save_path = args.save_path or default_model_output_dir("XGBoost", args.horizon)

    run_xgboost_experiment(
        kim=args.kim,
        horizon=args.horizon,
        output_dir=save_path,
        params=params,
        feature_cols=FEATURE_COLS,
        backend=args.backend,
        random_seed=args.random_seed,
        compute_device=args.compute_device,
    )


if __name__ == "__main__":
    main()
