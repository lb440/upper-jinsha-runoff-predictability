import argparse

from baseline_shared_utils import FEATURE_COLS, default_bo_output_dir, default_model_output_dir, load_best_params
from gru_core import run_gru_experiment
from model_shared_config import get_sequence_config


def main():
    default_kim, default_horizon = get_sequence_config("gru")
    parser = argparse.ArgumentParser(description="Run the GRU model.")
    parser.add_argument("--kim", type=int, default=default_kim)
    parser.add_argument("--horizon", type=int, default=default_horizon)
    parser.add_argument("--param-path", type=str, default=None)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--random-seed", type=int, default=222)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-6)
    args = parser.parse_args()

    param_path = args.param_path or (
        f"{default_bo_output_dir('GRU', args.horizon)}/gru_best_params.json"
    )
    params = load_best_params(param_path)
    save_path = args.save_path or default_model_output_dir("GRU", args.horizon)

    run_gru_experiment(
        kim=args.kim,
        horizon=args.horizon,
        output_dir=save_path,
        params=params,
        feature_cols=FEATURE_COLS,
        random_seed=args.random_seed,
        max_epochs=args.max_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        weight_decay=args.weight_decay,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        scheduler_min_lr=args.scheduler_min_lr,
    )


if __name__ == "__main__":
    main()
