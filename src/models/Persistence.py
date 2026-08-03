import argparse

from baseline_shared_utils import default_model_output_dir
from model_shared_config import get_sequence_config
from persistence_core import run_persistence_experiment


def main():
    default_kim, default_horizon = get_sequence_config("persistence")
    parser = argparse.ArgumentParser(description="Run the persistence baseline.")
    parser.add_argument("--kim", type=int, default=default_kim)
    parser.add_argument("--horizon", type=int, default=default_horizon)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--compute-device", choices=["auto", "cuda", "cpu"], default="cuda")
    args = parser.parse_args()

    save_path = args.save_path or default_model_output_dir("Persistence", args.horizon)
    run_persistence_experiment(
        kim=args.kim,
        horizon=args.horizon,
        output_dir=save_path,
        compute_device=args.compute_device,
    )


if __name__ == "__main__":
    main()
