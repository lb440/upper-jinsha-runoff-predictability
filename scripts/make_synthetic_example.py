"""Create a non-sensitive example input table for schema and smoke tests."""

from pathlib import Path

import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    out_path = root / "data" / "example_model_input_synthetic.csv"
    rng = np.random.default_rng(222)
    dates = pd.date_range("2006-01-01", periods=730, freq="D")
    seasonal = np.sin(np.arange(len(dates)) * 2 * np.pi / 365.25)
    qz = 300 + 110 * seasonal + rng.normal(0, 15, len(dates))
    pz = np.maximum(0, rng.gamma(1.4, 0.004, len(dates)))
    pi = np.maximum(0, rng.gamma(1.2, 0.003, len(dates)))
    frame = pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Q_shigu": qz * 1.12 + 30 + rng.normal(0, 12, len(dates)),
        "Qz": qz,
        "Pz": pz,
        "Tz": 1.5 + 12 * seasonal + rng.normal(0, 1.2, len(dates)),
        "Ez": -np.maximum(0, 0.0015 + 0.001 * seasonal + rng.normal(0, 0.0002, len(dates))),
        "Sz": np.maximum(0, 0.22 - 0.16 * seasonal + rng.normal(0, 0.01, len(dates))),
        "Pi": pi,
        "Ti": 2.2 + 11 * seasonal + rng.normal(0, 1.2, len(dates)),
        "Ei": -np.maximum(0, 0.0012 + 0.0009 * seasonal + rng.normal(0, 0.0002, len(dates))),
        "Si": np.maximum(0, 0.18 - 0.13 * seasonal + rng.normal(0, 0.01, len(dates))),
    })
    frame.to_csv(out_path, index=False)
    print(f"Wrote synthetic example: {out_path}")


if __name__ == "__main__":
    main()
