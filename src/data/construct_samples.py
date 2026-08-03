"""Construct chronological 90-day input windows for a selected lead time."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


FEATURES = ["Qz", "Pz", "Tz", "Ez", "Sz", "Pi", "Ti", "Ei", "Si"]
TARGET = "Q_shigu"


def parse_args():
    parser = argparse.ArgumentParser(description="Build chronological runoff samples.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/samples"))
    parser.add_argument("--lead-time", type=int, choices=[1, 3, 7, 15], required=True)
    parser.add_argument("--window", type=int, default=90)
    return parser.parse_args()


def split_name(target_date):
    if target_date <= pd.Timestamp("2014-12-31"):
        return "train"
    if target_date <= pd.Timestamp("2016-12-31"):
        return "validation"
    if target_date <= pd.Timestamp("2020-12-31"):
        return "test"
    return None


def main():
    args = parse_args()
    frame = pd.read_csv(args.input, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    required = ["Date", TARGET] + FEATURES
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")

    # Fit feature and target scalers using rows that can contribute to training windows.
    train_rows = frame[frame["Date"] <= pd.Timestamp("2014-12-31")]
    feature_scaler = MinMaxScaler().fit(train_rows[FEATURES])
    target_scaler = MinMaxScaler().fit(train_rows[[TARGET]])
    feature_values = feature_scaler.transform(frame[FEATURES])
    target_values = target_scaler.transform(frame[[TARGET]]).reshape(-1)

    groups = {"train": [], "validation": [], "test": []}
    for end_index in range(args.window - 1, len(frame) - args.lead_time):
        target_index = end_index + args.lead_time
        group = split_name(frame.loc[target_index, "Date"])
        if group is None:
            continue
        groups[group].append((
            feature_values[end_index - args.window + 1 : end_index + 1],
            target_values[target_index],
            frame.loc[target_index, "Date"].strftime("%Y-%m-%d"),
        ))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for group, records in groups.items():
        if not records:
            continue
        x, y, dates = zip(*records)
        np.savez_compressed(
            args.output_dir / f"{group}_h{args.lead_time}.npz",
            X=np.asarray(x), y=np.asarray(y), target_date=np.asarray(dates),
        )
    print(f"Wrote chronological samples to {args.output_dir}")


if __name__ == "__main__":
    main()
