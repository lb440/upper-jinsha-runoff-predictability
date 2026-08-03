"""Shared configuration for the public reproduction package.

Set ``RUNOFF_DATA_PATH`` to a lawful local copy of the processed input table.
The included CSV is synthetic and only demonstrates the required schema.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FILE_PATH = os.environ.get(
    "RUNOFF_DATA_PATH",
    str(REPOSITORY_ROOT / "data" / "example_model_input_synthetic.csv"),
)
TIME_COL = "Date"
TARGET_COL = "Q_shigu"

UPSTREAM_COLS = ["Qz", "Pz", "Tz", "Ez", "Sz"]
LOCAL_COLS = ["Pi", "Ti", "Ei", "Si"]
FEATURE_COLS = UPSTREAM_COLS + LOCAL_COLS

TRAIN_END_DATE = pd.Timestamp("2014-12-31")
VAL_END_DATE = pd.Timestamp("2016-12-31")
TEST_END_DATE = pd.Timestamp("2020-12-31")

SEQUENCE_CONFIGS = {
    "lstm": {"kim": 90, "horizon": 7},
    "gru": {"kim": 90, "horizon": 3},
    "mlp": {"kim": 90, "horizon": 15},
    "transformer": {"kim": 90, "horizon": 7},
    "lstm_transformer": {"kim": 90, "horizon": 7},
    "persistence": {"kim": 90, "horizon": 3},
    "xgboost": {"kim": 90, "horizon": 3},
}


def get_sequence_config(name):
    config = SEQUENCE_CONFIGS[name]
    return config["kim"], config["horizon"]


def get_split_indices(
    data_time,
    train_end_date=TRAIN_END_DATE,
    val_end_date=VAL_END_DATE,
    test_end_date=TEST_END_DATE,
):
    train_mask = data_time <= train_end_date
    val_mask = (data_time > train_end_date) & (data_time <= val_end_date)
    test_mask = (data_time > val_end_date) & (data_time <= test_end_date)
    return (
        np.where(train_mask)[0],
        np.where(val_mask)[0],
        np.where(test_mask)[0],
    )
