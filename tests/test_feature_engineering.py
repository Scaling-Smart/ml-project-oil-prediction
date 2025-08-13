import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

# Set up isolated MLflow environment before importing the module

# Ensure the project root is on the Python path for imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

#_tmpdir = tempfile.mkdtemp()
#os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")     
#_artifact_dir = os.path.join(_tmpdir, "artifacts")
#os.environ["MLFLOW_ARTIFACT_URI"] = "artifacts"
#os.environ["REGISTERED_MODEL_NAME"] = "TestModel"
#os.environ["MLFLOW_EXPERIMENT"] = "TestExperiment"
#os.makedirs(_artifact_dir, exist_ok=True)

from train_wti_mlflow_fallback import make_lags, make_rolls

BASE_COLS = ["WTI", "DJU", "Gold", "SP500", "US10Y", "USD_INDEX"]


def _sample_frame():
    return pd.DataFrame({
        "WTI": [1, 2, 3, 4, 5],
        "DJU": [5, 4, 3, 2, 1],
        "Gold": [10, 20, 30, 40, 50],
        "SP500": [100, 110, 120, 130, 140],
        "US10Y": [0.5, 0.6, 0.7, 0.8, 0.9],
        "USD_INDEX": [80, 82, 84, 86, 88],
    })


def test_make_lags():
    df = _sample_frame()
    lags = (1, 2)
    lagged = make_lags(df, BASE_COLS, lags=lags)
    for col in BASE_COLS:
        for lag in lags:
            expected = df[col].shift(lag)
            pd.testing.assert_series_equal(
                lagged[f"{col}_lag{lag}"], expected, check_names=False
            )


def test_make_rolls():
    df = _sample_frame()
    windows = (2, 3)
    rolled = make_rolls(df, BASE_COLS, windows=windows)
    for col in BASE_COLS:
        for w in windows:
            mean_expected = df[col].rolling(w).mean()
            std_expected = df[col].rolling(w).std()
            pd.testing.assert_series_equal(
                rolled[f"{col}_rollmean{w}"], mean_expected, check_names=False
            )
            pd.testing.assert_series_equal(
                rolled[f"{col}_rollstd{w}"], std_expected, check_names=False
            )
