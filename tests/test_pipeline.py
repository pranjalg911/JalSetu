"""
test_pipeline.py
-------------------
Basic sanity tests — run these after any code change to catch
regressions before they reach the field. Not exhaustive, but catches
the class of bug that matters most: severity/risk scoring silently
flipping direction (e.g. flagging a drought as normal).

Run with: python3 -m pytest test_pipeline.py -v
(or just: python3 test_pipeline.py)
"""

import sys
import os
import tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from feature_engineering import build_feature_table
from model_and_severity import train_model, compute_severity_index, predict_future_level
from reservoir_forecaster import engineer_reservoir_features, train_reservoir_model, predict_future_level as predict_reservoir_level, compute_reservoir_risk
from severity_engine import generate_synthetic_villages, train_severity_model, predict_severity, severity_band

TMP_DIR = tempfile.gettempdir()  # cross-platform temp dir (works on Windows, Mac, Linux)


def _make_borewell_csv_with_drought(path):
    dates = pd.date_range("2024-01-01", periods=730, freq="D")
    np.random.seed(0)
    doy = dates.dayofyear.to_numpy()
    seasonal = 3.5 + 1.2 * np.sin(2 * np.pi * (doy / 365))
    noise = np.random.normal(0, 0.15, len(dates))
    level = np.array(seasonal + noise)
    level[600:650] -= np.linspace(0, 1.8, 50)
    level[650:700] -= 1.8
    rainfall = np.clip(np.random.normal(5, 8, len(dates)), 0, None)
    rainfall[600:650] *= 0.1

    df = pd.DataFrame({"date": dates, "location_id": "loc_test", "water_level_m": level, "rainfall_mm": rainfall})
    df.to_csv(path, index=False)


def test_borewell_severity_flags_drought_event(tmp_path=None):
    """A model-predicted level during an injected drought dip should score
    as drought (score < 50), not as flood or normal."""
    tmp_path = tmp_path or os.path.join(TMP_DIR, "test_sensor_data.csv")
    _make_borewell_csv_with_drought(tmp_path)
    features = build_feature_table(tmp_path)
    model = train_model(features, model_path=os.path.join(TMP_DIR, "test_borewell_model.joblib"))

    drought_row = features[features["date"] == "2025-08-30"].iloc[0]
    pred = predict_future_level(model, drought_row)
    severity = compute_severity_index(
        predicted_level=pred,
        seasonal_mean=drought_row["level_seasonal_baseline"],
        seasonal_std=drought_row["level_seasonal_std"],
        recent_slope=drought_row["water_level_m_slope_7"],
    )
    assert severity["direction"] == "drought", f"Expected drought, got {severity}"
    assert severity["score"] < 50, f"Expected score < 50 for drought event, got {severity['score']}"
    print("PASS: borewell severity correctly flags injected drought event")


def test_reservoir_risk_flags_low_reservoir(tmp_path=None):
    """A steadily draining reservoir should be flagged Warning/Critical, not Normal."""
    tmp_path = tmp_path or os.path.join(TMP_DIR, "test_reservoir_data.csv")
    dates = pd.date_range("2025-01-01", periods=365, freq="D")
    np.random.seed(1)
    level = np.clip(80 - np.linspace(0, 55, 365) + np.random.normal(0, 1.5, 365), 5, 100)
    inflow = np.clip(np.random.exponential(1.5, 365) - 1, 0, None)

    df = pd.DataFrame({"timestamp": dates, "reservoir_id": "res_test", "level_pct_capacity": level, "inflow_mm": inflow})
    df.to_csv(tmp_path, index=False)

    features = engineer_reservoir_features(df)
    model = train_reservoir_model(features, model_path=os.path.join(TMP_DIR, "test_reservoir_model.joblib"))

    latest = features.iloc[-1]
    pred = predict_reservoir_level(model, latest)
    risk = compute_reservoir_risk(
        predicted_level_pct=pred,
        current_level_pct=latest["level_pct_capacity"],
        recent_slope_per_day=latest["level_slope_7"],
    )
    assert risk["category"] in ("Warning Low", "Critical Low"), f"Expected Warning/Critical, got {risk}"
    print("PASS: reservoir engine correctly flags draining reservoir")


def test_village_severity_bands_are_monotonic():
    """severity_band() thresholds must be monotonic: higher score never
    yields a lower band."""
    scores = [0, 15, 29, 30, 40, 54, 55, 60, 74, 75, 90, 100]
    band_order = {"Low": 0, "Moderate": 1, "High": 2, "Critical": 3}
    prev_rank = -1
    for s in scores:
        band = severity_band(s)
        rank = band_order[band]
        assert rank >= prev_rank, f"Non-monotonic band at score {s}: {band}"
        prev_rank = rank
    print("PASS: village severity bands are monotonic with score")


def test_village_severity_engine_runs_end_to_end():
    df = generate_synthetic_villages(n=20, seed=1)
    model, encoder, importances, (mae, r2) = train_severity_model(df)
    df = predict_severity(df, model, encoder)
    assert "severity_score_predicted" in df.columns
    assert df["severity_score_predicted"].between(0, 100).all()
    print(f"PASS: village severity engine runs end to end (MAE={mae:.2f}, R2={r2:.2f})")


if __name__ == "__main__":
    test_borewell_severity_flags_drought_event()
    test_reservoir_risk_flags_low_reservoir()
    test_village_severity_bands_are_monotonic()
    test_village_severity_engine_runs_end_to_end()
    print("\nAll sanity tests passed.")