"""
feature_engineering.py
-----------------------
Builds ML-ready features from raw sensor + rainfall time series.

Expected raw input (one row per day per sensor location):
    date, location_id, water_level_m, rainfall_mm

Output: a feature table ready for model training / inference.
"""

import pandas as pd
import numpy as np


def load_raw_data(path: str) -> pd.DataFrame:
    """Load raw sensor + rainfall readings. CSV columns:
    date, location_id, water_level_m, rainfall_mm
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["location_id", "date"]).reset_index(drop=True)
    return df


def add_lag_features(df: pd.DataFrame, col: str, lags=(1, 7, 14, 30)) -> pd.DataFrame:
    """Add lagged values of `col`, computed per location_id."""
    for lag in lags:
        df[f"{col}_lag{lag}"] = df.groupby("location_id")[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, col: str, windows=(7, 14, 30)) -> pd.DataFrame:
    """Add rolling mean, std, and slope (rate of change) per location_id."""
    for w in windows:
        grp = df.groupby("location_id")[col]
        df[f"{col}_roll_mean_{w}"] = grp.transform(lambda s: s.rolling(w, min_periods=1).mean())
        df[f"{col}_roll_std_{w}"] = grp.transform(lambda s: s.rolling(w, min_periods=1).std())
        # slope = (value now - value w days ago) / w  -> rate of change
        df[f"{col}_slope_{w}"] = grp.transform(
            lambda s: (s - s.shift(w)) / w
        )
    return df


def add_rainfall_accumulation(df: pd.DataFrame, windows=(7, 14, 30)) -> pd.DataFrame:
    """Cumulative rainfall over trailing windows — key drought/flood driver."""
    for w in windows:
        df[f"rainfall_sum_{w}"] = (
            df.groupby("location_id")["rainfall_mm"]
            .transform(lambda s: s.rolling(w, min_periods=1).sum())
        )
    return df


def add_seasonal_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare each reading to the historical seasonal norm for that
    location + day-of-year. This is what lets the model say
    'unusually low for this time of year' rather than just 'low'.
    """
    df["day_of_year"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month

    # Historical mean/std of water_level for this location + day-of-year window
    # (using a +/-7 day window around day_of_year to smooth noise)
    baseline = (
        df.groupby(["location_id", "month"])["water_level_m"]
        .transform("mean")
    )
    baseline_std = (
        df.groupby(["location_id", "month"])["water_level_m"]
        .transform("std")
    )
    df["level_seasonal_baseline"] = baseline
    df["level_seasonal_std"] = baseline_std
    df["level_deviation_from_norm"] = df["water_level_m"] - baseline
    # z-score style deviation — how many std-devs off the seasonal norm
    df["level_zscore_seasonal"] = df["level_deviation_from_norm"] / baseline_std.replace(0, np.nan)

    return df


def build_feature_table(raw_csv_path: str) -> pd.DataFrame:
    """Full pipeline: raw CSV -> model-ready feature table."""
    df = load_raw_data(raw_csv_path)

    df = add_lag_features(df, "water_level_m")
    df = add_rolling_features(df, "water_level_m")
    df = add_rainfall_accumulation(df)
    df = add_seasonal_baseline(df)

    # Target: water level N days ahead (forecast horizon = 7 days, adjust as needed)
    HORIZON = 7
    df["target_water_level_future"] = (
        df.groupby("location_id")["water_level_m"].shift(-HORIZON)
    )

    # Drop rows where we don't have enough history or future target yet
    df = df.dropna(subset=["target_water_level_future"])

    return df


if __name__ == "__main__":
    # Example usage:
    # features = build_feature_table("sensor_data.csv")
    # features.to_csv("features_ready.csv", index=False)
    print("Run build_feature_table('your_sensor_data.csv') to generate features.")
