"""
reservoir_forecaster.py
--------------------------
Forecasts low water levels in reservoirs/check-dams/tanks from IoT
sensor time series (ultrasonic/pressure level sensors reporting depth
or % capacity at regular intervals).

This is separate from severity_engine.py (which scores village-level
drought/flood risk from static + seasonal features). This module is
specifically for a single reservoir's live sensor feed, predicting
"how many days until this reservoir crosses a critical low threshold."

Expected raw input (one row per reading per reservoir):
    timestamp, reservoir_id, level_pct_capacity, inflow_mm (optional rainfall proxy)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

FEATURE_COLS = [
    "level_pct_capacity",
    "level_lag1", "level_lag3", "level_lag7", "level_lag14",
    "level_roll_mean_7", "level_roll_mean_14",
    "level_slope_3", "level_slope_7", "level_slope_14",
    "inflow_sum_7", "inflow_sum_14",
    "days_since_last_rain",
]
TARGET_COL = "target_level_future"
FORECAST_HORIZON_DAYS = 10  # predict level this many days ahead


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def load_reservoir_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values(["reservoir_id", "timestamp"]).reset_index(drop=True)
    return df


def engineer_reservoir_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    grp = df.groupby("reservoir_id")["level_pct_capacity"]

    for lag in (1, 3, 7, 14):
        df[f"level_lag{lag}"] = grp.shift(lag)

    for w in (7, 14):
        df[f"level_roll_mean_{w}"] = grp.transform(lambda s: s.rolling(w, min_periods=1).mean())

    for w in (3, 7, 14):
        df[f"level_slope_{w}"] = grp.transform(lambda s: (s - s.shift(w)) / w)

    if "inflow_mm" in df.columns:
        inflow_grp = df.groupby("reservoir_id")["inflow_mm"]
        for w in (7, 14):
            df[f"inflow_sum_{w}"] = inflow_grp.transform(lambda s: s.rolling(w, min_periods=1).sum())

        # days since last meaningful rain event (>2mm), per reservoir
        def days_since_rain(series):
            had_rain = series > 2
            out = []
            counter = 999
            for val in had_rain:
                counter = 0 if val else counter + 1
                out.append(counter)
            return pd.Series(out, index=series.index)

        df["days_since_last_rain"] = df.groupby("reservoir_id")["inflow_mm"].transform(days_since_rain)
    else:
        df["inflow_sum_7"] = 0
        df["inflow_sum_14"] = 0
        df["days_since_last_rain"] = 999

    df[TARGET_COL] = df.groupby("reservoir_id")["level_pct_capacity"].shift(-FORECAST_HORIZON_DAYS)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    return df


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
def train_reservoir_model(df: pd.DataFrame, model_path: str = "reservoir_model.joblib"):
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    tscv = TimeSeriesSplit(n_splits=5)
    maes, rmses = [], []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        maes.append(mean_absolute_error(y_val, preds))
        rmses.append(np.sqrt(mean_squared_error(y_val, preds)))

    print(f"Reservoir model cross-val MAE: {np.mean(maes):.2f}% capacity | RMSE: {np.mean(rmses):.2f}%")

    final_model = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1)
    final_model.fit(X, y)
    joblib.dump(final_model, model_path)

    importance = pd.Series(final_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nTop feature importances:")
    print(importance.head(6).to_string())

    return final_model


def predict_future_level(model, latest_row: pd.Series) -> float:
    X = latest_row[FEATURE_COLS].to_frame().T
    return float(model.predict(X)[0])


# ---------------------------------------------------------------------------
# Risk scoring: predicted level -> days-to-critical + category
# ---------------------------------------------------------------------------
def compute_reservoir_risk(predicted_level_pct: float, current_level_pct: float,
                            recent_slope_per_day: float,
                            critical_threshold_pct: float = 15.0,
                            warning_threshold_pct: float = 30.0) -> dict:
    """
    Turns a predicted % capacity into a risk category and, if the
    reservoir is declining, an estimated days-until-critical based on
    the current slope (rate of decline per day).
    """
    if predicted_level_pct <= critical_threshold_pct:
        category = "Critical Low"
    elif predicted_level_pct <= warning_threshold_pct:
        category = "Warning Low"
    elif predicted_level_pct >= 95:
        category = "Near Full / Overflow Risk"
    else:
        category = "Normal"

    days_to_critical = None
    if recent_slope_per_day < -0.01 and current_level_pct > critical_threshold_pct:
        days_to_critical = int((current_level_pct - critical_threshold_pct) / abs(recent_slope_per_day))

    return {
        "predicted_level_pct": round(predicted_level_pct, 1),
        "category": category,
        "days_to_critical": days_to_critical,
    }


if __name__ == "__main__":
    print("Import load_reservoir_data(), engineer_reservoir_features(), "
          "train_reservoir_model(), and compute_reservoir_risk() into your pipeline.")
