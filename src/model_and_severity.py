"""
model_and_severity.py
-----------------------
Trains the water-level forecasting model and converts predictions
into a drought/flood severity index + category.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

FEATURE_COLS = [
    "water_level_m",
    "water_level_m_lag1", "water_level_m_lag7", "water_level_m_lag14", "water_level_m_lag30",
    "water_level_m_roll_mean_7", "water_level_m_roll_mean_14", "water_level_m_roll_mean_30",
    "water_level_m_roll_std_7", "water_level_m_roll_std_14", "water_level_m_roll_std_30",
    "water_level_m_slope_7", "water_level_m_slope_14", "water_level_m_slope_30",
    "rainfall_sum_7", "rainfall_sum_14", "rainfall_sum_30",
    "level_deviation_from_norm", "level_zscore_seasonal",
]

TARGET_COL = "target_water_level_future"


def train_model(df: pd.DataFrame, model_path: str = "water_level_model.joblib"):
    """
    Trains a RandomForest regressor using time-series-aware validation
    (never shuffle time series data — train on past, validate on future).
    """
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    tscv = TimeSeriesSplit(n_splits=5)
    maes, rmses = [], []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        maes.append(mean_absolute_error(y_val, preds))
        rmses.append(np.sqrt(mean_squared_error(y_val, preds)))

    print(f"Cross-val MAE: {np.mean(maes):.3f} m  |  RMSE: {np.mean(rmses):.3f} m")

    # Final fit on all available data
    final_model = RandomForestRegressor(
        n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    final_model.fit(X, y)
    joblib.dump(final_model, model_path)
    print(f"Model saved to {model_path}")

    # Feature importance — useful to show judges/officials what drives predictions
    importance = pd.Series(final_model.feature_importances_, index=FEATURE_COLS)
    print("\nTop feature importances:")
    print(importance.sort_values(ascending=False).head(8))

    return final_model


def predict_future_level(model, latest_row: pd.Series) -> float:
    """Predict water level N days ahead for a single location's latest reading."""
    X = latest_row[FEATURE_COLS].to_frame().T
    return float(model.predict(X)[0])


# ---------------------------------------------------------------------
# Severity index: converts a predicted level into a 0-100 score + category
# ---------------------------------------------------------------------

def compute_severity_index(
    predicted_level: float,
    seasonal_mean: float,
    seasonal_std: float,
    recent_slope: float,
) -> dict:
    """
    Combines the predicted deviation from seasonal norm with the
    current trend (slope) into a single severity score.

    Returns dict with:
        score: 0-100 (0 = extreme drought risk, 50 = normal, 100 = extreme flood risk)
        category: text label
        direction: 'drought' or 'flood' or 'normal'
    """
    if seasonal_std == 0 or np.isnan(seasonal_std):
        seasonal_std = 1e-6  # avoid divide-by-zero

    zscore = (predicted_level - seasonal_mean) / seasonal_std

    # Blend zscore (where you'll be) with slope (how fast you're getting there)
    # weights: 70% predicted deviation, 30% current trend speed
    slope_component = np.clip(recent_slope / seasonal_std, -3, 3)
    combined = 0.7 * zscore + 0.3 * slope_component

    # Map combined score (~ -3 to +3) onto 0-100 scale, 50 = normal
    score = float(np.clip(50 + combined * 16.6, 0, 100))

    if score < 20:
        category, direction = "Severe Drought Risk", "drought"
    elif score < 35:
        category, direction = "Drought Warning", "drought"
    elif score < 45:
        category, direction = "Watch (Low)", "drought"
    elif score <= 55:
        category, direction = "Normal", "normal"
    elif score <= 65:
        category, direction = "Watch (High)", "flood"
    elif score <= 80:
        category, direction = "Flood Warning", "flood"
    else:
        category, direction = "Severe Flood Risk", "flood"

    return {"score": round(score, 1), "category": category, "direction": direction}


if __name__ == "__main__":
    print("Import train_model() and compute_severity_index() into your pipeline.")
