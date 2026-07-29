"""
severity_engine.py
---------------------
Village-level drought/flood severity forecasting, topographic twin
matching, and advisory rule engine.

This is a refactor of the original standalone script into importable
functions, so run_pipeline.py can call it daily/weekly without
re-fitting encoders inconsistently. Logic is unchanged from the
original design: same features, same synthetic label formula, same
advisory rules.

Real deployment: replace generate_synthetic_villages() with a loader
that reads actual village-level data (rainfall records, groundwater
board readings, satellite NDVI, cropping pattern surveys) into the
same column schema.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

NUM_FEATURE_COLS = [
    "avg_annual_rainfall_mm", "slope_pct", "elevation_m", "distance_to_river_km",
    "groundwater_depth_m", "groundwater_trend_m_per_month", "pct_area_sugarcane",
    "borewell_density_per_sqkm", "ndvi_deficit_pct", "rainfall_deficit_pct_vs_normal",
    "historical_drought_events_5yr", "historical_flood_events_5yr",
]
CAT_FEATURE_COLS = ["geology_type", "soil_type", "aquifer_recharge_potential"]

TOPO_NUM_COLS = ["slope_pct", "elevation_m", "distance_to_river_km", "avg_annual_rainfall_mm"]
TOPO_CAT_COLS = ["geology_type", "soil_type", "aquifer_recharge_potential"]


# ---------------------------------------------------------------------------
# 1. DATA — synthetic generator for demo/dev; swap for a real loader in prod
# ---------------------------------------------------------------------------
def generate_synthetic_villages(n: int = 40, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic village-level dataset with Marathwada-style feature ranges.
    Replace with real data ingestion (rainfall records, groundwater board
    readings, satellite NDVI, cropping surveys) keyed to the same schema
    for production use.
    """
    rng = np.random.default_rng(seed)
    talukas = ["Beed", "Latur", "Osmanabad", "Jalna", "Parbhani"]

    df = pd.DataFrame({
        "village_id": [f"V{i+1:03d}" for i in range(n)],
        "taluka": rng.choice(talukas, n),
        "avg_annual_rainfall_mm": rng.normal(700, 90, n).clip(450, 950),
        "geology_type": rng.choice(["basalt_hard_rock", "alluvial", "basalt_weathered"], n, p=[0.6, 0.15, 0.25]),
        "soil_type": rng.choice(["black_clay", "murum", "loamy"], n, p=[0.55, 0.30, 0.15]),
        "slope_pct": rng.uniform(0.5, 12, n),
        "elevation_m": rng.normal(520, 60, n),
        "distance_to_river_km": rng.uniform(0.5, 18, n),
        "aquifer_recharge_potential": rng.choice(["low", "medium", "high"], n, p=[0.5, 0.35, 0.15]),
        "groundwater_depth_m": rng.uniform(15, 260, n),
        "groundwater_trend_m_per_month": rng.uniform(0.2, 4.0, n),
        "pct_area_sugarcane": rng.uniform(2, 45, n),
        "borewell_density_per_sqkm": rng.uniform(3, 40, n),
        "ndvi_deficit_pct": rng.uniform(0, 35, n),
        "rainfall_deficit_pct_vs_normal": rng.uniform(-10, 45, n),
        "historical_drought_events_5yr": rng.poisson(1.8, n),
        "historical_flood_events_5yr": rng.poisson(0.4, n),
    })

    recharge_weight = df["aquifer_recharge_potential"].map({"low": 1.0, "medium": 0.55, "high": 0.2})
    geology_weight = df["geology_type"].map({"basalt_hard_rock": 1.0, "basalt_weathered": 0.6, "alluvial": 0.25})

    severity = (
        0.28 * (df["rainfall_deficit_pct_vs_normal"] / 45) * 100 +
        0.22 * (df["groundwater_trend_m_per_month"] / 4.0) * 100 +
        0.18 * (df["pct_area_sugarcane"] / 45) * 100 +
        0.12 * (df["borewell_density_per_sqkm"] / 40) * 100 +
        0.12 * recharge_weight * 100 +
        0.08 * geology_weight * 100
    )
    severity = (severity + rng.normal(0, 4, n)).clip(0, 100)
    df["severity_score_actual"] = severity.round(1)
    return df


# ---------------------------------------------------------------------------
# 2. SEVERITY FORECASTER — Gradient Boosting Regressor
# ---------------------------------------------------------------------------
def severity_band(score: float) -> str:
    if score < 30:
        return "Low"
    elif score < 55:
        return "Moderate"
    elif score < 75:
        return "High"
    else:
        return "Critical"


def train_severity_model(df: pd.DataFrame):
    """
    Fits the OneHotEncoder ONCE and reuses it for both training and
    later inference, so category encoding stays consistent (the
    original script fit two separate encoder instances, which is
    fragile if category sets differ between calls).

    Returns: model, fitted_encoder, feature_importances (Series), (mae, r2)
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = encoder.fit_transform(df[CAT_FEATURE_COLS])
    X_num = df[NUM_FEATURE_COLS].values
    X = np.hstack([X_num, X_cat])
    y = df["severity_score_actual"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = GradientBoostingRegressor(n_estimators=250, learning_rate=0.05, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred_test)
    r2 = r2_score(y_test, pred_test)

    feature_names = NUM_FEATURE_COLS + list(encoder.get_feature_names_out(CAT_FEATURE_COLS))
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)

    return model, encoder, importances, (mae, r2)


def predict_severity(df: pd.DataFrame, model, encoder) -> pd.DataFrame:
    """Adds severity_score_predicted and severity_band columns to df (copy)."""
    df = df.copy()
    X_cat = encoder.transform(df[CAT_FEATURE_COLS])
    X_num = df[NUM_FEATURE_COLS].values
    X = np.hstack([X_num, X_cat])

    df["severity_score_predicted"] = model.predict(X).clip(0, 100).round(1)
    df["severity_band"] = df["severity_score_predicted"].apply(severity_band)
    df["lead_time_weeks"] = 6  # forecast horizon this model is tuned for
    return df


# ---------------------------------------------------------------------------
# 3. TOPOGRAPHIC TWIN MATCHER — structural similarity only (no behavioral
#    features like sugarcane % or borewell density, so twins are matched on
#    geology/terrain, not on human choices)
# ---------------------------------------------------------------------------
def build_twin_matcher(df: pd.DataFrame):
    """Returns (nn_model, topo_matrix, scaler, encoder) for reuse in find_twins()."""
    scaler = StandardScaler()
    topo_num_scaled = scaler.fit_transform(df[TOPO_NUM_COLS])

    topo_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    topo_cat_encoded = topo_encoder.fit_transform(df[TOPO_CAT_COLS])

    topo_matrix = np.hstack([topo_num_scaled, topo_cat_encoded])
    nn = NearestNeighbors(n_neighbors=6, metric="euclidean")
    nn.fit(topo_matrix)
    return nn, topo_matrix, scaler, topo_encoder


def find_twins(df: pd.DataFrame, nn, topo_matrix, village_id: str, k: int = 5) -> pd.DataFrame:
    i = df.index[df.village_id == village_id][0]
    dist, ind = nn.kneighbors([topo_matrix[i]], n_neighbors=k + 1)
    ind = ind[0][1:]  # drop itself
    twins = df.loc[ind, [
        "village_id", "taluka", "geology_type", "aquifer_recharge_potential",
        "severity_score_predicted", "severity_band", "pct_area_sugarcane",
    ]].copy()
    twins["topo_distance"] = dist[0][1:].round(3)
    return twins.sort_values("severity_score_predicted")


# ---------------------------------------------------------------------------
# 4. ADVISORY RULE ENGINE — severity band + crop mix -> concrete action
# ---------------------------------------------------------------------------
def advise(row) -> str:
    actions = []
    if row.severity_band in ("High", "Critical"):
        if row.pct_area_sugarcane > 20:
            actions.append("Crop-shift advisory: move part of sugarcane area to drip-irrigated low-water crop")
        if row.aquifer_recharge_potential == "low" and row.geology_type.startswith("basalt"):
            actions.append("Site a check-dam/percolation tank at nearest low-slope contour (engineered recharge, not borewell)")
        if row.borewell_density_per_sqkm > 20:
            actions.append("Freeze new borewell permits in this zone via digital registry")
        actions.append("Pre-book tanker allocation for next month; alert Jal Sakhi + Village Water Committee now")
    elif row.severity_band == "Moderate":
        actions.append("Increase monitoring frequency; advisory-only WhatsApp/voice alert to Jal Sakhi")
    else:
        actions.append("No action needed; continue routine monthly monitoring")
    return " | ".join(actions)


def add_advisories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["recommended_action"] = df.apply(advise, axis=1)
    return df


# ---------------------------------------------------------------------------
# Standalone demo (only runs when executed directly, not on import)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    df = generate_synthetic_villages()
    model, encoder, importances, (mae, r2) = train_severity_model(df)
    df = predict_severity(df, model, encoder)
    df = add_advisories(df)

    nn, topo_matrix, scaler, topo_encoder = build_twin_matcher(df)

    print("=" * 100)
    print("MODEL PERFORMANCE (held-out test villages)")
    print("=" * 100)
    print(f"MAE  : {mae:.2f} severity points")
    print(f"R^2  : {r2:.3f}")

    print("\n" + "=" * 100)
    print("TOP FEATURE IMPORTANCE")
    print("=" * 100)
    print(importances.head(8).to_string())

    print("\n" + "=" * 100)
    print("SAMPLE SEVERITY FORECAST — top 10 villages, 6-week lead time")
    print("=" * 100)
    sample = df[["village_id", "taluka", "geology_type", "severity_score_actual",
                 "severity_score_predicted", "severity_band"]].sort_values(
        "severity_score_predicted", ascending=False).head(10)
    print(sample.to_string(index=False))

    worst_village = df.sort_values("severity_score_predicted", ascending=False).iloc[0]["village_id"]
    print("\n" + "=" * 100)
    print(f"TOPOGRAPHIC TWIN MATCH for most severe village: {worst_village}")
    print("=" * 100)
    print(find_twins(df, nn, topo_matrix, worst_village).to_string(index=False))

    print("\n" + "=" * 100)
    print("ADVISORY OUTPUT for the 5 most severe villages")
    print("=" * 100)
    for _, row in df.sort_values("severity_score_predicted", ascending=False).head(5).iterrows():
        print(f"\n{row.village_id} ({row.taluka}) | Severity: {row.severity_score_predicted} [{row.severity_band}]")
        print(f"  -> {row.recommended_action}")
