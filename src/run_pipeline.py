"""
run_pipeline.py
-----------------
Master orchestrator. Run this daily (e.g. via cron) on the backend
server. It runs THREE forecasting engines and dispatches SMS alerts
via GSM for anything that crosses a threshold:

  1. Borewell/groundwater water-level forecast + drought-flood severity
     index (model_and_severity.py) — per sensor location, 0-100 scale.
  2. Reservoir/check-dam/tank low-water forecast (reservoir_forecaster.py)
     — per reservoir, % capacity + days-to-critical.
  3. Village-level severity forecast + topographic twin matching +
     advisory engine (severity_engine.py) — per village, static/seasonal
     features (rainfall, geology, cropping), refreshed weekly/monthly
     rather than daily since these inputs change slowly.

Each engine is independent — comment out a block if you're only
running a subset of sensors so far.
"""

import os
import joblib
import pandas as pd
from datetime import datetime

import config
from gsm_alert_dispatch import GSMModem, build_alert_message

# Engine imports
from feature_engineering import build_feature_table as build_borewell_features
from model_and_severity import compute_severity_index, predict_future_level as predict_borewell_level

from reservoir_forecaster import (
    load_reservoir_data, engineer_reservoir_features,
    predict_future_level as predict_reservoir_level, compute_reservoir_risk,
)

from severity_engine import (
    generate_synthetic_villages, train_severity_model, predict_severity,
    add_advisories, build_twin_matcher, find_twins,
)


def log_alert(source: str, location_name: str, message: str, sent_ok: bool, recipient: str):
    """Append every alert attempt to logs/alerts_sent.log for auditability."""
    timestamp = datetime.now().isoformat(timespec="seconds")
    status = "SENT" if sent_ok else "FAILED"
    line = f"{timestamp}\t{source}\t{location_name}\t{recipient}\t{status}\t{message}\n"
    with open(config.ALERTS_LOG_PATH, "a") as f:
        f.write(line)


def send_and_log(modem, source, location_name, message, recipients):
    for number in recipients:
        sent_ok = modem.send_sms(number, message)
        log_alert(source, location_name, message, sent_ok, number)
        print(f"  -> {number}: {'sent' if sent_ok else 'FAILED'}")


# ---------------------------------------------------------------------------
# 1. Borewell / groundwater engine
# ---------------------------------------------------------------------------
def run_borewell_engine(modem):
    print("\n" + "─" * 72)
    print("BOREWELL / GROUNDWATER ENGINE")
    print("─" * 72 + "\n")
    if not os.path.exists(config.BOREWELL_SENSOR_CSV):
        print("No borewell sensor data found, skipping.")
        return

    features = build_borewell_features(config.BOREWELL_SENSOR_CSV)
    model = joblib.load(config.BOREWELL_MODEL_PATH)

    for location_id, loc_config in config.BOREWELL_LOCATIONS_CONFIG.items():
        loc_rows = features[features["location_id"] == location_id]
        if loc_rows.empty:
            print(f"No data for {location_id}, skipping.")
            continue

        latest = loc_rows.iloc[-1]
        predicted_level = predict_borewell_level(model, latest)
        severity = compute_severity_index(
            predicted_level=predicted_level,
            seasonal_mean=latest["level_seasonal_baseline"],
            seasonal_std=latest["level_seasonal_std"],
            recent_slope=latest["water_level_m_slope_7"],
        )
        print(f"  Location        : {loc_config['name']}")
        print(f"  Predicted level : {predicted_level:.2f} m")
        print(f"  Severity score  : {severity['score']} / 100")
        print(f"  Category        : {severity['category']}")
        print()

        if severity["category"] == "Normal":
            continue

        message = build_alert_message(loc_config["name"], severity, predicted_level)
        send_and_log(modem, "borewell", loc_config["name"], message, loc_config["recipients"])


# ---------------------------------------------------------------------------
# 2. Reservoir engine
# ---------------------------------------------------------------------------
def run_reservoir_engine(modem):
    print("\n" + "─" * 72)
    print("RESERVOIR ENGINE")
    print("─" * 72 + "\n")
    if not os.path.exists(config.RESERVOIR_SENSOR_CSV):
        print("No reservoir sensor data found, skipping.")
        return

    df = load_reservoir_data(config.RESERVOIR_SENSOR_CSV)
    features = engineer_reservoir_features(df)
    model = joblib.load(config.RESERVOIR_MODEL_PATH)

    for reservoir_id, res_config in config.RESERVOIR_CONFIG.items():
        rows = features[features["reservoir_id"] == reservoir_id]
        if rows.empty:
            print(f"No data for {reservoir_id}, skipping.")
            continue

        latest = rows.iloc[-1]
        predicted_level = predict_reservoir_level(model, latest)
        risk = compute_reservoir_risk(
            predicted_level_pct=predicted_level,
            current_level_pct=latest["level_pct_capacity"],
            recent_slope_per_day=latest["level_slope_7"],
            critical_threshold_pct=res_config["critical_threshold_pct"],
            warning_threshold_pct=res_config["warning_threshold_pct"],
        )
        print(f"  Location            : {res_config['name']}")
        print(f"  Predicted capacity  : {risk['predicted_level_pct']}%")
        print(f"  Category            : {risk['category']}")
        if risk["days_to_critical"]:
            print(f"  Days to critical    : {risk['days_to_critical']}")
        print()

        if risk["category"] == "Normal":
            continue

        days_note = f" Est. {risk['days_to_critical']} days to critical." if risk["days_to_critical"] else ""
        message = (
            f"[Jal Setu Alert] {res_config['name']}: {risk['category']} "
            f"({risk['predicted_level_pct']}% capacity, 10-day forecast).{days_note}"
        )[:300]
        send_and_log(modem, "reservoir", res_config["name"], message, res_config["recipients"])


# ---------------------------------------------------------------------------
# 3. Village-level severity engine (run weekly/monthly, not daily — inputs
#    like rainfall deficit/cropping pattern don't change day to day)
# ---------------------------------------------------------------------------
def run_village_severity_engine(modem):
    print("\n" + "─" * 72)
    print("VILLAGE SEVERITY ENGINE")
    print("─" * 72 + "\n")
    df = pd.DataFrame()
    if os.path.exists(config.VILLAGE_STATIC_CSV):
        df = pd.read_csv(config.VILLAGE_STATIC_CSV)

    if df.empty:
        print("No real village static data found (or file has no rows) — using synthetic demo data.\n")
        df = generate_synthetic_villages()

    model, encoder, importances, (mae, r2) = train_severity_model(df)
    df = predict_severity(df, model, encoder)
    df = add_advisories(df)
    nn, topo_matrix, scaler, topo_encoder = build_twin_matcher(df)

    for village_id, v_config in config.VILLAGE_SEVERITY_CONFIG.items():
        row_matches = df[df["village_id"] == village_id]
        if row_matches.empty:
            print(f"No data for {village_id}, skipping.")
            continue
        row = row_matches.iloc[0]
        print(f"  Village          : {v_config['name']}")
        print(f"  Severity score   : {row.severity_score_predicted} / 100")
        print(f"  Band             : {row.severity_band}")
        print(f"  Advisory:")
        for action in row.recommended_action.split(" | "):
            print(f"    - {action}")
        print()

        if row.severity_band in ("Low", "Moderate"):
            continue  # only SMS-alert High/Critical villages; Moderate handled via routine monitoring per advisory text

        message = (
            f"[Jal Setu Alert] {v_config['name']}: {row.severity_band} drought/flood risk "
            f"(score {row.severity_score_predicted}/100, {config.SEVERITY_LEAD_TIME_WEEKS}-wk outlook). "
            f"{row.recommended_action}"
        )[:300]
        send_and_log(modem, "village_severity", v_config["name"], message, v_config["recipients"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_daily_pipeline():
    modem = GSMModem(port=config.GSM_PORT, baudrate=config.GSM_BAUDRATE)
    try:
        run_borewell_engine(modem)
        run_reservoir_engine(modem)
        run_village_severity_engine(modem)
    finally:
        modem.close()


if __name__ == "__main__":
    run_daily_pipeline()