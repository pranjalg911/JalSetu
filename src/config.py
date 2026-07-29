"""
config.py
-----------
Central configuration: location metadata, recipient phone numbers,
file paths, and thresholds. Edit this file when adding a new
village/sensor/reservoir — don't touch the pipeline logic itself.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Data paths ---------------------------------------------------------
BOREWELL_SENSOR_CSV = os.path.join(BASE_DIR, "data", "sensor_data.csv")          # groundwater/borewell readings
RESERVOIR_SENSOR_CSV = os.path.join(BASE_DIR, "data", "reservoir_data.csv")      # reservoir/check-dam level readings
VILLAGE_STATIC_CSV = os.path.join(BASE_DIR, "data", "village_static_data.csv")   # geology/rainfall/cropping data for severity_engine
HISTORICAL_RAINFALL_CSV = os.path.join(BASE_DIR, "data", "historical_rainfall.csv")

# --- Model artifact paths ------------------------------------------------
BOREWELL_MODEL_PATH = os.path.join(BASE_DIR, "models", "water_level_model.joblib")
RESERVOIR_MODEL_PATH = os.path.join(BASE_DIR, "models", "reservoir_model.joblib")
SEVERITY_MODEL_PATH = os.path.join(BASE_DIR, "models", "severity_model.joblib")
SEVERITY_ENCODER_PATH = os.path.join(BASE_DIR, "models", "severity_encoder.joblib")
TRAIN_LOG_PATH = os.path.join(BASE_DIR, "models", "train_log.txt")

# --- Logs -----------------------------------------------------------------
ALERTS_LOG_PATH = os.path.join(BASE_DIR, "logs", "alerts_sent.log")

# --- GSM hardware ----------------------------------------------------------
GSM_PORT = "/dev/ttyUSB0"
GSM_BAUDRATE = 9600

# If using a cloud SMS gateway instead of a local modem:
SMS_GATEWAY_API_KEY = os.environ.get("SMS_GATEWAY_API_KEY", "")
SMS_SENDER_ID = "JALSETU"

# --- Locations: borewell/groundwater sensors -------------------------------
# Used by model_and_severity.py (per-location water level + drought/flood severity index)
BOREWELL_LOCATIONS_CONFIG = {
    "loc_001": {
        "name": "Village XYZ - Borewell 3",
        "recipients": ["+91XXXXXXXXXX"],
    },
    # "loc_002": {"name": "...", "recipients": [...]},
}

# --- Locations: reservoirs / check-dams / tanks ----------------------------
# Used by reservoir_forecaster.py (% capacity + days-to-critical)
RESERVOIR_CONFIG = {
    "res_001": {
        "name": "Village XYZ - Percolation Tank 1",
        "recipients": ["+91XXXXXXXXXX"],
        "critical_threshold_pct": 15.0,
        "warning_threshold_pct": 30.0,
    },
    # "res_002": {"name": "...", "recipients": [...], "critical_threshold_pct": 15.0, "warning_threshold_pct": 30.0},
}

# --- Village-level severity engine (severity_engine.py) --------------------
# village_id -> recipients, for villages scored by the GradientBoosting
# severity forecaster (rainfall deficit, groundwater trend, cropping, geology)
VILLAGE_SEVERITY_CONFIG = {
    "V001": {
        "name": "Village V001",
        "recipients": ["+91XXXXXXXXXX"],
    },
    # add more villages here as real static data is loaded
}

# --- Model / forecasting settings -----------------------------------------
BOREWELL_FORECAST_HORIZON_DAYS = 7
RESERVOIR_FORECAST_HORIZON_DAYS = 10
SEVERITY_LEAD_TIME_WEEKS = 6
RETRAIN_FREQUENCY_DAYS = 30   # how often to retrain all models (monthly by default)

# --- Severity thresholds for borewell/groundwater 0-100 index --------------
SEVERITY_BANDS = {
    "severe_drought_max": 20,
    "drought_warning_max": 35,
    "watch_low_max": 45,
    "normal_max": 55,
    "watch_high_max": 65,
    "flood_warning_max": 80,
    # anything above flood_warning_max = Severe Flood Risk
}
