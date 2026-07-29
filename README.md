# Jal Setu — Predictive Backend

Drought/flood severity forecasting, reservoir low-water forecasting, and
GSM/SMS alert dispatch for village-scale IoT water monitoring.

## Folder structure

```
jal-setu-backend/
├── data/
│   ├── sensor_data.csv            # raw borewell/groundwater sensor readings
│   ├── reservoir_data.csv         # raw reservoir/check-dam level readings
│   ├── village_static_data.csv    # geology/rainfall/cropping data per village (for severity_engine)
│   └── historical_rainfall.csv    # IMD/CHIRPS data, used to calibrate seasonal baselines
│
├── models/                        # trained model artifacts (generated, not hand-written)
│   ├── water_level_model.joblib
│   ├── reservoir_model.joblib
│   ├── severity_model.joblib
│   └── train_log.txt
│
├── src/
│   ├── config.py                  # all locations, phone numbers, thresholds, paths — edit this to add sensors
│   ├── feature_engineering.py     # lag/rolling/seasonal features for borewell time series
│   ├── model_and_severity.py      # borewell water-level forecast -> 0-100 drought/flood severity index
│   ├── reservoir_forecaster.py    # reservoir % capacity forecast -> risk category + days-to-critical
│   ├── severity_engine.py         # village-level severity forecaster + topographic twin matcher + advisory engine
│   ├── gsm_alert_dispatch.py      # SMS sending: local GSM modem (AT commands) or cloud SMS gateway
│   └── run_pipeline.py            # master orchestrator — run this daily via cron
│
├── logs/
│   └── alerts_sent.log            # append-only record of every SMS sent (timestamp, engine, recipient, status, message)
│
├── tests/
│   └── test_pipeline.py           # sanity checks — run after any code change
│
├── requirements.txt
└── README.md
```

## The three forecasting engines

1. **Borewell / groundwater** (`model_and_severity.py`) — per-sensor time
   series forecast of water level (RandomForest), converted into a 0-100
   drought↔flood severity score using deviation from the seasonal norm
   plus current trend.

2. **Reservoir / check-dam / tank** (`reservoir_forecaster.py`) — same
   time-series approach but framed as % capacity, with a "days to
   critical threshold" estimate based on current decline rate. Use this
   for any sensor reporting fill-level rather than groundwater depth.

3. **Village-level severity + advisory** (`severity_engine.py`) —
   GradientBoosting model over static/seasonal features (rainfall
   deficit, groundwater trend, geology, cropping pattern, borewell
   density) producing a 6-week-ahead severity score, a "topographic twin"
   match (structurally similar villages, for peer learning — matched on
   geology/terrain only, not on human choices like cropping), and a rule-based
   advisory (crop shift, recharge structure siting, borewell permit freeze,
   tanker pre-booking).

All three are independent — you can run just one if you're only
deploying one sensor type so far. `run_pipeline.py` runs all three and
routes anything above threshold to SMS.

## Setup

```bash
cd jal-setu-backend
pip install -r requirements.txt --break-system-packages
```

### 1. Train the models (one-time, then retrain periodically)

```python
from src.feature_engineering import build_feature_table
from src.model_and_severity import train_model
from src import config

features = build_feature_table(config.BOREWELL_SENSOR_CSV)
train_model(features, model_path=config.BOREWELL_MODEL_PATH)
```

Do the equivalent for `reservoir_forecaster.train_reservoir_model()` and
`severity_engine.train_severity_model()`.

### 2. Edit `src/config.py`

Add your real sensor/village IDs, display names, and recipient phone
numbers under `BOREWELL_LOCATIONS_CONFIG`, `RESERVOIR_CONFIG`, and
`VILLAGE_SEVERITY_CONFIG`.

### 3. Run the pipeline

```bash
cd src
python3 run_pipeline.py
```

Set this up as a daily cron job. The village-severity engine's inputs
change slowly (rainfall deficit, cropping pattern) — running it weekly
or monthly instead of daily is reasonable and saves compute.

### 4. Run tests after any code change

```bash
cd tests
python3 test_pipeline.py
```

## GSM hardware notes

`gsm_alert_dispatch.py` supports two paths:

- **Local GSM modem** (SIM800L / SIM900) wired via USB-to-TTL to the
  gateway device sitting next to the sensor. Uses AT commands over
  serial — no internet required at all, only cellular signal. Default
  port assumed: `/dev/ttyUSB0` at 9600 baud (edit in `config.py` if
  your wiring differs).
- **Cloud SMS gateway** (MSG91 / Fast2SMS / Twilio-style HTTPS API) — use
  this if your backend server has internet access but you still want to
  reach villagers via plain SMS (they may have no data plan/smartphone).

Only one path is needed depending on your deployment — `run_pipeline.py`
currently uses the local modem path (`GSMModem`) by default.

## Known limitations / what to validate before relying on this

- Severity thresholds and score-band cutoffs (in both `model_and_severity.py`
  and `severity_engine.py`) are reasonable starting points, not tuned on
  real historical drought/flood events for your specific villages —
  validate against known past events before trusting the categories.
- `severity_engine.py`'s synthetic village generator is a stand-in;
  replace `generate_synthetic_villages()` with a real data loader
  (rainfall records, groundwater board readings, satellite NDVI,
  cropping surveys) before using it for real advisories.
- Model accuracy scales with how much real sensor history you've
  accumulated — expect noisier predictions in the first season or two
  of deployment.
  
# Jal Setu 💧

**A women-led IoT + predictive analytics solution for rural water stress in Maharashtra**

Submitted for the CIF Water Innovation Challenge.

## The problem

Villages across drought-prone regions like Marathwada often find out they're in a water crisis only after wells run dry, tankers are booked in a panic, and crops are already lost. There's no early-warning system that tells a village *this week* that it will be in trouble in 4–8 weeks — and no easy way to see what similar villages have done to cope.

## What Jal Setu does

Jal Setu combines low-cost IoT groundwater sensors with a predictive analytics engine (PSTM — Predictive Severity & Topographic Twin-Matching Engine) that turns raw sensor and satellite data into a concrete, village-specific action plan, weeks before a crisis hits.

The engine has three linked modules:

1. **Severity Forecaster** — a Gradient Boosting model that predicts a 0–100 water-stress severity score 4–8 weeks ahead, using rainfall trends, groundwater depth/decline rate, crop mix, borewell density, and satellite-derived vegetation stress (NDVI).
2. **Topographic Twin Matcher** — finds villages with matching geology, slope, and aquifer type (deliberately ignoring crop/borewell choices) so a stressed village can see what worked for a structural "twin."
3. **Advisory Rule Engine** — converts severity + crop mix into specific actions: crop-shift advisories, recharge structure siting, borewell permit freezes, or tanker pre-booking.

## Repo contents

| File | What it is |
|---|---|
| `jal_setu_algorithm.py` | The full PSTM pipeline — runs end-to-end on synthetic sample data modeled on Marathwada-type villages |
| `Jal_Setu_Algorithm_Explained.docx` | A plain-language walkthrough of how the algorithm works, with a worked example |

## Running it

```bash
pip install numpy pandas scikit-learn
python jal_setu_algorithm.py
```

This runs the model on 40 synthetic villages and prints:
- Model performance (MAE, R²) on held-out villages
- Top feature importances — what actually drives predicted severity
- A sample severity forecast and topographic twin match
- Advisory output for the most severe villages

> The severity label used here is synthetic, standing in for real historical groundwater-committee records. A real deployment would swap in that historical data — the pipeline structure stays identical.

## Why this approach

- **Explainable, not a black box** — feature importances and rule triggers can be shown to a village water committee in plain language.
- **Separates geology from behavior** — twin-matching compares villages on what they physically are, not the choices farmers made, so the comparison stays fair.
- **Actionable lead time** — a 4–8 week forecast turns water stress from a crisis response into a scheduling problem.

## Team

Built by Manasi Deshpande as part of the Jal Setu submission for the CIF Water Innovation Challenge.
