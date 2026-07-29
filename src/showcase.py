"""
showcase.py
-------------
One-command demo that shows the FULL story of the Jal Setu backend:

  PART 1 — Village Severity Ranking + Topographic Twin Matching
           (the core algorithm: which villages are most at risk,
           what drives that risk, and which "twin" villages show
           what worked elsewhere)

  PART 2 — Multi-location Borewell + Reservoir Forecasting
           (sensor-driven predictions across several simulated sites,
           not just one)

  PART 3 — GSM/SMS Alert Dispatch (simulated)
           (the same alerts a real deployment would send)

  PART 4 — Validation Summary
           (proof the system was tested, not just built)

IMPORTANT: All data here is SYNTHETIC/DEMO data, not real sensor
readings — see README.md for what to replace before real deployment.

Run with: python showcase.py
"""

import sys
import io
import contextlib

import config
import gsm_alert_dispatch


# ---------------------------------------------------------------------------
# Mock GSM modem — same as demo.py, boxed SMS previews instead of real sending
# ---------------------------------------------------------------------------
class MockModem:
    def __init__(self, *a, **k):
        pass  # stay quiet here; showcase prints its own section headers

    def send_sms(self, number, message):
        print()
        print("  ┌─ SMS ALERT " + "─" * 50)
        print(f"  │  To      : {number}")
        print(f"  │  Message : {message}")
        print("  └" + "─" * 62)
        return True

    def close(self):
        pass


gsm_alert_dispatch.GSMModem = MockModem


def banner(text, char="="):
    print()
    print(char * 74)
    print(text)
    print(char * 74)
    print()


def subheader(text):
    print()
    print("─" * 74)
    print(text)
    print("─" * 74)
    print()


# ---------------------------------------------------------------------------
# PART 1 — Village severity ranking + twin matching
# ---------------------------------------------------------------------------
def run_part1_village_ranking():
    banner("PART 1 — VILLAGE SEVERITY RANKING + TOPOGRAPHIC TWIN MATCHING")
    print("  Synthetic demo data: 40 villages, Marathwada-style feature ranges.\n")

    from severity_engine import (
        generate_synthetic_villages, train_severity_model, predict_severity,
        add_advisories, build_twin_matcher, find_twins,
    )

    df = generate_synthetic_villages()
    model, encoder, importances, (mae, r2) = train_severity_model(df)
    df = predict_severity(df, model, encoder)
    df = add_advisories(df)
    nn, topo_matrix, scaler, topo_encoder = build_twin_matcher(df)

    subheader("Model performance (held-out test villages)")
    print(f"  MAE : {mae:.2f} severity points")
    print(f"  R^2 : {r2:.3f}")

    subheader("Top drivers of predicted severity")
    for feat, imp in importances.head(6).items():
        bar = "█" * int(imp * 40)
        print(f"  {feat:<32} {bar} {imp:.3f}")

    subheader("Top 10 most severe villages (6-week outlook)")
    top10 = df.sort_values("severity_score_predicted", ascending=False).head(10)
    print(f"  {'Village':<10} {'Taluka':<12} {'Geology':<18} {'Score':>6}  Band")
    print(f"  {'-'*10} {'-'*12} {'-'*18} {'-'*6}  {'-'*8}")
    for _, row in top10.iterrows():
        print(f"  {row.village_id:<10} {row.taluka:<12} {row.geology_type:<18} "
              f"{row.severity_score_predicted:>6.1f}  {row.severity_band}")

    worst = top10.iloc[0]
    subheader(f"Topographic twins for most severe village: {worst.village_id}")
    print("  (matched on geology/slope/rainfall only — not on human choices")
    print("   like cropping pattern, so a village can see what worked for")
    print("   structurally similar peers)\n")
    twins = find_twins(df, nn, topo_matrix, worst.village_id, k=5)
    print(f"  {'Village':<10} {'Taluka':<12} {'Recharge':<10} {'Score':>6}  {'Dist':>6}")
    print(f"  {'-'*10} {'-'*12} {'-'*10} {'-'*6}  {'-'*6}")
    for _, row in twins.iterrows():
        print(f"  {row.village_id:<10} {row.taluka:<12} {row.aquifer_recharge_potential:<10} "
              f"{row.severity_score_predicted:>6.1f}  {row.topo_distance:>6.2f}")

    subheader("Advisory output — top 3 most severe villages")
    for _, row in top10.head(3).iterrows():
        print(f"  {row.village_id} ({row.taluka}) — {row.severity_band} ({row.severity_score_predicted}/100)")
        for action in row.recommended_action.split(" | "):
            print(f"    - {action}")
        print()

    return df


# ---------------------------------------------------------------------------
# PART 2 — Multi-location borewell + reservoir forecasting
# ---------------------------------------------------------------------------
def run_part2_multi_location_forecasts():
    banner("PART 2 — MULTI-LOCATION SENSOR FORECASTING (Borewell + Reservoir)")

    import joblib
    from feature_engineering import build_feature_table
    from model_and_severity import compute_severity_index, predict_future_level as predict_borewell
    from reservoir_forecaster import (
        load_reservoir_data, engineer_reservoir_features,
        predict_future_level as predict_reservoir, compute_reservoir_risk,
    )

    subheader("Borewell / Groundwater forecasts")
    features = build_feature_table(config.BOREWELL_SENSOR_CSV)
    model = joblib.load(config.BOREWELL_MODEL_PATH)

    # Demo across a few simulated site labels reusing the same sensor series
    # (swap in real per-site CSVs once you have multiple physical sensors)
    demo_borewell_sites = ["Borewell 3 (XYZ)", "Borewell 7 (ABC)", "Borewell 1 (PQR)"]
    latest = features[features["location_id"] == "loc_001"].iloc[-1]
    pred = predict_borewell(model, latest)
    severity = compute_severity_index(
        predicted_level=pred,
        seasonal_mean=latest["level_seasonal_baseline"],
        seasonal_std=latest["level_seasonal_std"],
        recent_slope=latest["water_level_m_slope_7"],
    )
    print(f"  {'Site':<22} {'Predicted (m)':>14} {'Score':>7}  Category")
    print(f"  {'-'*22} {'-'*14} {'-'*7}  {'-'*14}")
    for site in demo_borewell_sites:
        print(f"  {site:<22} {pred:>14.2f} {severity['score']:>7.1f}  {severity['category']}")
    print("\n  (same underlying sensor series shown across sites for demo —")
    print("   wire in real per-site CSVs once multiple physical sensors exist)")

    subheader("Reservoir / Check-dam forecasts")
    df = load_reservoir_data(config.RESERVOIR_SENSOR_CSV)
    res_features = engineer_reservoir_features(df)
    res_model = joblib.load(config.RESERVOIR_MODEL_PATH)

    demo_reservoir_sites = ["Percolation Tank 1", "Check Dam 2", "Village Tank A"]
    latest_res = res_features.iloc[-1]
    pred_res = predict_reservoir(res_model, latest_res)
    risk = compute_reservoir_risk(
        predicted_level_pct=pred_res,
        current_level_pct=latest_res["level_pct_capacity"],
        recent_slope_per_day=latest_res["level_slope_7"],
    )
    print(f"  {'Site':<22} {'Predicted (%)':>14} {'Days-to-crit':>13}  Category")
    print(f"  {'-'*22} {'-'*14} {'-'*13}  {'-'*14}")
    for site in demo_reservoir_sites:
        days = risk["days_to_critical"] if risk["days_to_critical"] else "-"
        print(f"  {site:<22} {risk['predicted_level_pct']:>14} {str(days):>13}  {risk['category']}")


# ---------------------------------------------------------------------------
# PART 3 — Real alert dispatch through the actual pipeline
# ---------------------------------------------------------------------------
def run_part3_alert_dispatch():
    banner("PART 3 — GSM/SMS ALERT DISPATCH (simulated modem)")
    print("  Running the actual production pipeline (run_pipeline.py) —")
    print("  same code path that would run on real hardware.\n")

    import run_pipeline
    run_pipeline.run_daily_pipeline()


# ---------------------------------------------------------------------------
# PART 4 — Validation summary
# ---------------------------------------------------------------------------
def run_part4_validation():
    banner("PART 4 — VALIDATION SUMMARY")
    print("  Running automated sanity tests (tests/test_pipeline.py)...\n")

    import subprocess
    import os

    test_path = os.path.join(os.path.dirname(__file__), "..", "tests", "test_pipeline.py")
    result = subprocess.run([sys.executable, test_path], capture_output=True, text=True)

    # Only show PASS lines + final summary, not the training noise
    for line in result.stdout.splitlines():
        if line.startswith("PASS") or "All sanity tests passed" in line:
            print(f"  ✓ {line}")

    if result.returncode != 0:
        print("\n  ⚠ One or more tests FAILED — see full output below:\n")
        print(result.stdout)
        print(result.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    banner("JAL SETU — FULL SYSTEM SHOWCASE", "█")
    print("  NOTE: All data shown is SYNTHETIC/DEMO data, not real sensor")
    print("        readings. This showcases system capability, not real")
    print("        current conditions in any actual village.")

    run_part1_village_ranking()
    run_part2_multi_location_forecasts()
    run_part3_alert_dispatch()
    run_part4_validation()

    banner("SHOWCASE COMPLETE", "█")