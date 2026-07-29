"""
demo.py
---------
Clean demo run of the full Jal Setu pipeline using a mock GSM modem.
No real hardware or SMS gateway needed — shows the full prediction ->
severity scoring -> alert generation flow end-to-end.

IMPORTANT: This currently runs on SYNTHETIC/DEMO data (data/sensor_data.csv,
data/reservoir_data.csv, and severity_engine's generated villages) —
not real sensor readings. Replace these with real data before treating
any output as a real prediction.
"""

import gsm_alert_dispatch


class MockModem:
    def __init__(self, *a, **k):
        print("[Demo] GSM modem connected (simulated)\n")

    def send_sms(self, number, message):
        print()
        print("  ┌─ SMS ALERT " + "─" * 50)
        print(f"  │  To      : {number}")
        print(f"  │  Message : {message}")
        print("  └" + "─" * 62)
        print()
        return True

    def close(self):
        print("[Demo] GSM modem disconnected")


gsm_alert_dispatch.GSMModem = MockModem

import run_pipeline


def banner(text, char="="):
    print()
    print(char * 72)
    print(text)
    print(char * 72)
    print()


banner("JAL SETU — PREDICTIVE WATER SEVERITY & ALERT SYSTEM (DEMO MODE)")
print("  NOTE: Running on SYNTHETIC demo data, not real sensor readings.")
print("        Replace data/*.csv with real exports before trusting output.")
print()

run_pipeline.run_daily_pipeline()

banner("Demo complete — see logs/alerts_sent.log for the full alert history", "-")