"""
gsm_alert_dispatch.py
-----------------------
Sends severity alerts via GSM (SMS), so alerts reach Jal Sakhis /
villagers even with zero internet connectivity - just cellular signal.

Two supported paths, pick based on your hardware setup:

  1. Local GSM modem (SIM800L / SIM900 / similar) connected via serial
     -> uses AT commands directly. Good for a field gateway device
        sitting next to the water sensor with its own SIM card.

  2. Cloud SMS gateway API (if the backend server has internet, but
     you still want SMS as the delivery channel to reach villagers
     who have no data/internet, only basic phones).
     -> e.g. Twilio, or an Indian SMS gateway provider (TextLocal,
        MSG91, Fast2SMS) via simple HTTPS POST.

Use path 1 when the sensor/gateway itself is remote and offline.
Use path 2 when your backend is centralized and you're SMS-ing
out to end users' phones.
"""

import time
import serial  # pip install pyserial
import requests


# ------------------------------------------------------------------
# PATH 1: Direct GSM modem via serial + AT commands
# ------------------------------------------------------------------

class GSMModem:
    """
    Minimal AT-command wrapper for SIM800L/SIM900-style GSM modules.
    Connects over serial (USB-to-TTL or directly if on same board).
    """

    def __init__(self, port: str = "/dev/ttyUSB0", baudrate: int = 9600, timeout: int = 5):
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        time.sleep(2)  # let modem initialize

    def _send_at(self, command: str, wait: float = 1.0) -> str:
        self.ser.write((command + "\r\n").encode())
        time.sleep(wait)
        response = self.ser.read(self.ser.in_waiting or 200).decode(errors="ignore")
        return response

    def send_sms(self, phone_number: str, message: str) -> bool:
        """
        Sends an SMS using text mode AT commands.
        phone_number should be in international format, e.g. +919876543210
        """
        try:
            self._send_at("AT")                      # check modem responsive
            self._send_at("AT+CMGF=1")                # set text mode
            self._send_at(f'AT+CMGS="{phone_number}"')
            # Ctrl+Z (0x1A) terminates the message body and sends it
            self.ser.write((message + chr(26)).encode())
            time.sleep(3)
            response = self.ser.read(self.ser.in_waiting or 200).decode(errors="ignore")
            return "OK" in response or "+CMGS" in response
        except Exception as e:
            print(f"GSM send failed: {e}")
            return False

    def close(self):
        self.ser.close()


# ------------------------------------------------------------------
# PATH 2: Cloud SMS gateway (backend has internet, recipients don't need it)
# ------------------------------------------------------------------

def send_sms_via_gateway(phone_number: str, message: str, api_key: str, sender_id: str = "JALSETU") -> bool:
    """Fast2SMS quick SMS API (India)."""
    try:
        clean_number = phone_number.replace("+91", "").replace(" ", "")
        response = requests.post(
            "https://www.fast2sms.com/dev/bulkV2",
            headers={"authorization": api_key},
            data={
                "route": "q",              # quick/test route — uses trial credits
                "message": message,
                "language": "english",
                "flash": 0,
                "numbers": clean_number,
            },
            timeout=10,
        )
        result = response.json()
        return result.get("return", False)
    except Exception as e:
        print(f"Gateway SMS send failed: {e}")
        return False


# ------------------------------------------------------------------
# Alert message construction — ties severity_index output to wording
# ------------------------------------------------------------------

def build_alert_message(location_name: str, severity: dict, predicted_level: float) -> str:
    """
    Turns the severity dict from compute_severity_index() into a short,
    actionable SMS (SMS has a 160-char limit per segment — keep it tight).
    """
    category = severity["category"]
    direction = severity["direction"]

    action_map = {
        "Severe Drought Risk": "Conserve water urgently. Restrict non-essential use. Contact Jal Sakhi.",
        "Drought Warning": "Water levels falling fast. Begin conservation measures now.",
        "Watch (Low)": "Water levels below normal for this season. Monitor closely.",
        "Normal": "Water levels normal. No action needed.",
        "Watch (High)": "Water levels rising above normal. Monitor drainage.",
        "Flood Warning": "Rising water levels. Move valuables to higher ground, stay alert.",
        "Severe Flood Risk": "Flood risk high. Evacuate low-lying areas, follow local authority guidance.",
    }
    action = action_map.get(category, "Monitor situation.")

    msg = f"[Jal Setu Alert] {location_name}: {category}. Level: {predicted_level:.2f}m. {action}"
    return msg[:300]  # keep within a couple of SMS segments


# ------------------------------------------------------------------
# End-to-end example: prediction -> severity -> SMS
# ------------------------------------------------------------------

def dispatch_alert_for_location(
    location_name: str,
    predicted_level: float,
    severity: dict,
    recipient_numbers: list,
    modem: GSMModem = None,
    gateway_api_key: str = None,
):
    """
    Wire this into your pipeline after model_and_severity.compute_severity_index().
    Only sends an SMS if severity warrants it (skip 'Normal' to avoid alert fatigue).
    """
    if severity["category"] == "Normal":
        return  # no alert needed

    message = build_alert_message(location_name, severity, predicted_level)

    for number in recipient_numbers:
        if modem is not None:
            sent = modem.send_sms(number, message)
        elif gateway_api_key is not None:
            sent = send_sms_via_gateway(number, message, gateway_api_key)
        else:
            raise ValueError("Provide either a GSMModem instance or a gateway_api_key")

        print(f"Alert to {number}: {'sent' if sent else 'FAILED'}")


if __name__ == "__main__":
    # Example (uncomment and configure for real use):
    #
    # from model_and_severity import compute_severity_index
    #
    # severity = compute_severity_index(
    #     predicted_level=2.1, seasonal_mean=3.5, seasonal_std=0.6, recent_slope=-0.05
    # )
    #
    # modem = GSMModem(port="/dev/ttyUSB0")
    # dispatch_alert_for_location(
    #     location_name="Village XYZ - Borewell 3",
    #     predicted_level=2.1,
    #     severity=severity,
    #     recipient_numbers=["+91XXXXXXXXXX"],
    #     modem=modem,
    # )
    # modem.close()
    print("Import dispatch_alert_for_location() into your pipeline.")
