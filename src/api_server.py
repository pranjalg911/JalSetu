"""
api_server.py
----------------
The bridge between the Python ML backend and the HTML frontend.

Run this instead of run_pipeline.py / showcase.py directly — it exposes
the same severity/advisory/twin-matching logic as a local web API, AND
serves the frontend HTML itself, so opening one URL gets you the whole
connected system:

    python api_server.py
    -> open http://localhost:5000

Persistence: leak reports, tanker requests, and user accounts are
stored in a local SQLite file (jalsetu.db) — not in-memory JS arrays,
so a page refresh (or a different device on the network) sees the
same data.

Chatbot: intents are matched against a small fixed set (tanker status,
water quality/severity, leak status, general status) and answered
ONLY from real backend data (village severity, logged leaks, logged
tanker requests). Anything outside that set gets an honest
"I don't have an answer for that yet" instead of a made-up response —
this is what fixes the "2 real answers, rest hallucinated" problem.
"""

import os
import sqlite3
import hashlib
import secrets
from datetime import datetime

from flask import Flask, jsonify, request, send_from_directory, g
from flask_cors import CORS

import config
from feature_engineering import build_feature_table
from model_and_severity import compute_severity_index, predict_future_level as predict_borewell
from reservoir_forecaster import (
    load_reservoir_data, engineer_reservoir_features,
    predict_future_level as predict_reservoir, compute_reservoir_risk,
)
from severity_engine import (
    generate_synthetic_villages, train_severity_model, predict_severity,
    add_advisories, build_twin_matcher, find_twins,
)

import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "jalsetu.db")
FRONTEND_DIR = BASE_DIR  # "Jal Setu - Full Platform.html" lives at repo root

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# SQLite persistence — leaks, tanker requests, users
# ---------------------------------------------------------------------------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS leaks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            village_id TEXT NOT NULL,
            location TEXT NOT NULL,
            severity TEXT NOT NULL,
            reported_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'reported'
        );

        CREATE TABLE IF NOT EXISTS tanker_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            village_id TEXT NOT NULL,
            requested_date TEXT NOT NULL,
            requested_time TEXT NOT NULL,
            requested_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending'
        );

        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,
            village_id TEXT
        );

        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            role TEXT NOT NULL,
            village_id TEXT,
            created_at TEXT NOT NULL
        );
    """)

    # Seed a few demo accounts if none exist yet (change/remove before real deployment)
    existing = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    if existing == 0:
        demo_users = [
            ("jalsakhi", "jalsakhi123", "jal_sakhi", "V023"),
            ("company", "company123", "company", None),
            ("admin", "admin123", "admin", None),
        ]
        for username, password, role, village_id in demo_users:
            pw_hash = hashlib.sha256(password.encode()).hexdigest()
            conn.execute(
                "INSERT INTO users (username, password_hash, role, village_id) VALUES (?, ?, ?, ?)",
                (username, pw_hash, role, village_id),
            )
        conn.commit()
        print("Seeded demo accounts: jalsakhi/jalsakhi123, company/company123, admin/admin123")
        print("CHANGE THESE before any real deployment.")

    conn.close()


def require_auth():
    """Returns the session row if the request has a valid token, else None."""
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not token:
        return None
    db = get_db()
    return db.execute("SELECT * FROM sessions WHERE token = ?", (token,)).fetchone()


# ---------------------------------------------------------------------------
# ML model loading (cached at process start — retrain via a separate script,
# not on every request, since training takes real time)
# ---------------------------------------------------------------------------
_state = {}


def load_models():
    print("Loading/training models — this happens once at server startup...")

    # Village severity engine
    if os.path.exists(config.VILLAGE_STATIC_CSV):
        village_df = pd.read_csv(config.VILLAGE_STATIC_CSV)
    else:
        village_df = pd.DataFrame()
    if village_df.empty:
        print("  No real village_static_data.csv rows — using synthetic demo villages.")
        village_df = generate_synthetic_villages()

    sev_model, sev_encoder, importances, (mae, r2) = train_severity_model(village_df)
    village_df = predict_severity(village_df, sev_model, sev_encoder)
    village_df = add_advisories(village_df)
    nn, topo_matrix, scaler, topo_encoder = build_twin_matcher(village_df)

    _state["village_df"] = village_df
    _state["village_model"] = sev_model
    _state["village_encoder"] = sev_encoder
    _state["village_importances"] = importances
    _state["village_nn"] = nn
    _state["village_topo_matrix"] = topo_matrix

    # Borewell + reservoir engines (best-effort — only if pre-trained model files exist)
    try:
        _state["borewell_model"] = joblib.load(config.BOREWELL_MODEL_PATH)
        _state["borewell_features"] = build_feature_table(config.BOREWELL_SENSOR_CSV)
    except Exception as e:
        print(f"  Borewell model not loaded ({e}) — train it first via train_model().")
        _state["borewell_model"] = None

    try:
        _state["reservoir_model"] = joblib.load(config.RESERVOIR_MODEL_PATH)
        res_df = load_reservoir_data(config.RESERVOIR_SENSOR_CSV)
        _state["reservoir_features"] = engineer_reservoir_features(res_df)
    except Exception as e:
        print(f"  Reservoir model not loaded ({e}) — train it first via train_reservoir_model().")
        _state["reservoir_model"] = None

    print("Models ready.")


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------
@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(force=True)
    username = data.get("username", "").strip()
    password = data.get("password", "")

    db = get_db()
    user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    if not user:
        return jsonify({"error": "Invalid username or password"}), 401

    pw_hash = hashlib.sha256(password.encode()).hexdigest()
    if pw_hash != user["password_hash"]:
        return jsonify({"error": "Invalid username or password"}), 401

    token = secrets.token_hex(24)
    db.execute(
        "INSERT INTO sessions (token, username, role, village_id, created_at) VALUES (?, ?, ?, ?, ?)",
        (token, user["username"], user["role"], user["village_id"], datetime.now().isoformat()),
    )
    db.commit()

    return jsonify({
        "token": token,
        "role": user["role"],
        "village_id": user["village_id"],
        "username": user["username"],
    })


@app.route("/api/logout", methods=["POST"])
def logout():
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    db = get_db()
    db.execute("DELETE FROM sessions WHERE token = ?", (token,))
    db.commit()
    return jsonify({"ok": True})


@app.route("/api/me")
def me():
    session = require_auth()
    if not session:
        return jsonify({"error": "Not authenticated"}), 401
    return jsonify({
        "username": session["username"],
        "role": session["role"],
        "village_id": session["village_id"],
    })


# ---------------------------------------------------------------------------
# Village / severity endpoints
# ---------------------------------------------------------------------------
def village_row_to_dict(row):
    return {
        "village_id": row["village_id"],
        "taluka": row["taluka"],
        "geology_type": row["geology_type"],
        "aquifer_recharge_potential": row["aquifer_recharge_potential"],
        "rainfall_deficit_pct_vs_normal": round(float(row["rainfall_deficit_pct_vs_normal"]), 1),
        "pct_area_sugarcane": round(float(row["pct_area_sugarcane"]), 1),
        "borewell_density_per_sqkm": round(float(row["borewell_density_per_sqkm"]), 1),
        "severity_score_predicted": float(row["severity_score_predicted"]),
        "severity_band": row["severity_band"],
        "recommended_action": row["recommended_action"],
    }


@app.route("/api/villages")
def api_villages():
    df = _state["village_df"]
    return jsonify([village_row_to_dict(row) for _, row in df.iterrows()])


@app.route("/api/village/<village_id>")
def api_village_detail(village_id):
    df = _state["village_df"]
    row_matches = df[df["village_id"] == village_id]
    if row_matches.empty:
        return jsonify({"error": "Village not found"}), 404

    row = row_matches.iloc[0]
    detail = village_row_to_dict(row)

    twins = find_twins(df, _state["village_nn"], _state["village_topo_matrix"], village_id, k=5)
    detail["twins"] = [
        {
            "village_id": t.village_id,
            "taluka": t.taluka,
            "geology_type": t.geology_type,
            "aquifer_recharge_potential": t.aquifer_recharge_potential,
            "severity_score_predicted": float(t.severity_score_predicted),
            "severity_band": t.severity_band,
            "topo_distance": float(t.topo_distance),
        }
        for _, t in twins.iterrows()
    ]
    return jsonify(detail)


@app.route("/api/feature-importance")
def api_feature_importance():
    imp = _state["village_importances"]
    return jsonify([{"feature": k, "importance": float(v)} for k, v in imp.head(8).items()])


# ---------------------------------------------------------------------------
# Borewell / reservoir endpoints
# ---------------------------------------------------------------------------
@app.route("/api/borewell/<location_id>")
def api_borewell(location_id):
    if _state.get("borewell_model") is None:
        return jsonify({"error": "Borewell model not trained yet"}), 503

    features = _state["borewell_features"]
    rows = features[features["location_id"] == location_id]
    if rows.empty:
        return jsonify({"error": "Location not found"}), 404

    latest = rows.iloc[-1]
    predicted = predict_borewell(_state["borewell_model"], latest)
    severity = compute_severity_index(
        predicted_level=predicted,
        seasonal_mean=latest["level_seasonal_baseline"],
        seasonal_std=latest["level_seasonal_std"],
        recent_slope=latest["water_level_m_slope_7"],
    )
    return jsonify({"location_id": location_id, "predicted_level_m": round(predicted, 2), **severity})


@app.route("/api/reservoir/<reservoir_id>")
def api_reservoir(reservoir_id):
    if _state.get("reservoir_model") is None:
        return jsonify({"error": "Reservoir model not trained yet"}), 503

    features = _state["reservoir_features"]
    rows = features[features["reservoir_id"] == reservoir_id]
    if rows.empty:
        return jsonify({"error": "Reservoir not found"}), 404

    latest = rows.iloc[-1]
    predicted = predict_reservoir(_state["reservoir_model"], latest)
    risk = compute_reservoir_risk(
        predicted_level_pct=predicted,
        current_level_pct=latest["level_pct_capacity"],
        recent_slope_per_day=latest["level_slope_7"],
    )
    return jsonify({"reservoir_id": reservoir_id, **risk})


# ---------------------------------------------------------------------------
# Leak reporting — persisted, with a real "how is severity decided" answer
# ---------------------------------------------------------------------------
@app.route("/api/leak", methods=["POST", "GET"])
def api_leak():
    db = get_db()
    if request.method == "POST":
        data = request.get_json(force=True)
        village_id = data.get("village_id")
        location = data.get("location", "Unspecified location")
        severity = data.get("severity", "Minor drip")
        if not village_id:
            return jsonify({"error": "village_id required"}), 400

        db.execute(
            "INSERT INTO leaks (village_id, location, severity, reported_at) VALUES (?, ?, ?, ?)",
            (village_id, location, severity, datetime.now().isoformat()),
        )
        db.commit()
        return jsonify({"ok": True})

    village_id = request.args.get("village_id")
    if village_id:
        rows = db.execute("SELECT * FROM leaks WHERE village_id = ? ORDER BY id DESC", (village_id,)).fetchall()
    else:
        rows = db.execute("SELECT * FROM leaks ORDER BY id DESC").fetchall()
    return jsonify([dict(r) for r in rows])


# ---------------------------------------------------------------------------
# Tanker requests — persisted, requires date + time (not auto-assigned)
# ---------------------------------------------------------------------------
@app.route("/api/tanker", methods=["POST", "GET"])
def api_tanker():
    db = get_db()
    if request.method == "POST":
        data = request.get_json(force=True)
        village_id = data.get("village_id")
        req_date = data.get("date")
        req_time = data.get("time")
        if not village_id or not req_date or not req_time:
            return jsonify({"error": "village_id, date, and time are all required"}), 400

        db.execute(
            "INSERT INTO tanker_requests (village_id, requested_date, requested_time, requested_at) VALUES (?, ?, ?, ?)",
            (village_id, req_date, req_time, datetime.now().isoformat()),
        )
        db.commit()
        return jsonify({"ok": True})

    village_id = request.args.get("village_id")
    if village_id:
        rows = db.execute(
            "SELECT * FROM tanker_requests WHERE village_id = ? ORDER BY id DESC", (village_id,)
        ).fetchall()
    else:
        rows = db.execute("SELECT * FROM tanker_requests ORDER BY id DESC").fetchall()
    return jsonify([dict(r) for r in rows])


# ---------------------------------------------------------------------------
# Chatbot — ONLY answers from real backend data, no hallucination.
# Unknown intents get an honest "don't have that yet" instead of invented text.
# ---------------------------------------------------------------------------
@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True)
    village_id = data.get("village_id")
    message = (data.get("message") or "").lower()

    df = _state["village_df"]
    row_matches = df[df["village_id"] == village_id]
    if row_matches.empty:
        return jsonify({"reply": "I don't have data for this village yet."})

    row = row_matches.iloc[0]
    db = get_db()

    if any(w in message for w in ["tanker", "truck", "water delivery"]):
        pending = db.execute(
            "SELECT * FROM tanker_requests WHERE village_id = ? ORDER BY id DESC LIMIT 1", (village_id,)
        ).fetchone()
        if pending:
            reply = (f"Latest tanker request: {pending['requested_date']} at {pending['requested_time']} "
                      f"(status: {pending['status']}).")
        else:
            reply = "No tanker requests logged for this village yet. Use 'Request tanker' to book one."
        return jsonify({"reply": reply})

    if any(w in message for w in ["leak", "pipe", "burst"]):
        count = db.execute("SELECT COUNT(*) c FROM leaks WHERE village_id = ?", (village_id,)).fetchone()["c"]
        reply = f"{count} leak(s) reported for this village so far. Use 'Report a leak' to log a new one."
        return jsonify({"reply": reply})

    if any(w in message for w in ["severity", "quality", "status", "drought", "water level", "how bad"]):
        reply = (f"Current severity score: {row['severity_score_predicted']}/100 ({row['severity_band']}). "
                  f"Advisory: {row['recommended_action']}")
        return jsonify({"reply": reply})

    return jsonify({"reply": "I can only answer questions about tanker status, leaks, and current water severity right now. Try asking about one of those."})


# ---------------------------------------------------------------------------
# Serve the frontend
# ---------------------------------------------------------------------------
@app.route("/")
def serve_frontend():
    return send_from_directory(FRONTEND_DIR, "Jal Setu - Full Platform.html")


if __name__ == "__main__":
    init_db()
    load_models()
    app.run(debug=True, port=5000)
