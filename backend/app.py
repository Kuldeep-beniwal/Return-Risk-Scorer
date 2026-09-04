"""
app.py
Flask backend for the Return-Risk Console.

Endpoints:
  GET  /api/health        -> {"status": "ok"}
  GET  /api/options       -> categorical choices + numeric ranges (drives the form)
  POST /api/score         -> {order fields} -> risk score + reason codes
  GET  /                  -> serves the frontend (frontend/index.html)

Run:
  cd backend
  pip install flask flask-cors  (flask-cors only needed if frontend is served
                                   from a different origin during development)
  python app.py
  -> open http://localhost:5000
"""

import os
from flask import Flask, jsonify, request, send_from_directory

from model_utils import RiskModel

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")

# Load models once at startup, not per-request.
risk_model = RiskModel()


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/options")
def options():
    return jsonify(risk_model.feature_metadata)


@app.post("/api/score")
def score():
    order = request.get_json(force=True, silent=True)
    if order is None:
        return jsonify({"errors": ["Request body must be valid JSON."]}), 400

    result = risk_model.score_order(order)
    if "errors" in result:
        return jsonify(result), 400
    return jsonify(result)


@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


if __name__ == "__main__":
    # host=0.0.0.0 so it's reachable if you're running this in a container/VM
    app.run(host="0.0.0.0", port=5000, debug=True)
