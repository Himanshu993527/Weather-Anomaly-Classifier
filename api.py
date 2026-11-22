"""Prediction API for the React 'Predict' tab.   python api.py   (after running train.py)"""
import datetime as dt
import json

import joblib
import numpy as np
from flask import Flask, jsonify, request

from backend.config import ART, CLASSES, MODEL_SPECS
from backend.prepare_data import make_features

app = Flask(__name__)
_models = {}


@app.after_request
def cors(resp):                                              # allow the React dev server / standalone page to call the API
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


def load_model(key):
    import tensorflow as tf
    if key not in _models:
        _models[key] = tf.keras.models.load_model(ART / f"{key}.keras", compile=False)
    return _models[key]


def available():
    return [k for k in MODEL_SPECS if (ART / f"{k}.keras").exists()]


@app.get("/health")
def health():
    return jsonify(status="ok", models=available())


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return ("", 204)
    try:
        j = request.get_json(force=True)
        mx, mn, lat, lon = (float(j[k]) for k in ("max_temp", "min_temp", "latitude", "longitude"))
        date = dt.date.fromisoformat(str(j["date"]))
        if not (-60 <= mn <= mx <= 60):
            return jsonify(error="Need -60 <= min_temp <= max_temp <= 60 (degrees C)"), 400
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return jsonify(error="Latitude/longitude out of range"), 400
        key = j.get("model") or (available() or [None])[0]
        if key not in available():
            return jsonify(error=f"Model '{key}' not trained yet. Run train.py first."), 400
        stations = json.loads((ART / "stations.json").read_text())
        sid = stations.get(str(j.get("station_id", "")).strip(), 0)             # 0 = unknown station
        x = joblib.load(ART / "scaler.joblib").transform(make_features([mx], [mn], [lat], [lon], [date.timetuple().tm_yday]))
        prob = load_model(key).predict({"num": x.astype("float32"), "station": np.array([sid], dtype="int32")}, verbose=0)[0]
        i = int(prob.argmax())
        return jsonify(model=key, predicted=CLASSES[i], confidence=float(prob[i]), station_known=bool(sid),
                       probabilities={c: float(p) for c, p in zip(CLASSES, prob)})
    except (KeyError, ValueError) as e:
        return jsonify(error=f"Invalid input: {e}"), 400


if __name__ == "__main__":
    app.run(port=5000)
