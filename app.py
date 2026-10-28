
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import math

app = Flask(__name__)

# Load trained model and default feature values
artifact = joblib.load("model.joblib")
model = artifact["model"]
columns = artifact["columns"]
defaults = artifact["defaults"]


@app.get("/")
def home():
    return jsonify({
        "message": "House Price Prediction API",
        "target": "SalePrice",
        "endpoint": "POST /predict"
    })


@app.get("/health")
def health():
    return jsonify({"status": "healthy"}), 200


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Send a JSON object"
        }), 400

    # Ignore the Kaggle Id column; reject unknown features
    allowed = set(columns) | {"Id"}
    unknown = set(data) - allowed

    if unknown:
        return jsonify({
            "error": "Unknown feature(s)",
            "features": sorted(unknown)
        }), 400

    # Start with sensible defaults for all model features
    row = defaults.copy()

    for key, value in data.items():
        if key == "Id":
            continue

        if value is None:
            return jsonify({
                "error": f"Value for {key} cannot be null"
            }), 400

        if isinstance(value, float) and not math.isfinite(value):
            return jsonify({
                "error": f"Value for {key} must be finite"
            }), 400

        row[key] = value

    try:
        features = pd.DataFrame(
            [[row[col] for col in columns]],
            columns=columns
        )
        price = float(model.predict(features)[0])

        return jsonify({
            "predicted_sale_price": round(price, 2),
            "currency": "USD",
            "model": "RandomForestRegressor"
        }), 200

    except (ValueError, TypeError) as exc:
        return jsonify({
            "error": "Invalid feature value",
            "details": str(exc)
        }), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
