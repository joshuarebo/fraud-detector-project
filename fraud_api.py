from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Define paths for model and scaler
MODEL_PATH = "models/fraud_model.pkl"
SCALER_PATH = "models/scaler.pkl"

# Load trained model & scaler
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError:
    raise RuntimeError("🚨 Model or scaler file not found. Train the model first.")

@app.route("/", methods=["GET"])
def home():
    """Root endpoint showing API information"""
    return jsonify({
        "message": "Welcome to the Fraud Detection API 🚀",
        "endpoints": {
            "health_check": "/health",
            "predict": "/predict"
        }
    }), 200

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "API is running"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    """Predict fraud based on input features"""
    try:
        data = request.get_json()

        # Validate request body
        if not data or "features" not in data:
            return jsonify({"error": "Missing 'features' in request body"}), 400

        # Ensure the features are a list
        if not isinstance(data["features"], list):
            return jsonify({"error": "Features must be a list"}), 400

        # Convert input to NumPy array
        features = np.array(data["features"]).reshape(1, -1)

        # Scale input
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]

        return jsonify({
            "fraud_prediction": int(prediction),
            "fraud_probability": round(probability, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
