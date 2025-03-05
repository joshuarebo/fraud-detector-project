import logging
import joblib
import numpy as np
import os
import mlflow.pyfunc
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    filename="logs/api_requests.log", 
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define paths for model and scaler
MODEL_PATH = "models/fraud_model.pkl"
SCALER_PATH = "models/scaler.pkl"

# Load trained model & scaler
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logging.info("✅ Model and scaler loaded successfully.")
except FileNotFoundError:
    logging.error("🚨 Model or scaler file not found. Train the model first.")
    raise RuntimeError("Model or scaler file not found. Train the model first.")

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
            logging.warning("Received invalid request: Missing 'features'.")
            return jsonify({"error": "Missing 'features' in request body"}), 400

        # Ensure the features are a list
        if not isinstance(data["features"], list):
            logging.warning("Received invalid request: Features must be a list.")
            return jsonify({"error": "Features must be a list"}), 400

        # Convert input to NumPy array
        features = np.array(data["features"]).reshape(1, -1)

        # Scale input
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]

        response = {
            "fraud_prediction": int(prediction),
            "fraud_probability": round(probability, 4)
        }

        # Log request and response
        logging.info(f"Request: {data}, Response: {response}")

        return jsonify(response), 200

    except Exception as e:
        logging.error(f"🚨 Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)  # Ensure log directory exists
    app.run(host="0.0.0.0", port=5000, debug=True)
