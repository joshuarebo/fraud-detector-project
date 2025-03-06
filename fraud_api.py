import os
import logging
import joblib
import numpy as np
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure logging with JSON format for better observability
logging.basicConfig(
    filename="logs/api_requests.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# Define model paths
MODEL_PATH = os.path.abspath("models/fraud_model.pkl")
SCALER_PATH = os.path.abspath("models/scaler.pkl")

logging.info(f"🔎 Checking model at: {MODEL_PATH}")
logging.info(f"🔎 Checking scaler at: {SCALER_PATH}")

# Ensure model & scaler exist
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    logging.error("🚨 Model or scaler file not found. Train the model first.")
    raise RuntimeError("Model or scaler file not found. Train the model first.")

# Load trained model & scaler
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logging.info("✅ Model and scaler loaded successfully.")
except Exception as e:
    logging.exception("🚨 Error loading model or scaler.")
    raise RuntimeError(f"Error loading model or scaler: {e}")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles fraud prediction requests.
    Expects a JSON body: {"features": [list_of_values]}
    Returns a JSON response with fraud prediction and probability.
    """
    try:
        # Parse request JSON
        data = request.get_json()

        # Validate request body
        if not data or "features" not in data:
            logging.warning("Received invalid request: Missing 'features'.")
            return jsonify({"error": "Missing 'features' in request body"}), 400

        # Convert input to NumPy array & scale
        features = np.array(data["features"]).reshape(1, -1)
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = float(model.predict_proba(features_scaled)[0][1])  # Convert to float for JSON serialization

        response = {
            "fraud_prediction": int(prediction),
            "fraud_probability": round(probability, 4)
        }

        logging.info(f"📩 Request: {data} | 📤 Response: {response}")
        return jsonify(response), 200

    except Exception as e:
        logging.exception("🚨 Error processing request.")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    # Get port from environment variable (default to 8080)
    port = int(os.environ.get("PORT", 8080))

    # Run Flask app (remove debug=True for production)
    app.run(host="0.0.0.0", port=port)
