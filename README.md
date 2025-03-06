Fraud Detection System

🚀 Overview

The Fraud Detection System is a machine learning-powered solution designed to detect fraudulent transactions in real-time. It incorporates MLOps best practices, automated deployment pipelines, and cloud scalability to ensure efficiency and reliability.

✨ Features

Machine Learning Model: Predicts fraudulent transactions with high accuracy.

RESTful API: Exposes a real-time fraud detection endpoint.

MLOps Pipeline: Automates training, tracking, and deployment using MLflow.

CI/CD Automation: GitHub Actions and Google Cloud Build ensure continuous integration and deployment.

Logging & Monitoring: Tracks API requests, model performance, and system health.

Cloud Deployment: Hosted on Google Cloud Run for seamless scalability.

🛠️ Setup & Installation

1️⃣ Clone the Repository

git clone https://github.com/joshuarebo/fraud-detector.git
cd fraud-detector

2️⃣ Set Up a Virtual Environment & Install Dependencies

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt

3️⃣ Train the Fraud Detection Model

python train_model.py

This script trains the model and saves it in the models/ directory.

It also applies feature scaling using a fitted scaler.

4️⃣ Run the API Locally

python fraud_api.py

The API will be available at: http://127.0.0.1:8080/predict

Note: The API only accepts POST requests. Directly clicking on the URL in a browser will result in a 405 Method Not Allowed error. Use curl or a tool like Postman to test it.

5️⃣ Make a Prediction (Example Request)

curl -X POST http://127.0.0.1:8080/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [35, 75000, 12000, 50, 600, 1, 4, 5]}'

Expected response:

{
  "fraud_prediction": 1,
  "fraud_probability": 0.92
}

☁️ Cloud Deployment (Google Cloud Run)

Deploying to Google Cloud Run

Build & Push Docker Image

gcloud builds submit --tag gcr.io/$PROJECT_ID/fraud-api:latest

Deploy to Cloud Run

gcloud run deploy fraud-api \
    --image gcr.io/$PROJECT_ID/fraud-api:latest \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --port 8080

Get the Deployed URL

gcloud run services describe fraud-api --platform managed --region us-central1 --format="value(status.url)"

Accessing the Cloud API

The API is now live at:

https://fraud-api-716102832289.us-central1.run.app

To make predictions on the deployed API, use:

curl -X POST "https://fraud-api-716102832289.us-central1.run.app/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [35, 75000, 12000, 50, 600, 1, 4, 5]}'

🔄 MLOps & CI/CD Integration

MLflow: Tracks experiments, model versions, and logs performance.

GitHub Actions & Cloud Build: Automates testing, training, and API deployment.

Logging: API requests and model performance are monitored.

Triggering the CI/CD Workflow

Simply push any changes to GitHub:

git add .
git commit -m "Updated model & API improvements"
git push origin main

This triggers Google Cloud Build, which:

Builds and pushes a new Docker image.

Deploys the updated API to Google Cloud Run.

Runs test_fraud_api.py to validate the deployment.

📈 Monitoring & Scaling

Model Monitoring: MLflow logs performance metrics.

Scalability: Google Cloud Run scales automatically based on traffic.

Containerization: Docker & Kubernetes-ready for future expansion.

👤 Author

Joshua Rebo – Applied Artificial Intelligence Student
Matriculation Number: 9213334
The International University of Applied Sciences

📧 Contact

Email: joshua.rebo@iu-study.org