Fraud Detection System

This project is a machine learning-powered fraud detection system that identifies fraudulent transactions with high accuracy. It includes automated MLOps workflows for seamless model training, deployment, and monitoring.

🚀 Features

Machine Learning Model: Trained to detect fraudulent transactions.

RESTful API: Serves real-time fraud predictions.

MLOps Pipeline: Integrated with MLflow for experiment tracking.

Automated CI/CD: GitHub Actions automates testing and deployment.

Logging & Monitoring: Requests and model performance are logged.

🛠️ Setup & Installation

1️⃣ Clone the Repository

git clone https://github.com/joshuarebo/fraud-detector-project.git
cd fraud-detector-project

2️⃣ Create a Virtual Environment & Install Dependencies

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt

3️⃣ Train the Model

python train_model.py

4️⃣ Run the API

python fraud_api.py

5️⃣ Make a Prediction (Example Request)

curl -X POST http://127.0.0.1:5001/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [35, 75000, 12000, 50, 600, 1, 4, 5]}'

MLOps Integration

MLflow: Tracks experiments, model versions, and logs performance.

GitHub Actions: Automates testing, model training, and API deployment.

Logging: API requests and responses are logged for analysis.

📈 Monitoring & Scaling

Model Monitoring: MLflow logs model performance metrics.

Scalability: Ready for cloud deployment using AWS/GCP/Azure.

Containerization: Can be deployed via Docker & Kubernetes.

Author:
Joshua Rebo – Applied Artificial Intelligence Student
Matriculation Number- 9213334
The International University of Applied Sciences