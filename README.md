# Fraud Detection System

This project detects fraudulent transactions using machine learning.

## Setup Instructions
1. Clone the repository:
git clone https://github.com/joshuarebo/fraud-detector-project.git cd fraud-detector-project

markdown
Copy
Edit
2. Install dependencies:
pip install -r requirements.txt

markdown
Copy
Edit
3. Train the model:
python train_model.py

markdown
Copy
Edit
4. Run the API:
python fraud_api.py

css
Copy
Edit
5. Make a prediction:
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{"features": [35, 75000, 12000]}"

markdown
Copy
Edit

## MLOps Features
- **MLflow:** Experiment tracking.
- **GitHub Actions:** Automates model training & API deployment.
- **API Logging:** Logs every request & response.

## Author
- **Joshua Rebo** 🚀