import pandas as pd
import numpy as np
import joblib
import os
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Enable MLflow tracking
mlflow.set_experiment("fraud_detection_experiment")

# Function to load and preprocess the dataset
def load_data():
    df = pd.read_csv("data/processed/train.csv")

    # Drop non-numeric columns (e.g., transaction IDs or user IDs)
    non_numeric_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    df = df.drop(columns=non_numeric_cols)

    # Encode categorical variables if present
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Define feature columns and target
    X = df.drop(columns=["fraud_label"])
    y = df["fraud_label"]

    # Split data into training & validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    return X_train_scaled, X_val_scaled, y_train, y_val, scaler

# Function to train and log the model using MLflow
def train_and_log_model():
    X_train, X_test, y_train, y_test, scaler = load_data()
    
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        predictions_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions),
            "Recall": recall_score(y_test, predictions),
            "F1-score": f1_score(y_test, predictions),
            "AUC-ROC": roc_auc_score(y_test, predictions_proba)
        }

        # Log metrics to MLflow
        for key, value in metrics.items():
            mlflow.log_metric(key.lower(), value)

        # Log model to MLflow
        mlflow.sklearn.log_model(model, "random_forest_model")

        # Save model & scaler locally
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/fraud_model.pkl")
        joblib.dump(scaler, "models/scaler.pkl")

        # Print evaluation metrics
        print("✅ Model Training Complete. Evaluation Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    train_and_log_model()
