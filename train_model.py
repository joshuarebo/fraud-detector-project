import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load the preprocessed dataset
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
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train a RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_val_scaled)
y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]

# Model evaluation
metrics = {
    "Accuracy": accuracy_score(y_val, y_pred),
    "Precision": precision_score(y_val, y_pred),
    "Recall": recall_score(y_val, y_pred),
    "F1-score": f1_score(y_val, y_pred),
    "AUC-ROC": roc_auc_score(y_val, y_pred_proba)
}

# Save trained model & scaler
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/fraud_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# Print metrics
print("✅ Model Training Complete. Evaluation Metrics:")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")
