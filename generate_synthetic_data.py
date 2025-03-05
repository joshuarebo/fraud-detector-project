"""
Synthetic Fraud Data Generator

This script generates a dataset for fraud detection, simulating real-world banking and financial transactions.
It applies logical fraud detection conditions to flag potential fraudulent cases.

Author: Joshua Rebo
Date: March 2025
"""

import os
import pandas as pd
import numpy as np
from faker import Faker

# -------------------------- CONFIGURATION -------------------------- #
# Define output directory and file
DATA_DIR = "C:/Users/Hp/Fraud Detector/data/"
OUTPUT_FILE = os.path.join(DATA_DIR, "synthetic_fraud_data.csv")

# Set a random seed for reproducibility
np.random.seed(42)

# Initialize Faker instance
fake = Faker()

# Define number of samples
NUM_SAMPLES = 10_000


# -------------------------- HELPER FUNCTIONS -------------------------- #
def ensure_directory_exists(directory: str):
    """
    Ensures that the specified directory exists. Creates it if not present.

    :param directory: The directory path to check/create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"📁 Created directory: {directory}")
    else:
        print(f"✅ Directory already exists: {directory}")


def generate_synthetic_data(num_samples: int) -> pd.DataFrame:
    """
    Generates synthetic fraud detection data with meaningful patterns.

    :param num_samples: Number of records to generate.
    :return: Pandas DataFrame with synthetic fraud data.
    """

    print("\n🚀 Generating synthetic fraud data...\n")

    # Generate synthetic applicant data
    data = {
        "applicant_id": [fake.uuid4() for _ in range(num_samples)],
        "name": [fake.name() for _ in range(num_samples)],
        "age": np.random.randint(18, 70, num_samples),
        "income": np.random.randint(15_000, 200_000, num_samples),
        "loan_amount_requested": np.random.randint(1_000, 50_000, num_samples),
        "employment_status": np.random.choice(["Employed", "Self-Employed", "Unemployed"], num_samples),
        "credit_score": np.random.randint(300, 850, num_samples),
        "previous_loans_defaulted": np.random.randint(0, 5, num_samples),
        "country": [fake.country() for _ in range(num_samples)],
        "suspicious_activity": np.random.choice([0, 1], num_samples, p=[0.9, 0.1]),  # 10% cases are suspicious
    }

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Define fraud detection rules
    df["fraud_label"] = np.where(
        ((df["credit_score"] < 500) & (df["previous_loans_defaulted"] > 1) & 
         (df["loan_amount_requested"] > df["income"] * 0.5)) | 
        (df["suspicious_activity"] == 1), 1, 0
    )

    return df


def save_synthetic_data(df: pd.DataFrame, file_path: str):
    """
    Saves the generated data to a CSV file.

    :param df: Pandas DataFrame containing synthetic data.
    :param file_path: Path to save the CSV file.
    """
    df.to_csv(file_path, index=False)
    print(f"✅ Synthetic fraud data successfully saved at: {file_path}")


# -------------------------- SCRIPT EXECUTION -------------------------- #
if __name__ == "__main__":
    ensure_directory_exists(DATA_DIR)  # Ensure 'data/' directory exists
    df_synthetic = generate_synthetic_data(NUM_SAMPLES)  # Generate synthetic data
    save_synthetic_data(df_synthetic, OUTPUT_FILE)  # Save data to CSV

    print("\n🎯 Data generation complete! Ready for model training.")
