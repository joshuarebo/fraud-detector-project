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
import logging

# -------------------------- CONFIGURATION -------------------------- #
# Define output directory and file
DATA_DIR = "C:/Users/Hp/Fraud Detector/data/"
OUTPUT_FILE = os.path.join(DATA_DIR, "synthetic_fraud_data.csv")

# Set a random seed for reproducibility
np.random.seed(42)

# Initialize Faker instance
fake = Faker()

# Define number of months for time-series simulation
MONTHS = 12
TRANSACTIONS_PER_MONTH = 1000  # Simulating 1000 transactions per month

# -------------------------- LOGGING SETUP -------------------------- #
LOG_FILE = os.path.join(DATA_DIR, "data_generation.log")
logging.basicConfig(
    filename=LOG_FILE, 
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# -------------------------- HELPER FUNCTIONS -------------------------- #
def ensure_directory_exists(directory: str):
    """
    Ensures that the specified directory exists. Creates it if not present.

    :param directory: The directory path to check/create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"📁 Created directory: {directory}")
    else:
        logging.info(f"✅ Directory already exists: {directory}")


def generate_synthetic_data(months: int, transactions_per_month: int) -> pd.DataFrame:
    """
    Generates synthetic fraud detection data with meaningful patterns.

    :param months: Number of months to simulate.
    :param transactions_per_month: Number of transactions per month.
    :return: Pandas DataFrame with synthetic fraud data.
    """

    logging.info("\n🚀 Generating synthetic fraud data...\n")

    data = []
    for month in range(1, months + 1):
        for _ in range(transactions_per_month):
            age = np.random.randint(18, 70)
            income = np.random.randint(15_000, 200_000)
            loan_amount_requested = np.random.randint(1_000, 50_000)
            employment_status = np.random.choice(["Employed", "Self-Employed", "Unemployed"])
            credit_score = np.random.randint(300, 850)
            previous_loans_defaulted = np.random.randint(0, 5)
            transaction_value = np.random.randint(100, 50_000)
            country = fake.country()
            suspicious_activity = np.random.choice([0, 1], p=[0.9, 0.1])  # 10% suspicious activity
            fraud = np.random.choice([0, 1], p=[0.97, 0.03])  # 3% fraud rate

            # Apply fraud detection rules
            fraud_label = 1 if ((credit_score < 500 and previous_loans_defaulted > 1 and 
                                loan_amount_requested > income * 0.5) or suspicious_activity == 1) else fraud

            data.append([
                month, age, income, loan_amount_requested, employment_status, credit_score,
                previous_loans_defaulted, transaction_value, country, suspicious_activity, fraud_label
            ])

    df = pd.DataFrame(data, columns=[
        "month", "age", "income", "loan_amount_requested", "employment_status",
        "credit_score", "previous_loans_defaulted", "transaction_value", "country",
        "suspicious_activity", "fraud_label"
    ])

    logging.info(f"✅ Generated {len(df)} transactions across {months} months.")
    return df


def save_synthetic_data(df: pd.DataFrame, file_path: str):
    """
    Saves the generated data to a CSV file.

    :param df: Pandas DataFrame containing synthetic data.
    :param file_path: Path to save the CSV file.
    """
    df.to_csv(file_path, index=False)
    logging.info(f"✅ Synthetic fraud data successfully saved at: {file_path}")
    print(f"✅ Synthetic fraud data saved at: {file_path}")


# -------------------------- SCRIPT EXECUTION -------------------------- #
if __name__ == "__main__":
    ensure_directory_exists(DATA_DIR)  # Ensure 'data/' directory exists
    df_synthetic = generate_synthetic_data(MONTHS, TRANSACTIONS_PER_MONTH)  # Generate synthetic data
    save_synthetic_data(df_synthetic, OUTPUT_FILE)  # Save data to CSV

    print("\n🎯 Data generation complete! Ready for model training.")
    logging.info("\n🎯 Data generation complete! Ready for model training.")