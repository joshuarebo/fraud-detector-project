"""
Data Preprocessing & Exploratory Data Analysis (EDA)

This script loads the synthetic fraud dataset, performs data cleaning, handles missing values,
encodes categorical variables, applies feature scaling, detects outliers, and splits the data
into training, validation, and test sets.

Author: Joshua Rebo
Date: March 2025
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# -------------------------- CONFIGURATION -------------------------- #
# Define file paths
DATA_DIR = "C:/Users/Hp/Fraud Detector/data/"
INPUT_FILE = os.path.join(DATA_DIR, "synthetic_fraud_data.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, "processed/")
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Ensure processed data folder exists

# Define output file paths
TRAIN_FILE = os.path.join(OUTPUT_DIR, "train.csv")
VALID_FILE = os.path.join(OUTPUT_DIR, "valid.csv")
TEST_FILE = os.path.join(OUTPUT_DIR, "test.csv")


# -------------------------- LOAD DATA -------------------------- #
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.

    :param file_path: Path to the dataset file.
    :return: Pandas DataFrame
    """
    print("\n📂 Loading dataset...")
    df = pd.read_csv(file_path)
    print(f"✅ Dataset loaded successfully! Shape: {df.shape}\n")
    return df


# -------------------------- EXPLORATORY DATA ANALYSIS (EDA) -------------------------- #
def perform_eda(df: pd.DataFrame):
    """
    Perform basic exploratory data analysis.

    :param df: Pandas DataFrame
    """
    print("\n📊 Performing Exploratory Data Analysis (EDA)...\n")

    # Display dataset info
    print("🔹 Dataset Overview:\n")
    print(df.info())

    # Display summary statistics
    print("\n🔹 Summary Statistics:\n")
    print(df.describe())

    # Plot fraud distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x="fraud_label", data=df, palette="coolwarm")
    plt.title("Fraud Label Distribution")
    plt.savefig(os.path.join(OUTPUT_DIR, "fraud_distribution.png"))
    print("✅ Fraud distribution plot saved!\n")

    # Check for missing values
    missing_values = df.isnull().sum()
    print("\n🔹 Missing Values:\n", missing_values[missing_values > 0])

    return df


# -------------------------- DATA CLEANING -------------------------- #
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    :param df: Pandas DataFrame
    :return: Cleaned DataFrame
    """
    print("\n🛠 Handling missing values...")

    # Fill missing categorical values with mode
    categorical_columns = df.select_dtypes(include=["object"]).columns
    for col in categorical_columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Fill missing numerical values with median
    numerical_columns = df.select_dtypes(include=["number"]).columns
    for col in numerical_columns:
        df[col].fillna(df[col].median(), inplace=True)

    print("✅ Missing values handled successfully!\n")
    return df


# -------------------------- FEATURE ENGINEERING -------------------------- #
def encode_categorical_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical variables using Label Encoding.

    :param df: Pandas DataFrame
    :return: DataFrame with encoded categorical variables
    """
    print("\n🔠 Encoding categorical variables...")
    label_encoder = LabelEncoder()

    categorical_columns = ["employment_status", "country"]
    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col])

    print("✅ Categorical encoding completed!\n")
    return df


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale numerical features using StandardScaler.

    :param df: Pandas DataFrame
    :return: Scaled DataFrame
    """
    print("\n📏 Scaling numerical features...")
    scaler = StandardScaler()

    numerical_columns = ["age", "income", "loan_amount_requested", "credit_score", "previous_loans_defaulted"]
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    print("✅ Feature scaling applied!\n")
    return df


# -------------------------- OUTLIER DETECTION -------------------------- #
def detect_outliers(df: pd.DataFrame):
    """
    Detect outliers using interquartile range (IQR) method.

    :param df: Pandas DataFrame
    """
    print("\n🔍 Detecting outliers...")

    numerical_columns = ["income", "loan_amount_requested", "credit_score"]
    for col in numerical_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"📌 {col}: {len(outliers)} outliers detected.")

    print("✅ Outlier detection completed!\n")


# -------------------------- TRAIN-VALIDATION-TEST SPLIT -------------------------- #
def split_data(df: pd.DataFrame):
    """
    Split data into training, validation, and test sets.

    :param df: Pandas DataFrame
    """
    print("\n📂 Splitting dataset...")

    X = df.drop(columns=["fraud_label"])
    y = df["fraud_label"]

    # Split: 70% train, 15% validation, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Save to CSV
    pd.concat([X_train, y_train], axis=1).to_csv(TRAIN_FILE, index=False)
    pd.concat([X_valid, y_valid], axis=1).to_csv(VALID_FILE, index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(TEST_FILE, index=False)

    print("✅ Data successfully split and saved!\n")


# -------------------------- SCRIPT EXECUTION -------------------------- #
if __name__ == "__main__":
    df = load_data(INPUT_FILE)
    df = perform_eda(df)
    df = handle_missing_values(df)
    df = encode_categorical_variables(df)
    df = scale_features(df)
    detect_outliers(df)
    split_data(df)

    print("\n🎯 Data preprocessing complete! Ready for model training.")
