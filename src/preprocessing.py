import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib
import logging

# Define absolute paths
BASE_DIR = "E:\\hyperspectral_don_prediction"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Ensure required directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Configure logging
LOG_FILE = os.path.join(LOGS_DIR, "pipeline.log")
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

def load_data(filepath):
    """Load hyperspectral data from a CSV file."""
    df = pd.read_csv(filepath)
    logging.info(f"Data loaded successfully with shape: {df.shape}")
    return df

def preprocess_data(df):
    """Preprocess the data: handle missing values, normalize, and detect anomalies."""
    # Handle missing values
    imputer = SimpleImputer(strategy="median")
    X = df.iloc[:, 1:-1]  # Features
    y = df["vomitoxin_ppb"]  # Target
    X_imputed = imputer.fit_transform(X)

    # Normalize features
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X_imputed)

    # Detect anomalies
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    outliers = iso_forest.fit_predict(X_normalized)
    X_clean = X_normalized[outliers == 1]
    y_clean = y[outliers == 1]

    # Save scaler for later use
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    logging.info(f"Scaler saved to {scaler_path}")

    return X_clean, y_clean

# Example usage
if __name__ == "__main__":
    data_path = os.path.join(DATA_DIR, "MLE-Assignment.csv")  # Updated dataset path
    df = load_data(data_path)

    X_clean, y_clean = preprocess_data(df)

    np.save(os.path.join(DATA_DIR, "X_clean.npy"), X_clean)
    np.save(os.path.join(DATA_DIR, "y_clean.npy"), y_clean)
    logging.info(f"Preprocessed data saved to {DATA_DIR}/X_clean.npy and {DATA_DIR}/y_clean.npy")
