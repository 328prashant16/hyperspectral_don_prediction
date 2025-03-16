# src/preprocessing.py

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib
import logging

# Configure logging
logging.basicConfig(filename='../logs/pipeline.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filepath):
    """Load hyperspectral data from a CSV file."""
    df = pd.read_csv(filepath)
    logging.info(f"Data loaded successfully with shape: {df.shape}")
    return df

def preprocess_data(df):
    """Preprocess the data: handle missing values, normalize, and detect anomalies."""
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X = df.iloc[:, 1:-1]  # Features
    y = df['vomitoxin_ppb']  # Target
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
    joblib.dump(scaler, '../models/scaler.pkl')
    logging.info("Scaler saved to '../models/scaler.pkl'")

    return X_clean, y_clean

# Example usage
if __name__ == '__main__':
    df = load_data('../data/hyperspectral_corn_data.csv')
    X_clean, y_clean = preprocess_data(df)
    np.save('../data/X_clean.npy', X_clean)
    np.save('../data/y_clean.npy', y_clean)
    logging.info("Preprocessed data saved to '../data/X_clean.npy' and '../data/y_clean.npy'")