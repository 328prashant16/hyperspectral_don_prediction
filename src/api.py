from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Optional  # To handle Optional types for MSE, MAE, R²

# Ensure logs directory exists
LOG_DIR = r"E:\hyperspectral_don_prediction\logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
LOG_FILE = os.path.join(LOG_DIR, "pipeline.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize FastAPI app
app = FastAPI()

# Define absolute paths for model and scaler
MODEL_PATH = r"E:\hyperspectral_don_prediction\models\cnn_model.h5"
SCALER_PATH = r"E:\hyperspectral_don_prediction\models\scaler.pkl"

# Load the CNN model and scaler
try:
    logging.info("Loading model and scaler...")
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logging.info("Model and scaler loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model or scaler: {e}")
    raise RuntimeError(f"Error loading model or scaler: {e}")

# Define request body schema for CSV file upload
class CSVResponse(BaseModel):
    predictions: list[float]
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "API is running"}

# Prediction endpoint for CSV file upload
@app.post("/predict_csv", response_model=CSVResponse)
async def predict_csv(file: UploadFile = File(...)):
    try:
        # Read the CSV file
        logging.info(f"Reading CSV file: {file.filename}")
        df = pd.read_csv(file.file)
        logging.info(f"CSV file loaded successfully with shape: {df.shape}")

        # Check if the required column exists
        if "vomitoxin_ppb" not in df.columns:
            raise HTTPException(status_code=400, detail="CSV file must contain a 'vomitoxin_ppb' column.")
        
        # Remove 'hsi_id' column since it's not required for prediction
        if 'hsi_id' in df.columns:
            df = df.drop(columns=["hsi_id"])

        # Separate features and target
        X = df.drop(columns=["vomitoxin_ppb"]).values
        y = df["vomitoxin_ppb"].values

        # Validate input shape
        if X.shape[1] != scaler.n_features_in_:
            raise HTTPException(status_code=400, detail=f"Expected {scaler.n_features_in_} features, got {X.shape[1]}.")

        # Preprocess the data
        X_scaled = scaler.transform(X)
        X_cnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

        # Make predictions
        predictions = model.predict(X_cnn).flatten().tolist()
        logging.info(f"Predictions made successfully: {predictions}")

        # Calculate evaluation metrics
        mse = mean_squared_error(y, predictions)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)
        logging.info(f"Evaluation metrics - MSE: {mse}, MAE: {mae}, R²: {r2}")

        return {
            "predictions": predictions,
            "mse": mse,
            "mae": mae,
            "r2": r2
        }
    
    except HTTPException as e:
        logging.error(f"Client error: {e.detail}")
        raise e
    except Exception as e:
        logging.error(f"Internal server error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    logging.info("Starting FastAPI server on port 5000...")
    uvicorn.run(app, host="0.0.0.0", port=5000)
