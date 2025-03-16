import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import shap
import tensorflow as tf
load_model = tf.keras.models.load_model

import joblib
import logging
import os

# Define base directory
BASE_DIR = "E:/hyperspectral_don_prediction"

# Configure logging
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(filename=os.path.join(LOG_DIR, "pipeline.log"), level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Load the best model (e.g., CNN)
MODEL_PATH = os.path.join(BASE_DIR, "models", "cnn_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Load test data
DATA_DIR = os.path.join(BASE_DIR, "data")
X_test = np.load(os.path.join(DATA_DIR, "X_clean.npy"))
y_test = np.load(os.path.join(DATA_DIR, "y_clean.npy"))

# Reshape data for CNN
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Make predictions
y_pred = model.predict(X_test_cnn).flatten()

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

logging.info(f"Test MAE: {mae:.2f}")
logging.info(f"Test RMSE: {rmse:.2f}")
logging.info(f"Test R²: {r2:.2f}")

# Plot actual vs predicted
IMAGE_DIR = os.path.join(BASE_DIR, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title("Actual vs Predicted DON Concentration")
plt.xlabel("Actual DON (ppb)")
plt.ylabel("Predicted DON (ppb)")
plt.savefig(os.path.join(IMAGE_DIR, "actual_vs_predicted.png"))
plt.show()


# SHAP analysis
# Ensure feature names are defined
num_features = X_test.shape[1]
feature_names = [f"Band_{i}" for i in range(num_features)]

# Run SHAP analysis with proper background data
explainer = shap.DeepExplainer(model, X_test_cnn[:100])  # Ensure correct input shape
shap_values = explainer.shap_values(X_test_cnn[:100])  # Compute SHAP values

# Ensure shap_values is correctly shaped
if isinstance(shap_values, list):  # DeepExplainer returns a list for multi-output models
    shap_values = shap_values[0]

# Ensure reshaping matches input format
shap_values_reshaped = np.array(shap_values)  
X_test_cnn_reshaped = X_test_cnn[:100].reshape(100, -1)

# Generate SHAP summary plot
# Check shapes
print(f"shap_values_reshaped shape: {shap_values_reshaped.shape}")
print(f"X_test_cnn_reshaped shape: {X_test_cnn_reshaped.shape}")
print(f"feature_names length: {len(feature_names)}")

# Ensure correct reshaping
X_test_cnn_reshaped = X_test_cnn[:100].reshape(100, num_features)

# Ensure feature names match feature dimension
if X_test_cnn_reshaped.shape[1] != len(feature_names):
    feature_names = [f"Band_{i}" for i in range(X_test_cnn_reshaped.shape[1])]

# Generate SHAP summary plot
shap.summary_plot(shap_values_reshaped, X_test_cnn_reshaped, feature_names=feature_names)
plt.savefig(os.path.join(IMAGE_DIR, "shap_summary.png"))
plt.show()
