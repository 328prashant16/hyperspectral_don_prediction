# src/evaluate.py

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import shap
from tensorflow.keras.models import load_model
import joblib
import logging

# Configure logging
logging.basicConfig(filename='../logs/pipeline.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the best model (e.g., CNN)
model = load_model('../models/cnn_model.h5')
scaler = joblib.load('../models/scaler.pkl')

# Load test data
X_test = np.load('../data/X_clean.npy')
y_test = np.load('../data/y_clean.npy')

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
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title("Actual vs Predicted DON Concentration")
plt.xlabel("Actual DON (ppb)")
plt.ylabel("Predicted DON (ppb)")
plt.savefig('../images/actual_vs_predicted.png')
plt.show()

# SHAP analysis
explainer = shap.DeepExplainer(model, X_test_cnn[:100])
shap_values = explainer.shap_values(X_test_cnn[:100])
shap.summary_plot(shap_values[0], X_test_cnn[:100], feature_names=[f"Band_{i}" for i in range(X_test.shape[1])])
plt.savefig('../images/shap_summary.png')
plt.show()