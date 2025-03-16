# src/train.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tpot import TPOTRegressor
from autokeras import StructuredDataRegressor
import joblib
import pandas as pd
import logging

# Configure logging
logging.basicConfig(filename='../logs/pipeline.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load preprocessed data
X = np.load('../data/X_clean.npy')
y = np.load('../data/y_clean.npy')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a DataFrame to store results
results = pd.DataFrame(columns=['Model', 'MAE', 'RMSE', 'R²'])

# Function to evaluate and store results
def evaluate_model(model, name, X_test, y_test):
    y_pred = model.predict(X_test).flatten()
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results.loc[len(results)] = [name, mae, rmse, r2]
    logging.info(f"{name} - Test MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

# 1. XGBoost
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
evaluate_model(xgb_model, 'XGBoost', X_test, y_test)
joblib.dump(xgb_model, '../models/xgboost_model.pkl')

# 2. LightGBM
lgbm_model = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
lgbm_model.fit(X_train, y_train)
evaluate_model(lgbm_model, 'LightGBM', X_test, y_test)
joblib.dump(lgbm_model, '../models/lightgbm_model.pkl')

# 3. CNN
X_cnn = X.reshape(X.shape[0], X.shape[1], 1)
X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(X_cnn, y, test_size=0.2, random_state=42)

cnn_model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_cnn.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(128, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1)
])
cnn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
cnn_model.fit(X_train_cnn, y_train_cnn, validation_split=0.2, epochs=50, batch_size=32, callbacks=[EarlyStopping(patience=10)], verbose=0)
evaluate_model(cnn_model, 'CNN', X_test_cnn, y_test_cnn)
cnn_model.save('../models/cnn_model.h5')

# 4. TPOT
tpot = TPOTRegressor(generations=5, population_size=20, random_state=42, verbosity=2)
tpot.fit(X_train, y_train)
evaluate_model(tpot, 'TPOT', X_test, y_test)
tpot.export('../models/tpot_model.py')

# 5. AutoKeras
autokeras_model = StructuredDataRegressor(max_trials=10, overwrite=True)
autokeras_model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=0)
evaluate_model(autokeras_model, 'AutoKeras', X_test, y_test)
autokeras_model.export_model().save('../models/autokeras_model.h5')

# Save results to a CSV file
results.to_csv('../models/model_comparison.csv', index=False)
logging.info("Model comparison saved to '../models/model_comparison.csv'")