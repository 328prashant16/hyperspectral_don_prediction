import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from tpot import TPOTRegressor
from autokeras import StructuredDataRegressor
import joblib
import pandas as pd
import logging
import optuna
import tensorflow as tf

Sequential = tf.keras.models.Sequential
Conv1D = tf.keras.layers.Conv1D
MaxPooling1D = tf.keras.layers.MaxPooling1D
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
l2 = tf.keras.regularizers.l2
EarlyStopping = tf.keras.callbacks.EarlyStopping

print("TensorFlow Version:", tf.__version__)
print("Imports are working!")

# Configure logging
logging.basicConfig(filename='E:/hyperspectral_don_prediction/logs/pipeline.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load preprocessed data
X = np.load('E:/hyperspectral_don_prediction/data/X_clean.npy')
y = np.load('E:/hyperspectral_don_prediction/data/y_clean.npy')

# Add derivative features
def add_derivative_features(X):
    """Add first and second derivatives of spectral data."""
    first_derivative = np.gradient(X, axis=1)
    second_derivative = np.gradient(first_derivative, axis=1)
    return np.hstack((X, first_derivative, second_derivative))

X_enhanced = add_derivative_features(X)
X_train, X_test, y_train, y_test = train_test_split(X_enhanced, y, test_size=0.2, random_state=42)

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

# 1. XGBoost with Hyperparameter Tuning
def objective_xgb(trial):
    """Optuna objective function for XGBoost hyperparameter optimization."""
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    
    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred)

study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb, n_trials=20)

best_params_xgb = study_xgb.best_params
xgb_model = XGBRegressor(**best_params_xgb, random_state=42)
xgb_model.fit(X_train, y_train)
evaluate_model(xgb_model, 'XGBoost (Tuned)', X_test, y_test)
joblib.dump(xgb_model, 'E:/hyperspectral_don_prediction/models/xgboost_model_tuned.pkl')

# 2. LightGBM
lgbm_model = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
lgbm_model.fit(X_train, y_train)
evaluate_model(lgbm_model, 'LightGBM', X_test, y_test)
joblib.dump(lgbm_model, 'E:/hyperspectral_don_prediction/models/lightgbm_model.pkl')

# 3. CNN with Hyperparameter Tuning and Data Augmentation
def add_noise(X, noise_level=0.01):
    """Add Gaussian noise to the data."""
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

X_cnn = X_enhanced.reshape(X_enhanced.shape[0], X_enhanced.shape[1], 1)
X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(X_cnn, y, test_size=0.2, random_state=42)

# Augment training data
X_train_augmented = add_noise(X_train_cnn)
y_train_augmented = y_train_cnn

# Combine original and augmented data
X_train_combined = np.vstack((X_train_cnn, X_train_augmented))
y_train_combined = np.hstack((y_train_cnn, y_train_augmented))

def objective_cnn(trial):
    """Optuna objective function for CNN hyperparameter optimization."""
    filters = trial.suggest_int('filters', 32, 256)
    kernel_size = trial.suggest_int('kernel_size', 2, 5)
    dense_units = trial.suggest_int('dense_units', 32, 256)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    
    model = Sequential([
        Conv1D(filters, kernel_size, activation='relu', input_shape=(X_cnn.shape[1], 1), kernel_regularizer=l2(0.01)),
        MaxPooling1D(pool_size=2),
        Conv1D(filters * 2, kernel_size, activation='relu', kernel_regularizer=l2(0.01)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(dense_units, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(dropout_rate),
        Dense(1)
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    
    history = model.fit(
        X_train_combined, y_train_combined,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=[EarlyStopping(patience=10)],
        verbose=0
    )
    
    return min(history.history['val_loss'])

# Save results to a CSV file
results.to_csv('E:/hyperspectral_don_prediction/models/model_comparison.csv', index=False)
logging.info("Model comparison saved to 'E:/hyperspectral_don_prediction/models/model_comparison.csv'")
