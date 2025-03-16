# src/config.py

# Hyperparameters for models
XGBOOST_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'random_state': 42
}

LIGHTGBM_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'random_state': 42
}

CNN_PARAMS = {
    'input_shape': (200, 1),  # Update based on your data
    'filters': [64, 128],
    'kernel_size': 3,
    'pool_size': 2,
    'dense_units': 64,
    'dropout_rate': 0.3
}

TPOT_PARAMS = {
    'generations': 5,
    'population_size': 20,
    'random_state': 42,
    'verbosity': 2
}

AUTOKERAS_PARAMS = {
    'max_trials': 10,
    'overwrite': True
}