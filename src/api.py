# src/api.py

from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Load the best model (e.g., CNN)
model = load_model('../models/cnn_model.h5')
scaler = joblib.load('../models/scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    spectral_data = np.array(data['spectral_values'])
    spectral_data_scaled = scaler.transform(spectral_data)
    spectral_data_cnn = spectral_data_scaled.reshape(spectral_data_scaled.shape[0], spectral_data_scaled.shape[1], 1)
    predictions = model.predict(spectral_data_cnn).flatten().tolist()
    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)