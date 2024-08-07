from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import logging

app = Flask(__name__)

# Setup logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'app.log')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file),
                              logging.StreamHandler()])

# Load the model
model_path = os.path.join('model', 'best_model.joblib')
model = joblib.load(model_path)
logging.info("Model loaded from %s", model_path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    logging.info("Prediction made for features: %s", features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)