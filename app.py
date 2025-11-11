import os
import gdown
from tensorflow.keras.models import load_model

MODEL_PATH = "cancer_detection_model.h5"

# Download model file if it doesn't exist
if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?export=download&id=1ElUbH5YK2henRCD6yy14AN6J2w8RfHHSV"  # 👈 Paste your link here
    gdown.download(url, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH)
from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Don't load model yet
model = None

# Function to load model only when needed
def get_model():
    global model
    if model is None:
        model = load_model('cancer_detection_model.h5')  # loads once
    return model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get model (load it if not already loaded)
        model = get_model()

        # Get values from HTML form
        data = [float(x) for x in request.form.values()]
        final_input = np.array(data).reshape(1, -1)

        # Make prediction
        prediction = model.predict(final_input)
        result = "Cancer (Positive)" if prediction[0][0] > 0.5 else "No Cancer (Negative)"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True) 