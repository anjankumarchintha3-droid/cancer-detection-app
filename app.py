import os
import gdown
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

# Model file name
MODEL_PATH = "cancer_detection_model.h5"

# ✅ Correct Google Drive direct download link
url = "https://drive.google.com/uc?export=download&id=1ELkbH5YK2henRCD6vjY4AN6v2wRiHHSX"

# ✅ Download model from Google Drive if it doesn’t exist locally
if not os.path.exists(MODEL_PATH):
    gdown.download(url, MODEL_PATH, quiet=False)

# ✅ Load the model
model = load_model(MODEL_PATH)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from HTML form
        data = [float(x) for x in request.form.values()]
        final_input = np.array(data).reshape(1, -1)

        # Predict using the model
        prediction = model.predict(final_input)
        result = "Cancer (Positive)" if prediction[0][0] > 0.5 else "No Cancer (Negative)"
        return render_template('index.html', prediction_text=result)
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)