from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load your model (choose VGG or CNN)
MODEL_PATH = "model_vgg.h5"   # or "model_cnn.h5"
model = load_model(MODEL_PATH)

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    # Save temporarily
    filepath = "temp.jpg"
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=(224, 224))  # adjust if CNN was different
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)[0][0]
    result = "Crack" if pred > 0.5 else "No Crack"

    # Clean up
    os.remove(filepath)

    return jsonify({"prediction": result, "confidence": float(pred)})

if __name__ == "__main__":
    app.run(debug=True)

    