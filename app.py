from flask import Flask, request, jsonify
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Download model from Hugging Face repo
model_path = hf_hub_download(
    repo_id="Dee-pak-13/vgg_mode",  # replace with your repo name
    filename="model_vgg.h5"
)
print(f"Model downloaded to: {model_path}")

model = load_model(model_path)
print("Model loaded successfully.")

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

    