import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io

app = Flask(__name__)

MODEL_PATH = "waste_model.keras"
IMG_SIZE = (160, 160)
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

model = tf.keras.models.load_model(MODEL_PATH)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]

    results = [
        {"class": CLASS_NAMES[i], "confidence": round(float(predictions[i]) * 100, 1)}
        for i in range(len(CLASS_NAMES))
    ]
    results.sort(key=lambda x: x["confidence"], reverse=True)

    return jsonify({
        "prediction": results[0]["class"],
        "confidence": results[0]["confidence"],
        "all": results
    })

if __name__ == "__main__":
    app.run(debug=True)
