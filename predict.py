import tensorflow as tf
import numpy as np
import cv2
import sys
import os

# ======================
# LOAD MODEL
# ======================
model = tf.keras.models.load_model("waste_model.h5")

# ======================
# CLASS NAMES (must match training order)
# ======================
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# ======================
# IMAGE SIZE (MUST MATCH TRAINING)
# ======================
img_size = (128, 128)

# ======================
# CHECK INPUT
# ======================
if len(sys.argv) != 2:
    print("Usage: python predict.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

if not os.path.exists(image_path):
    print("❌ Image not found:", image_path)
    sys.exit(1)

# ======================
# LOAD IMAGE
# ======================
img = cv2.imread(image_path)

if img is None:
    print("❌ Failed to read image")
    sys.exit(1)

# resize
img = cv2.resize(img, img_size)

# normalize
img = img / 255.0

# expand dims (batch)
img = np.expand_dims(img, axis=0)

# ======================
# PREDICT
# ======================
pred = model.predict(img)

predicted_class = class_names[np.argmax(pred)]
confidence = np.max(pred)

print("Prediction:", predicted_class)
print("Confidence:", confidence)
