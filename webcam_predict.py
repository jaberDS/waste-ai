import cv2
import numpy as np
import tensorflow as tf

# ======================
# LOAD MODEL
# ======================
model = tf.keras.models.load_model("waste_model.h5")

# class names (same order as training)
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# image size used in training
img_size = (128, 128)

# ======================
# OPEN WEBCAM
# ======================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

print("🎥 Webcam started... Press Q to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to grab frame")
        break

    # ======================
    # PREPROCESS FRAME
    # ======================
    img = cv2.resize(frame, img_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # ======================
    # PREDICT
    # ======================
    pred = model.predict(img, verbose=0)

    class_id = np.argmax(pred)
    label = class_names[class_id]
    confidence = np.max(pred)

    # ======================
    # DISPLAY TEXT
    # ======================
    text = f"{label} ({confidence:.2f})"

    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    # show webcam
    cv2.imshow("Waste Classifier", frame)

    # press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release
cap.release()
cv2.destroyAllWindows()
