import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load your saved model
model = load_model("mask_detector_vgg16.h5")

# Class labels (order must match training)
labels = ["with_mask", "without_mask"]

# -----------------------------
# Function to preprocess image
# -----------------------------
def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError("Image not found or path is incorrect.")

    img_resized = cv2.resize(img, (224, 224))   # same as training size
    img_normalized = img_resized / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)
    return img, img_expanded

# -----------------------------
# Predict function
# -----------------------------
def predict_mask(image_path):
    original_img, processed_img = preprocess_image(image_path)

    prediction = model.predict(processed_img)
    class_index = np.argmax(prediction)  # get highest prob class
    label = labels[class_index]

    # Display result on image
    display_img = original_img.copy()
    cv2.putText(display_img, label, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Prediction", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Prediction:", label)

# -----------------------------
# Test your image
# -----------------------------
image_path = "test_image.jpg"   # change this path
predict_mask(image_path)
