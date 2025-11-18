import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.preprocessing import image

# ============================
# CONFIGURATION
# ============================

MODEL_DIR = "./models"

# Map each class to its corresponding model file
CLASS_MODEL_MAP = {
    "class1": "resnet50_group1_model.h5",
    "class2": "resnet50_group1_model.h5",
    "class3": "resnet50_group1_model.h5",
    "class4": "resnet50_group2_model.h5",
    "class5": "resnet50_group2_model.h5",
    "class6": "resnet50_group2_model.h5",
    "class7": "resnet50_group3_model.h5",
    "class8": "resnet50_group3_model.h5",
    "class9": "resnet50_group3_model.h5",
    "class10": "resnet50_group4_model.h5",
    "class11": "resnet50_group4_model.h5",
    "class12": "resnet50_group4_model.h5",
    "class13": "resnet50_group5_model.h5",
    "class14": "resnet50_group5_model.h5",
    "class15": "resnet50_group5_model.h5",
    "class16": "resnet50_group6_model.h5",
    "class17": "resnet50_group6_model.h5",
    "class18": "resnet50_group6_model.h5",
    "class19": "resnet50_group7_model.h5",
    "class20": "resnet50_group7_model.h5",
    "class21": "resnet50_group7_model.h5",
    "class22": "resnet50_group8_model.h5",
    "class23": "resnet50_group8_model.h5",
    "class24": "resnet50_group8_model.h5",
    "class25": "resnet50_group9_model.h5",
    "class26": "resnet50_group9_model.h5",
    "class27": "resnet50_group9_model.h5",
    "class28": "resnet50_group10_model.h5",
    "class29": "resnet50_group10_model.h5",
    "class30": "resnet50_group10_model.h5",
    "class31": "resnet50_group11_model.h5",
    "class32": "resnet50_group11_model.h5",
    "class33": "resnet50_group11_model.h5",
    "class34": "resnet50_group11_model.h5"
}

# ============================
# FUNCTIONS
# ============================

def load_model_for_class(class_name):
    if class_name not in CLASS_MODEL_MAP:
        raise ValueError(f"No model found for class '{class_name}'")

    model_path = os.path.join(MODEL_DIR, CLASS_MODEL_MAP[class_name])
    print(f"\n📦 Loading model for '{class_name}' → {model_path}")
    return tf.keras.models.load_model(model_path)


def prepare_image(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array


def evaluate_model(model, X_test, y_test):
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred)

    print(f"\n✅ Accuracy: {acc:.4f}")
    print("\n📊 Confusion Matrix:\n", cm)
    print("\n📋 Classification Report:\n", cr)


# ============================
# MAIN EXECUTION
# ============================

if __name__ == "__main__":
    image_path = input("C:\\Users\\Mahesh Porla\\Downloads\\foodproject\\validation_dataset\\Group_5\\cheesecake\\148078.jpg").strip()
    class_name = input("cheesecake").strip()

    model = load_model_for_class(class_name)
    img_array = prepare_image(image_path)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    print(f"\n🔍 Predicted Class Index: {predicted_class}")
    print(f"🎯 Confidence Score: {confidence*100:.2f}%")

    # Optional: Evaluate model on test data if available
    # evaluate_model(model, X_test, y_test)
