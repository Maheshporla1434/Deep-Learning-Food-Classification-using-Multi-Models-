# app.py
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import json
import os
import cv2

app = Flask(__name__)

MODELS_BASE = "models"
STATIC_TEMP = "static/temp.jpg"
FOOD_JSON_PATH = "food_json_files"
MODEL_JSON_PATH = "model_json_files"
IMG_SIZE = (256, 256)

ALL_CLASSES_GROUPS = {
    "Group_1": ["Baked Potato", "Crispy Chicken", "Donut"],
    "Group_2": ["Fries", "Hot Dog", "Sandwich"],
    "Group_3": ["apple_pie", "Taco", "Taquito"],
    "Group_4": ["burger", "butter_naan", "chai"],
    "Group_5": ["chapati", "cheesecake", "chicken_curry"],
    "Group_6": ["chole_bhature", "dal_makhani", "dhokla"],
    "Group_7": ["fried_rice", "ice_cream", "idli"],
    "Group_8": ["jalebi", "kaathi_rolls", "kadai_paneer"],
    "Group_9": ["kulfi", "masala_dosa", "momos"],
    "Group_10": ["omelette", "paani_puri", "pakode"],
    "Group_11": ["pav_bhaji", "pizza", "samosa", "sushi"]
}


def get_model_file_for_class(selected_class, model_group):
    group_name = None
    for g, cls_list in ALL_CLASSES_GROUPS.items():
        if selected_class in cls_list:
            group_name = g
            break

    if not group_name:
        raise ValueError(f"Class {selected_class} not found in groups.")

    group_num = int(group_name.split("_")[1])

    if model_group == "custom_models":
        return f"custom_model_group{group_num}.h5"
    if model_group == "vgg16_models":
        return f"vgg16_group{group_num}_model.h5"
    if model_group == "resnet50_models":
        return f"resnet50_group{group_num}_model.h5"

    raise ValueError("Invalid model group")


# Load big JSONs that contain model-level metrics
BIG_JSONS = {}
if os.path.exists(MODEL_JSON_PATH):
    for fname in os.listdir(MODEL_JSON_PATH):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(MODEL_JSON_PATH, fname), "r") as fh:
                    BIG_JSONS[fname] = json.load(fh)
            except Exception:
                BIG_JSONS[fname] = {}
else:
    BIG_JSONS = {}


def list_class_files():
    if not os.path.exists(FOOD_JSON_PATH):
        return []
    return [f for f in os.listdir(FOOD_JSON_PATH) if f.endswith(".json")]


def load_class_json(file_name):
    path = os.path.join(FOOD_JSON_PATH, file_name)
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def find_big_json(selected_class):
    for name, js in BIG_JSONS.items():
        if isinstance(js, dict) and selected_class in js:
            return name, js
    return None, None


def load_keras_model(model_group, model_name):
    model_path = os.path.join(MODELS_BASE, model_group, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return load_model(model_path)


def predict_image_from_file(model):
    if not os.path.exists(STATIC_TEMP):
        raise FileNotFoundError("Uploaded image not found on server.")
    img = cv2.imread(STATIC_TEMP)
    if img is None:
        raise ValueError("Failed to read uploaded image.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, 0)
    preds = model.predict(img)[0]
    idx = int(np.argmax(preds))
    return idx, float(preds[idx]), preds.tolist()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/list_json_files")
def route_list_json_files():
    return jsonify(list_class_files())


@app.route("/get_per_class")
def route_get_per_class():
    file = request.args.get("file")
    if not file:
        return jsonify({}), 400
    data = load_class_json(file)
    return jsonify(data)


@app.route("/find_big_json")
def route_find_big_json():
    cls = request.args.get("class")
    if not cls:
        return jsonify({}), 400
    name, js = find_big_json(cls)
    if not js:
        return jsonify({"found": False})
    return jsonify({
        "found": True,
        "big_json_file": name,
        "class_list": list(js.keys()),
        "metrics": js.get(cls, {})
    })


@app.route("/upload", methods=["POST"])
def route_upload():
    f = request.files.get("image")
    if not f:
        return jsonify({"error": "no file"}), 400
    os.makedirs(os.path.dirname(STATIC_TEMP), exist_ok=True)
    f.save(STATIC_TEMP)
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def route_predict():
    data = request.get_json() or {}
    selected_class = data.get("selected_class")
    model_group = data.get("model_group")

    if not selected_class:
        return jsonify({"error": "selected_class required"}), 400

    if not model_group:
        return jsonify({"error": "model_group required"}), 400

    big_file, big_json = find_big_json(selected_class)
    if not big_json:
        return jsonify({"error": f"No metrics found for {selected_class} in big JSONs"}), 400

    metrics = big_json.get(selected_class, {})

    try:
        model_file = get_model_file_for_class(selected_class, model_group)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    try:
        model = load_keras_model(model_group, model_file)
    except Exception as e:
        return jsonify({"error": f"Failed loading model: {str(e)}"}), 500

    try:
        pred_idx, conf, preds = predict_image_from_file(model)
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

    class_list = list(big_json.keys())
    predicted_class = class_list[pred_idx] if 0 <= pred_idx < len(class_list) else "IndexOutOfRange"

    response = {
        "selected_class": selected_class,
        "predicted_class": predicted_class,
        "model_used": model_group,
        "model_name": model_file,
        "confidence": round(conf, 4),
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "confusion_matrix": metrics.get("confusion_matrix"),
        "classification_report": metrics.get("classification_report"),
        "full_predictions": preds
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
