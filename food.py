import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------- CONFIG ---------------- #
MODELS_DIR = r"C:\Users\Mahesh Porla\Downloads\foodproject\models"
TEST_DATASET_DIR = r"C:\Users\Mahesh Porla\Downloads\foodproject\testing_dataset"
BATCH_SIZE = 16

MODEL_NAMES = ["custom_models", "resnet50_models", "vgg16_models"]
GROUP_NAMES = [f"Group_{i}" for i in range(1, 12)]  # Group_1 → Group_11

# --------------- FUNCTIONS ---------------- #
def load_test_data(group_name, target_size, color_mode):
    """Load test images for a group with given target size and color mode"""
    test_dir = os.path.join(TEST_DATASET_DIR, group_name)
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    datagen = ImageDataGenerator(rescale=1/255.)
    test_data = datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        color_mode=color_mode,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )
    return test_data


def evaluate_model_per_group(model_path, group_name):
    """Evaluate a single model on a single group and return group-level metrics"""
    model = tf.keras.models.load_model(model_path)

    # Detect input shape dynamically
    input_shape = model.input_shape[1:]  # exclude batch dimension
    height, width, channels = input_shape
    color_mode = 'grayscale' if channels == 1 else 'rgb'

    # Load the test dataset resized for this model
    test_data = load_test_data(group_name, target_size=(height, width), color_mode=color_mode)

    predictions = model.predict(test_data)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_data.classes
    class_names = list(test_data.class_indices.keys())

    # Multi-class confusion matrix (e.g., 3x3)
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    acc = accuracy_score(y_true, y_pred)

    group_metrics = {
        "accuracy": acc,
        "confusion_matrix": cm,
        "classification_report": report,
        "class_names": class_names
    }

    return group_metrics


def process_model(model_name):
    """Generate JSON file for a single model"""
    results = {"model_name": model_name, "groups": {}}
    model_dir = os.path.join(MODELS_DIR, model_name)

    all_h5_files = [f for f in os.listdir(model_dir) if f.endswith(".h5")]

    for group_name in GROUP_NAMES:
        # Match group number in filename
        group_number = group_name.split('_')[1]
        h5_file = next((f for f in all_h5_files if f"group{group_number}" in f.lower()), None)

        if h5_file is None:
            print(f"⚠️ No .h5 file found for {model_name} → {group_name}")
            continue

        model_file_path = os.path.join(model_dir, h5_file)
        print(f"\n📌 Evaluating {model_name} → {group_name}")

        metrics_per_group = evaluate_model_per_group(model_file_path, group_name)
        results["groups"][group_name] = metrics_per_group

    # Save JSON
    output_file = f"{model_name}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n✅ Saved results to {output_file}")


# ---------------- MAIN ---------------- #
for model_name in MODEL_NAMES:
    process_model(model_name)
