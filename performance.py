import tensorflow as tf
import numpy as np
import os, json
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ------------------------------
# CONFIG
# ------------------------------
MODEL_DIR = "models"
TEST_DIR = "testing_dataset"
OUTPUT_DIR = "evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# mapping model → group folder
MODEL_FOLDER_MAP = {
    "resnet50_group1_model.h5": "Group_1",
    "resnet50_group2_model.h5": "Group_2",
    "resnet50_group3_model.h5": "Group_3",
    "resnet50_group4_model.h5": "Group_4",
    "resnet50_group5_model.h5": "Group_5",
    "resnet50_group6_model.h5": "Group_6",
    "resnet50_group7_model.h5": "Group_7",
    "resnet50_group8_model.h5": "Group_8",
    "resnet50_group9_model.h5": "Group_9",
    "resnet50_group10_model.h5": "Group_10",
    "resnet50_group11_model.h5": "Group_11",
}

TARGET_SIZE = (256, 256)  # change if your model input is different

# ------------------------------
# EVALUATION FUNCTION
# ------------------------------
def evaluate_model(model_file):
    print(f"\n📦 Evaluating {model_file} ...")

    # Load model
    model_path = os.path.join(MODEL_DIR, model_file)
    model = tf.keras.models.load_model(model_path)

    # Get test folder
    test_folder = os.path.join(TEST_DIR, MODEL_FOLDER_MAP[model_file])
    if not os.path.exists(test_folder):
        print(f"⚠️ Test folder not found: {test_folder}, skipping...")
        return

    # Prepare test data
    datagen = ImageDataGenerator(rescale=1./255)
    test_gen = datagen.flow_from_directory(
        test_folder,
        target_size=TARGET_SIZE,
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    # Predict
    y_pred_probs = model.predict(test_gen)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, target_names=list(test_gen.class_indices.keys()), output_dict=True, zero_division=0)

    # Save JSON
    result = {
        "accuracy": round(acc, 4),
        "confusion_matrix": cm,
        "classification_report": report
    }

    output_path = os.path.join(OUTPUT_DIR, model_file.replace(".h5", ".json"))
    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)

    print(f"✅ Saved results to {output_path}")

# ------------------------------
# MAIN LOOP
# ------------------------------
for model_file in MODEL_FOLDER_MAP.keys():
    evaluate_model(model_file)

print("\n🎯 All models evaluated and JSON files saved!")
