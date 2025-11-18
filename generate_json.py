import os
import json
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ---------------- CONFIG ---------------- #
MODELS = ['custom_models', 'resnet50_models', 'vgg16_models']
MODEL_DIR = 'models'
TEST_DIR = 'testing_dataset'
OUTPUT_DIR = 'model_json_files'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Exact class names in your dataset
ALL_CLASSES = [
    "Baked Potato", "Crispy Chicken", "Donut", "Fries", "Hot Dog", "Sandwich",
    "apple_pie", "Taco", "Taquito", "burger", "butter_naan", "chai",
    "chapati", "cheesecake", "chicken_curry", "chole_bhature", "dal_makhani", "dhokla",
    "fried_rice", "ice_cream", "idli", "jalebi", "kaathi_rolls", "kadai_paneer",
    "kulfi", "masala_dosa", "momos", "omelette", "paani_puri", "pakode",
    "pav_bhaji", "pizza", "samosa", "sushi"
]

# ---------------- HELPERS ---------------- #
def get_group_for_class(cls_name):
    """Determine which group (Group_1..Group_11) the class belongs to."""
    for i, start in enumerate(range(0, len(ALL_CLASSES), 3)):
        if cls_name in ALL_CLASSES[start:start+3]:
            return f'Group_{i+1}'
    return None

def get_model_file(model_folder, model_name, group_name):
    """
    Robustly find the .h5 model file for a given model type and group.
    """
    group_name_lower = group_name.lower().replace("_", "")
    for f in os.listdir(model_folder):
        f_lower = f.lower().replace("_", "")
        if model_name == "custom_models" and f_lower.startswith("custommodel") and group_name_lower in f_lower:
            return os.path.join(model_folder, f)
        elif model_name == "resnet50_models" and f_lower.startswith("resnet50") and group_name_lower in f_lower:
            return os.path.join(model_folder, f)
        elif model_name == "vgg16_models" and f_lower.startswith("vgg16") and group_name_lower in f_lower:
            return os.path.join(model_folder, f)
    return None

def load_test_data(group_name, target_size=(224, 224), batch_size=16):
    """Load testing data for a group using ImageDataGenerator."""
    datagen = ImageDataGenerator(rescale=1./255)
    test_dir = os.path.join(TEST_DIR, group_name)
    return datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

# ---------------- MAIN ---------------- #
for model_name in MODELS:
    print(f"\nProcessing model: {model_name}")
    model_folder = os.path.join(MODEL_DIR, model_name)
    json_output = {}

    for cls in ALL_CLASSES:
        group_name = get_group_for_class(cls)
        if group_name is None:
            continue

        # Find correct model file
        model_file = get_model_file(model_folder, model_name, group_name)
        if model_file is None:
            print(f"Model file not found for {cls} in {model_name}")
            json_output[cls] = {
                'model_used': model_name,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'true_positive': 0,
                'true_negative': 0,
                'false_positive': 0,
                'false_negative': 0,
                'confusion_matrix': [[0,0,0],[0,0,0],[0,0,0]],
                'classification_report': {}
            }
            continue

        # Load model
        model = load_model(model_file)

        # Load testing data
        target_size = (model.input_shape[1], model.input_shape[2])
        test_data = load_test_data(group_name, target_size=target_size, batch_size=16)

        # Predict
        y_true = test_data.classes
        class_indices = test_data.class_indices
        class_list = list(class_indices.keys())
        y_pred_probs = model.predict(test_data, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Metrics
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        cr = classification_report(y_true, y_pred, target_names=class_list, output_dict=True)

        # Compute TP, TN, FP, FN for this class
        if cls in class_list:
            idx = class_list.index(cls)
            tp = int(cm[idx, idx])
            fn = int(np.sum(cm[idx, :]) - tp)
            fp = int(np.sum(cm[:, idx]) - tp)
            tn = int(np.sum(cm) - (tp + fp + fn))
        else:
            tp = tn = fp = fn = 0

        json_output[cls] = {
            'model_used': model_name,
            'accuracy': float(acc),
            'precision': float(cr[cls]['precision']) if cls in cr else 0.0,
            'recall': float(cr[cls]['recall']) if cls in cr else 0.0,
            'true_positive': tp,
            'true_negative': tn,
            'false_positive': fp,
            'false_negative': fn,
            'confusion_matrix': cm.tolist(),
            'classification_report': cr
        }

        print(f"Processed class: {cls} in {model_name}")

    # Save JSON
    output_file = os.path.join(OUTPUT_DIR, f"{model_name}_full.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=4)
    print(f"JSON saved for {model_name} → {output_file}")
