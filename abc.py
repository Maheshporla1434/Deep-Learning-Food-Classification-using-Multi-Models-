import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras

# Load your trained model
model = keras.models.load_model('./group1_custom_model.h5')
labels=['Baked Potato','Crispy Chicken','Donut']
test_dir = 'C:\\Users\\Mahesh Porla\\Downloads\\foodproject\\testing_dataset\\Group_1'  # <-- change to your test path

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

print(f"Accuracy score : {accuracy_score(y_true, y_pred_classes)*100:.2f}%")
print(f"confusion matrix: {confusion_matrix(y_true, y_pred_classes)}")
print(f"Classification Report:{classification_report(y_true, y_pred_classes, target_names=labels)}")



