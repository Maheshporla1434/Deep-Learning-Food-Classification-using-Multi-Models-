import json
import logging
from log_code import Logger
logger=Logger.get_logs('jsonfiles')
import warnings
warnings.filterwarnings('ignore')
import os
def json_files():
    logger.info('json files creation started')
    output_folder = "food_json_files"
    os.makedirs(output_folder, exist_ok=True)
    food_nutrition = {
        "apple_pie": {"protein": 2.4, "fiber": 1.0, "calories": 237, "carbohydrates": 34.0, "fat": 11.0},
        "Baked Potato": {"protein": 4.3, "fiber": 3.8, "calories": 161, "carbohydrates": 37.0, "fat": 0.2},
        "burger": {"protein": 17.0, "fiber": 1.5, "calories": 295, "carbohydrates": 30.0, "fat": 12.0},
        "butter_naan": {"protein": 8.0, "fiber": 2.1, "calories": 300, "carbohydrates": 45.0, "fat": 10.0},
        "chai": {"protein": 1.0, "fiber": 0.0, "calories": 120, "carbohydrates": 20.0, "fat": 4.0},
        "chapati": {"protein": 3.0, "fiber": 2.0, "calories": 120, "carbohydrates": 18.0, "fat": 4.0},
        "cheesecake": {"protein": 7.0, "fiber": 0.4, "calories": 321, "carbohydrates": 25.0, "fat": 23.0},
        "chicken_curry": {"protein": 27.0, "fiber": 2.0, "calories": 270, "carbohydrates": 8.0, "fat": 15.0},
        "chole_bhature": {"protein": 12.0, "fiber": 6.0, "calories": 427, "carbohydrates": 45.0, "fat": 18.0},
        "Crispy Chicken": {"protein": 26.0, "fiber": 0.0, "calories": 320, "carbohydrates": 16.0, "fat": 18.0},
        "dal_makhani": {"protein": 12.0, "fiber": 7.0, "calories": 278, "carbohydrates": 32.0, "fat": 11.0},
        "dhokla": {"protein": 5.0, "fiber": 2.0, "calories": 150, "carbohydrates": 22.0, "fat": 5.0},
        "Donut": {"protein": 3.0, "fiber": 0.8, "calories": 195, "carbohydrates": 22.0, "fat": 11.0},
        "fried_rice": {"protein": 8.0, "fiber": 1.5, "calories": 250, "carbohydrates": 45.0, "fat": 6.0},
        "Fries": {"protein": 3.4, "fiber": 3.8, "calories": 312, "carbohydrates": 41.0, "fat": 15.0},
        "Hot Dog": {"protein": 11.0, "fiber": 0.5, "calories": 290, "carbohydrates": 24.0, "fat": 18.0},
        "ice_cream": {"protein": 3.5, "fiber": 0.0, "calories": 207, "carbohydrates": 24.0, "fat": 11.0},
        "idli": {"protein": 2.0, "fiber": 0.6, "calories": 58, "carbohydrates": 12.0, "fat": 0.4},
        "jalebi": {"protein": 0.5, "fiber": 0.1, "calories": 150, "carbohydrates": 35.0, "fat": 4.0},
        "kaathi_rolls": {"protein": 10.0, "fiber": 2.0, "calories": 320, "carbohydrates": 35.0, "fat": 14.0},
        "kadai_paneer": {"protein": 18.0, "fiber": 3.0, "calories": 350, "carbohydrates": 12.0, "fat": 26.0},
        "kulfi": {"protein": 6.0, "fiber": 0.0, "calories": 200, "carbohydrates": 20.0, "fat": 10.0},
        "masala_dosa": {"protein": 6.0, "fiber": 2.5, "calories": 168, "carbohydrates": 25.0, "fat": 6.0},
        "momos": {"protein": 8.0, "fiber": 1.0, "calories": 175, "carbohydrates": 25.0, "fat": 5.0},
        "omelette": {"protein": 12.0, "fiber": 0.0, "calories": 150, "carbohydrates": 1.0, "fat": 13.0},
        "paani_puri": {"protein": 2.5, "fiber": 1.0, "calories": 180, "carbohydrates": 25.0, "fat": 8.0},
        "pakode": {"protein": 6.0, "fiber": 2.0, "calories": 230, "carbohydrates": 20.0, "fat": 14.0},
        "pav_bhaji": {"protein": 7.0, "fiber": 5.0, "calories": 390, "carbohydrates": 45.0, "fat": 18.0},
        "pizza": {"protein": 11.0, "fiber": 2.5, "calories": 266, "carbohydrates": 33.0, "fat": 10.0},
        "samosa": {"protein": 4.0, "fiber": 1.5, "calories": 262, "carbohydrates": 31.0, "fat": 13.0},
        "Sandwich": {"protein": 12.0, "fiber": 3.0, "calories": 250, "carbohydrates": 30.0, "fat": 8.0},
        "sushi": {"protein": 6.0, "fiber": 0.6, "calories": 200, "carbohydrates": 28.0, "fat": 5.0},
        "Taco": {"protein": 13.0, "fiber": 3.0, "calories": 226, "carbohydrates": 19.0, "fat": 12.0},
        "Taquito": {"protein": 8.0, "fiber": 1.0, "calories": 220, "carbohydrates": 23.0, "fat": 10.0}
    }

    # Create individual JSON files
    for food, values in food_nutrition.items():
        file_path = os.path.join(output_folder, f"{food}.json")
        with open(file_path, 'w') as f:
            json.dump({food: values}, f, indent=4)
    logger.info('json files creation completed successfully')