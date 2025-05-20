import pyautogui
import time
import os
import sys
import keyboard
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Disable pyautogui's fail-safe
pyautogui.FAILSAFE = False

print("WARNING: This script requires administrator privileges to send keystrokes.")
print("Please run this script as administrator if keys are not being registered.")

# Screenshot parameters
x = 1120
y = 711
width = 100
height = 100
interval = 0.01 # 10ms between screenshots
THROTTLE_TIME = 0.01 # 10ms minimum between predictions

# Key mapping
key_mapping = {
    'up': 'w',
    'down': 's',
    'left': 'a',
    'right': 'd',
    'space': 'space',
    'nothing': None
}

def load_model_safe():
    try:
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'rhythm_game_model.h5')
        return load_model(model_path)
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        sys.exit(1)

def preprocess_image(image):
    try:
        img_array = img_to_array(image)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None

def get_class_names():
    try:
        dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset', 'train')
        if not os.path.exists(dataset_path):
            raise Exception("Training dataset directory not found")
        classes = sorted(os.listdir(dataset_path))
        if not classes:
            raise Exception("No classes found in training directory")
        return classes
    except Exception as e:
        print(f"Failed to load class names: {str(e)}")
        sys.exit(1)

def press_key(key):
    if key:
        try:
            # Using pyautogui as an alternative to keyboard library
            pyautogui.keyDown(key)
            time.sleep(0.05)  # Hold for 50ms
            pyautogui.keyUp(key)
            print(f"Pressed key: {key}")
        except Exception as e:
            print(f"Failed to press key {key}: {e}")
            # Fallback to keyboard library if pyautogui fails
            try:
                keyboard.press(key)
                time.sleep(0.05)
                keyboard.release(key)
                print(f"Pressed key (fallback): {key}")
            except Exception as e2:
                print(f"Fallback also failed: {e2}")

# Load model and class names
print("Loading model and initializing...")
model = load_model_safe()
class_names = get_class_names()
print("Initialization complete. Bot is now running.")

# Main loop variables
last_processed_time = 0
prediction_cache = None

try:
    while True:
        current_time = time.time()
        
        # Only process if enough time has passed
        if current_time - last_processed_time >= interval:
            try:
                # Take screenshot
                screenshot = pyautogui.screenshot(region=(x, y, width, height))
                
                # Preprocess the image
                processed_image = preprocess_image(screenshot)
                if processed_image is None:
                    time.sleep(THROTTLE_TIME)
                    continue
                
                # Check frame similarity
                if prediction_cache is not None:
                    new_frame = processed_image.reshape(-1)
                    if np.mean(np.abs(new_frame - prediction_cache['frame'])) < 0.1:
                        time.sleep(THROTTLE_TIME)
                        continue
                
                # Make prediction
                prediction = model.predict(processed_image, verbose=0)
                predicted_class = class_names[np.argmax(prediction[0])]
                confidence = np.max(prediction[0])
                
                # Cache the prediction and frame
                prediction_cache = {
                    'frame': processed_image.reshape(-1),
                    'class': predicted_class,
                    'confidence': confidence
                }
                
                # Only act if confidence is high enough
                if confidence > 0.6:
                    print(f"Detected: {predicted_class} ({confidence:.2f})")
                    key_to_press = key_mapping.get(predicted_class)
                    press_key(key_to_press)
                
                last_processed_time = current_time
                
            except Exception as e:
                print(f"Error in main loop: {str(e)}")
                time.sleep(THROTTLE_TIME)
        else:
            # Short sleep when skipping processing
            time.sleep(THROTTLE_TIME)

except KeyboardInterrupt:
    print("\nBot terminated by user.")
