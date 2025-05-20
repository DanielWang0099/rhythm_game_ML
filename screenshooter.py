import pyautogui
import time
import os
import win32api
from datetime import datetime

# Create training_images directory if it doesn't exist
if not os.path.exists('training_images'):
    os.makedirs('training_images')

# Screenshot parameters
x = 1120
y = 711
width = 100
height = 100
interval = 0.1

counter = 0

try:
    while True:
        # Check if left mouse button is being held down
        if win32api.GetKeyState(0x01) < 0:  # 0x01 is the code for left mouse button
            # Take screenshot of the specified region
            screenshot = pyautogui.screenshot(region=(x, y, width, height))
            
            # Generate filename with timestamp and counter
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'training_images/screenshot_{timestamp}_{counter:06d}.png'
            
            # Save the screenshot
            screenshot.save(filename)
            
            # Increment counter
            counter += 1
            
            # Wait for the specified interval
            time.sleep(interval)
        else:
            # Small sleep to prevent high CPU usage when not capturing
            time.sleep(0.01)
        
except KeyboardInterrupt:
    print("Screenshot capture stopped by user")