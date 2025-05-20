# Rhythm Game AI Bot

An end-to-end machine learning project that creates an AI bot capable of playing rhythm games in real-time. The project includes everything from data collection to model training and real-time gameplay automation.

## Project Overview

This project demonstrates the complete pipeline of creating an AI-powered game bot:
1. Data collection through screen capture
2. Dataset creation and organization
3. Model training using transfer learning
4. Real-time inference and gameplay automation

## Features

- **Automated Data Collection**: Captures and organizes gameplay screenshots
- **Transfer Learning Model**: Uses MobileNetV2 for efficient real-time inference
- **Real-time Game Control**: Automatically detects and responds to in-game prompts
- **Performance Optimized**: Includes frame similarity detection and throttling for efficient processing
- **GUI Monitoring**: Optional GUI interface to monitor bot performance (in main.py)
- **Lightweight Version**: Simplified version without GUI (in auto_bot.py)

## Requirements

- Python 3.9+
- Windows OS (for keyboard control)
- Administrator privileges (for key input simulation)

## Installation

1. Clone the repository:
```powershell
git clone [repository-url]
cd rhythm-game-bot
```

2. Create a virtual environment (recommended):
```powershell
python -m venv venv
.\venv\Scripts\Activate
```

3. Install dependencies:
```powershell
pip install -r requirements.txt
```

## Project Structure

```
├── main.py               # Main bot with GUI monitoring
├── auto_bot.py          # Simplified bot without GUI
├── model_training.py    # Model training script
├── screenshooter.py     # Data collection script
├── dataset/             # Training and validation data
│   ├── train/          # Training images
│   └── val/            # Validation images
└── training_images/     # Raw captured images
```

## Usage

### 1. Data Collection

Run the screenshooter to collect training data:
```powershell
python screenshooter.py
```
- Hold left mouse button to capture screenshots
- Images will be saved in the training_images folder
- Manually sort images into appropriate categories in dataset/train and dataset/val

### 2. Model Training

Train the model on your collected data:
```powershell
python model_training.py
```
This will:
- Load and preprocess your dataset
- Train a MobileNetV2-based model
- Save the trained model as 'rhythm_game_model.h5'
- Also save a TensorFlow Lite version for efficiency

### 3. Running the Bot

#### With GUI Monitoring
```powershell
# Run as administrator
python main.py
```
- Press 't' to toggle the bot on/off
- GUI shows real-time status and predictions

#### Lightweight Version
```powershell
# Run as administrator
python auto_bot.py
```
- Starts automatically
- Prints status to console

## Configuration

Key parameters can be adjusted in the scripts:
- Screenshot region (x, y, width, height)
- Confidence threshold (default: 0.6)
- Processing interval (default: 0.01s)
- Key mappings can be customized in the key_mapping dictionary

## Troubleshooting

1. **Keys not registering:**
   - Ensure script is run as administrator
   - Check game window has focus
   - Verify key mappings match your game settings

2. **High CPU Usage:**
   - Adjust THROTTLE_TIME and interval values
   - Use auto_bot.py for better performance

3. **Low Accuracy:**
   - Collect more training data
   - Adjust confidence threshold
   - Check screenshot region alignment

## License

[Your chosen license]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Disclaimer

This project is for educational purposes only. Please ensure you comply with the terms of service of any games you use this with.
