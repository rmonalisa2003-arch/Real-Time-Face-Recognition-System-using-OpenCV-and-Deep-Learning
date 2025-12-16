# Real-Time Face Recognition System

This project is a deep-learning-based face recognition system that uses a Convolutional Neural Network (CNN) and OpenCV to identify individuals in real-time. It is designed to work with an IP Webcam stream, making it compatible with mobile device cameras.

## Features
- **Real-time Detection:** Uses Haar Cascade Classifiers for fast face localization.
- **CNN Classification:** A trained Keras model (`final_model.h5`) for high-accuracy recognition.
- **Mobile Integration:** Connects to mobile cameras via IP Webcam (urllib).
- **Automated Pipeline:** Scripts for data collection, preprocessing, and live inference.

## Project Structure
- `collect_data.py`: Captures 100 face samples from the live stream and saves them.
- `consolidated_data.py`: Preprocesses images (grayscale, 100x100) and prepares the dataset.
- `recognize.py`: The main execution script for real-time face recognition.
- `final_model.h5`: The saved CNN model weights.
- `haarcascade_frontalface_default.xml`: Pre-trained face detection model.

## Prerequisites
Ensure you have Python installed, then install the required libraries:
```bash
pip install opencv-python tensorflow numpy keras

### How to Use
1. Data Collection
Open collect_data.py, update the url variable with your IP Webcam address, and run:

Bash

python collect_data.py
It will capture 100 samples of your face.

2. Data Preparation
Run the consolidation script to process your captured images into a format suitable for the model:

Bash

python consolidated_data.py
3. Face Recognition
Launch the recognition system to start identifying faces:

Bash

python recognize.py
##Model Details
The system uses a Convolutional Neural Network (CNN). Images are converted to grayscale and resized to 100x100 pixels before being fed into the model. The labels (names) are managed via a list in the prediction function.

##License
This project is open-source and available under the MIT License.

### **One Final Step (Requirements File)**
For a professional GitHub repo, you should also create a file named `requirements.txt` and paste this inside:
```text
opencv-python
tensorflow
numpy
keras
This allows other users to install everything at once by typing pip install -r requirements.txt.
