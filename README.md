# Liver Cancer Detection System Using AI

## Overview
This project presents an AI-powered system for liver cancer detection that combines state-of-the-art techniques for image enhancement, tumor segmentation, and classification. It provides a user-friendly web interface for medical professionals to upload liver CT scans and receive processed outputs for better diagnosis.

---

## Features
- **Image Enhancement**: Uses Generative Adversarial Networks (GANs) to enhance CT scan image quality.
- **Tumor Segmentation**: Accurately isolates liver regions and tumor areas.
- **Tumor Classification**: Categorizes tumors into Malignant, Benign, or Normal.
- **User-Friendly Interface**: Intuitive web-based platform for seamless interaction.

---

## Technology Stack
- **Frontend**: HTML, CSS, JavaScript, Bootstrap.
- **Backend**: Flask Framework.
- **AI Models**: TensorFlow, Keras.
- **Image Processing**: OpenCV, NumPy.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Chaos-72/Liver-Cancer-Detection.git

2. Navigate to the project directory:
   ```bash
   cd liver-cancer-detection

3. Create a virtual environment:
   ```bash
   python -m venv env

4. Activate the virtual environment:
   ```bash
   env\Scripts\activate

5. Install requirement dependencies:
   ```bash
   pip install -r requirements.txt

6. Set the Flask app and run it:
   ```bash
   export FLASK_APP=app.py
   flask run

## Usage

1. Launch the application:
   ```bash
   flask run --debug

2. Open the provided URL in your browser (default: http://127.0.0.1:5000).
3. Upload a liver CT scan image via the web interface.
4. View the enhanced image, segmented liver region, and classification results.

## Folder Structure
```
├── model
│   ├── generator_epoch_50.h5       # GAN model for image enhancement
│   ├── liver_tumor_classifier.h5   # Model for tumor classification
│   ├── unet_efficientnet_model.h5  # Model for liver segmentation
├── static
│   ├── CSS                         # Frontend stylesheets
│   ├── Images                      # Frontend assets
├── templates
│   ├── index.html                  # Main web interface
├── uploads                         # Folder for user uploads
├── app.py                          # Flask application code
├── requirements.txt                # Python dependencies
```

## Demo

Upload a liver CT scan image to the platform and receive:

1. Enhanced Image: High-quality version of the original scan.
2. Liver Segmentation: Region of interest isolated for analysis.
3. Tumor Classification: Diagnosis as Malignant, Benign, or Normal.
