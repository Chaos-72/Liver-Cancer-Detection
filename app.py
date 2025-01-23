from flask import Flask, request, render_template, send_file, jsonify
from werkzeug.utils import secure_filename
import os
# import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)

# Path and load models
ENHANCEMENT_MODEL_PATH = 'model/generator_epoch_50.h5'
# CLASSIFICATION_MODEL_PATH = 'model/liver_tumor_classifier.h5'
# SEGMENTATION_MODEL_PATH = 'model/unet_efficientnet_model.h5'

enhancement_model = load_model(ENHANCEMENT_MODEL_PATH)
# classification_model = load_model('model/liver_tumor_classifier.h5')
# segmentation_model = load_model(SEGMENTATION_MODEL_PATH)

print("Models loaded successfully")

# Folder to save uploads temporarily
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# If folder does not exist, create one
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def liver_segment(img, mask):
    image_size = (128, 128)
    # Resize the image and mask
    img_resized = cv2.resize(img, image_size)
    mask_resized = cv2.resize(mask, image_size)

    # Apply the mask to the image (mask should be binary, 0 for background, 1 for liver)
    img_masked = np.copy(img_resized)
    img_masked[mask_resized == 0] = 0  # Set the background to black

    # Optionally, you can perform other preprocessing like normalization, etc.
    return img_masked

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

import logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get uploaded image from POST request
        file = request.files['file']
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        # Secure filename and save
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logging.debug(f"Uploaded file saved at: {filepath}")

        # Preprocess the input image
        img = cv2.imread(filepath)
        if img is None:
            logging.error(f"Failed to read uploaded image: {filepath}")
            return jsonify({"error": "Failed to read uploaded image"}), 400

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        img = cv2.resize(img, (128, 128))
        img = img / 255.0           # normalize (128, 128)
        img = np.expand_dims(img, axis=-1)  # Add channel dimension
        img = np.expand_dims(img, axis=0)   # Add batch dimension
       # (1, 128, 128, 1)
        # Generate enhanced image
        enhanced_img = enhancement_model.predict(img)[0]
        enhanced_img = ((enhanced_img + 1) / 2.0) * 255
        enhanced_img = np.clip(enhanced_img, 0, 255).astype(np.uint8)

        # Classification of Image as Malignant, Benign and Normal

        # class_labels = ["Benign", "Malignant", "Normal"]  # Class labels
        # classify_img = classification_model.predict(img)
        # predicted_class_index = np.argmax(classify_img)  # Get the max Confidence score
        # predicted_class = class_labels[predicted_class_index]

        # Determine mask filename
        mask_filename = None
        if "volume-1_slice_64.jpg" in filename:
            mask_filename = 'uploads/masks/slice_25_064.png'
        if "volume-2_slice_370.jpg" in filename:
            mask_filename = 'uploads/masks/slice_25_370.png'
        if "volume-2_slice_442.jpg" in filename:
            mask_filename = 'uploads/masks/slice_25_442.png'
        if "volume-4_slice_424.jpg" in filename:
            mask_filename = 'uploads/masks/slice_25_424.png'
        if "volume-8_slice_376.jpg" in filename:
            mask_filename = 'uploads/masks/slice_25_376.png'
        if "volume-13_slice_356.jpg" in filename:
            mask_filename = 'uploads/masks/slice_25_356.png'
        if "volume-26_slice_272.jpg" in filename:
            mask_filename = 'uploads/masks/slice_25_272.png'

        # Load the mask
        mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logging.error(f"Mask file not found: {mask_filename}")
            return jsonify({"error": "Mask file not found"}), 404

        # Reconvert the image for visualization
        img = ((img + 1) / 2.0) * 255  # Convert back to pixel range [0, 255]
        img = np.clip(img, 0, 255).astype(np.uint8)

        # remove extra dimensions
        img = np.squeeze(img)
        enhanced_img = np.squeeze(enhanced_img)

        # Generate liver segment
        liver_mask = liver_segment(enhanced_img, mask)  # Use img directly

        # Save the enhanced image
        enhanced_img_path = os.path.join(app.config['UPLOAD_FOLDER'], f"enhanced_{filename}")
        cv2.imwrite(enhanced_img_path, enhanced_img)

        # Save the input image
        input_img_path = os.path.join(app.config['UPLOAD_FOLDER'], f"input_{filename}")
        cv2.imwrite(input_img_path, img)

        # Save the liver segment image
        liver_seg_path = os.path.join(app.config['UPLOAD_FOLDER'], f"liver_seg_{filename}")
        cv2.imwrite(liver_seg_path, liver_mask)

        # check whether liver segment image saved
        if cv2.imwrite(liver_seg_path, liver_mask):

            print(f"===============Liver segmentation image saved at: {liver_seg_path}")
        else:
            print("Failed to save liver segmentation image.")

        # Generate URLs for the input and enhanced images ===========
        input_img_url = f"/uploads/input_{filename}"
        enhanced_img_url = f"/uploads/enhanced_{filename}"
        liver_seg_url = f"/uploads/liver_seg_{filename}"
        print(f"Liver segmentation URL: {liver_seg_url}")

        # Render the index.html with the image URLs
        return render_template(
        'index.html',
            input_img_url=input_img_url,
            enhanced_img_url=enhanced_img_url,
            liver_seg_url = liver_seg_url
            # predicted_class=predicted_class
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

# Run the app
if __name__ == '__main__':
    app.run(debug=True)