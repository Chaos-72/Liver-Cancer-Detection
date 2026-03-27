# CT Scan Image Enhancement for Liver Cancer Detection

This project implements a deep learning-based system designed to enhance computed tomography (CT) images, specifically for improving the accuracy of liver cancer detection. By leveraging Generative Adversarial Networks (GANs), the system addresses common diagnostic hurdles such as low image contrast, noise, and complex anatomical structures that often impede early-stage tumor identification.

---

## System Workflow

The core pipeline follows an adversarial training process where a generator network refines low-quality CT scans, and a discriminator network evaluates the authenticity of the enhancements.


1.  **Input**: Non-enhanced, low-contrast CT images.
2.  **Generation**: The Generative Network $G(z)$ creates an enhanced version of the input.
3.  **Loss Calculation**: The system applies Perceptual Loss and Wasserstein GAN (WGAN) loss to ensure structural fidelity.
4.  **Discrimination**: The Discriminator Network $D(x)$ distinguishes between the generated image and high-quality reference scans.
5.  **Optimization**: An iterative process of error maximization for the discriminator and error minimization for the generator refines the output until it is indistinguishable from high-quality medical imaging.
6.  **Classification**: Enhanced images are passed to a Convolutional Neural Network (CNN) to distinguish between malignant and normal liver tissues.

---

## Data Collection and Datasets

The system is developed and validated using a variety of publicly available, high-resolution medical datasets:

* **Liver Tumor Segmentation Challenge (LiTS)**: Contains detailed liver and tumor annotations across multiple patients, essential for training robust segmentation algorithms.
* **Sliver07**: Offers a standardized set of images used for benchmarking liver segmentation methods.
* **Ircadb**: Provides high-resolution scans that assist in the fine-tuning of the image enhancement models.

---

## Data Preprocessing

A rigorous preprocessing phase ensures consistency across different imaging conditions and scanner types.

* **Intensity Normalization**: Normalizes pixel intensity values to focus the model on pathological features rather than hardware-induced variations.
* **Artifact Removal**: Filters are applied to eliminate specks and artifacts that might distort image data.
* **Contrast Enhancement**: Histogram equalization is utilized to improve the initial visual quality of the CT scans.

---

## Model Architecture

The architecture utilizes a dual-network GAN setup combined with a classification-specific CNN.

### 1. Generative Network
The generator is designed to learn the complex mapping from low-quality inputs to high-quality outputs. It focuses on preserving critical structural details while reducing noise.

### 2. Discriminator Network
The discriminator is trained to differentiate between real, high-quality reference images and the synthetic outputs produced by the generator.

### 3. Classification CNN
A specialized Convolutional Neural Network extracts intricate features from the GAN-enhanced images to facilitate tissue classification. This network uses advanced techniques like data augmentation (rotation, scaling, and translation) to improve its robustness against image acquisition variations.

---

## Loss Functions

Two primary loss functions guide the adversarial training process:

* **Perceptual Loss**: Ensures that the generated images maintain the semantic and structural integrity of the original liver anatomy.
* **Wasserstein Loss (WGAN)**: Provides more stable training and higher-quality image generation compared to traditional GAN loss functions.

---

## Training Procedure

* **Initialization**: GAN networks are initialized with random weights.
* **Adversarial Training**: The generator and discriminator are trained iteratively. The generator continuously improves its output to "fool" the discriminator, while the discriminator learns to be more precise in its evaluations.
* **Validation**: To prevent overfitting and ensure generalizability, the model's performance is validated on a separate, unseen set of images.
* **Optimization**: High-performance computing resources, such as GPUs and cloud platforms, are used to manage large-scale data processing and intensive training cycles.

---

## Results and Performance Analysis

The application of GAN-based enhancement significantly improves both visual clarity and diagnostic metrics.

### Image Quality Metrics
| Metric | Original Image (Average) | GAN-Enhanced Image (Average) |
| :--- | :--- | :--- |
| **Structural Similarity Index (SSIM)** | 0.72  | **0.91** |
| **Peak Signal-to-Noise Ratio (PSNR)** | 22.5 dB  | **31.8 dB**  |

### Diagnostic Accuracy
* **Tumor Segmentation**: The Dice Similarity Coefficient (DSC), measuring the overlap between predicted and ground-truth segmentation, increased from **0.68 to 0.85**.
* **Classification Accuracy**: The ability to distinguish between malignant and normal tissues improved from **78% to 92%**.


---

## Implementation Details

* **Frameworks**: The system is built using **TensorFlow**, allowing for the development of complex neural network architectures and the management of large-scale datasets.
* **Platform**: The imaging system is integrated into a **web-based platform** that supports real-time image analysis. This ensures the solution is scalable and accessible for clinical use globally.

### 📄 Publication

[CT Scan Image Enhancement for Liver Cancer Detection](https://www.ijraset.com/research-paper/ct-scan-image-enhancement-for-liver-cancer-detection)

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
