# CattleVision — Image-Based Cattle Breed Recognition

An AI-powered web application for cattle detection, breed classification, confidence scoring, and visual explanation using YOLO, MobileNetV2, Grad-CAM, and Streamlit.

## Overview

CattleVision is a computer vision project that identifies cattle breeds from uploaded images. It combines a pretrained YOLO object detector with a fine-tuned MobileNetV2 classification model to provide breed predictions through a user-friendly web interface.

The application also integrates Grad-CAM (Gradient-weighted Class Activation Mapping) to generate heatmap visualizations of image regions that influenced the model's selected prediction.

**Current implementation:** The classifier recognizes five cattle breeds. Buffalo breed recognition is planned as future work.

## Features

* **Cattle Detection:** Uses YOLO11n pretrained on the COCO dataset to check for cattle in uploaded images.
* **Breed Classification:** Uses a fine-tuned MobileNetV2 model to classify cattle into five supported breeds.
* **Confidence Scoring:** Displays the model's prediction confidence.
* **Top-3 Predictions:** Shows the three highest-scoring breed predictions.
* **Explainable AI:** Uses Grad-CAM to visualize regions contributing to the selected class prediction.
* **Interactive Web Interface:** Provides image upload, preview, prediction results, and visualization through Streamlit.
* **Input Validation:** Handles unsupported or invalid images and images in which cattle are not detected.

## Supported Cattle Breeds

The current classification model supports:

1. Ayrshire
2. Brown Swiss
3. Holstein Friesian
4. Jersey
5. Red Dane

> **Scope limitation:** The current model does not classify buffalo breeds. The YOLO detector is a general-purpose detector, not a dedicated Indian cattle and buffalo detection model.

## Technology Stack

| Component               | Technology         |
| ----------------------- | ------------------ |
| Programming language    | Python             |
| Web application         | Streamlit          |
| Object detection        | YOLO11n            |
| Image classification    | MobileNetV2        |
| Deep learning framework | TensorFlow / Keras |
| Explainable AI          | Grad-CAM           |
| Image processing        | Pillow, NumPy      |
| Evaluation              | Scikit-learn       |
| Visualization           | Matplotlib         |

## System Architecture

The application follows a sequential image-processing pipeline:

1. **Image Upload:** The user uploads a cattle image through the Streamlit interface.
2. **Image Preprocessing:** The image is converted to RGB and resized to 224 × 224 pixels for classification.
3. **Cattle Detection:** YOLO checks whether a cow is detected in the image.
4. **Breed Classification:** The fine-tuned MobileNetV2 model generates scores for the five supported breeds.
5. **Confidence and Ranking:** The application displays the predicted breed, confidence score, and top-three predictions.
6. **Grad-CAM Visualization:** A heatmap is generated to highlight image regions that contributed to the selected class prediction.
7. **Result Display:** The user views the classification results and visualization in the Streamlit application.

## Dataset

The dataset currently contains 1,208 valid images distributed across five cattle breeds.

| Breed             | Number of Images |
| ----------------- | ---------------: |
| Ayrshire          |              260 |
| Brown Swiss       |              238 |
| Holstein Friesian |              254 |
| Jersey            |              252 |
| Red Dane          |              204 |
| **Total**         |        **1,208** |

### Dataset Split

The dataset was divided into training, validation, and testing subsets.

| Subset     | Number of Images | Approximate Percentage |
| ---------- | ---------------: | ---------------------: |
| Training   |              843 |                  69.8% |
| Validation |              182 |                  15.1% |
| Testing    |              183 |                  15.1% |
| **Total**  |        **1,208** |               **100%** |

The training subset is used to train the model, the validation subset is used to monitor training performance, and the test subset is used for final evaluation.

## Model Training

MobileNetV2 was used as the base architecture because it provides a relatively lightweight convolutional neural network suitable for image classification.

The training workflow consists of:

1. Loading and preprocessing the dataset.
2. Initializing MobileNetV2 with pretrained ImageNet weights.
3. Training the classification head for the five cattle breeds.
4. Applying data augmentation during fine-tuning.
5. Unfreezing selected upper layers of the pretrained network.
6. Fine-tuning with a low learning rate.
7. Saving the best model checkpoint.
8. Evaluating the trained model on the held-out test dataset.

The application uses the fine-tuned best checkpoint:

`models/mobilenetv2_finetuned_best.keras`

## Model Evaluation

The fine-tuned MobileNetV2 model achieved the following results on the held-out test dataset.

| Metric              | Result |
| ------------------- | -----: |
| Test accuracy       | 89.62% |
| Macro precision     | 89.71% |
| Macro recall        | 89.87% |
| Macro F1-score      | 89.61% |
| Weighted F1-score   | 89.55% |
| Test images         |    183 |
| Correct predictions |    164 |

These results represent performance on the current test dataset. Actual predictions may vary depending on image quality, lighting, pose, background, and visual similarities between breeds.

## Project Structure

```text
CattleVision/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── dataset/
│   └── README.md
│
├── models/
│   ├── class_names.json
│   ├── mobilenetv2_finetuned_best.keras
│   ├── mobilenetv2_confusion_matrix.png
│   └── training_accuracy.png
│
├── src/
│   ├── cattle_detector.py
│   ├── gradcam.py
│   ├── predict_breed.py
│   ├── train_mobilenetv2.py
│   ├── finetune_mobilenetv2.py
│   ├── evaluate_mobilenetv2.py
│   ├── evaluate_finetuned.py
│   ├── split_dataset.py
│   ├── inspect_dataset.py
│   ├── plot_training_history.py
│   └── test_cattle_detector.py
│
└── screenshots/
    ├── home_page.png
    ├── breed_prediction.png
    └── gradcam_visualization.png
```

*This is the recommended repository structure. Your actual directory may contain additional files.*

## Installation and Setup

### Prerequisites

* Python 3.10 or another Python version compatible with the selected TensorFlow release.
* pip package manager.
* Git.
* An internet connection for installing dependencies and downloading pretrained YOLO weights when they are not already cached.

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd CattleVision
```

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with your actual repository URL.

### 2. Create a Virtual Environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Ensure that `requirements.txt` contains the packages and compatible versions required by your application.

### 4. Verify Required Files

Confirm that the following files are available:

```text
app.py
models/class_names.json
models/mobilenetv2_finetuned_best.keras
src/cattle_detector.py
src/gradcam.py
```

The dataset is required only if you intend to rerun training, evaluation, or other dataset-dependent scripts.

### 5. Run the Application

From the project root, execute:

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser to use CattleVision.

## How to Use

1. Start the Streamlit application.
2. Upload a supported image containing cattle.
3. Allow the application to check whether a cow is detected.
4. View the predicted breed and confidence score.
5. Review the top-three breed predictions.
6. Inspect the Grad-CAM heatmap overlay.

If the detector does not identify a cow, the application displays a rejection message rather than proceeding to breed classification.

## Explainable AI with Grad-CAM

Grad-CAM creates a heatmap using gradients associated with the selected class and feature maps from a convolutional layer.

The heatmap is overlaid on the original image to help users inspect which regions influenced the model's prediction.

Grad-CAM provides a visual explanation of model behavior; it does not prove that the model is relying on biologically meaningful breed characteristics.

## Limitations

* The classifier supports only five cattle breeds.
* Buffalo breed recognition is not currently implemented.
* The YOLO detector is pretrained on COCO and may fail to detect some cattle images or produce false detections.
* Classification accuracy depends on the quality and diversity of the training dataset.
* Visually similar breeds may be confused by the model.
* Confidence scores are model outputs and should not automatically be interpreted as calibrated probabilities.
* Grad-CAM visualizations are explanatory aids, not proof of prediction correctness.
* The reported test accuracy does not guarantee the same performance on new, real-world images.

## Future Enhancements

* Extend the dataset to include Indian cattle and buffalo breeds.
* Improve detection using a cattle-specific dataset and fine-tuned object detector.
* Increase dataset diversity across lighting conditions, poses, ages, and backgrounds.
* Evaluate performance on an independent real-world dataset.
* Add prediction history and downloadable reports.
* Deploy the application to a cloud hosting platform.
* Explore model optimization for faster inference.

## Project Status

**Current status:** Functional prototype completed for five cattle breeds.

The application integrates cattle detection, breed classification, confidence scoring, top-three predictions, and Grad-CAM visualization.

## License

Add an appropriate license before publicly distributing this project. Also verify the usage and redistribution terms for the dataset, pretrained models, and other third-party components.
