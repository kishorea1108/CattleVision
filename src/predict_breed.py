
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


# ==========================================
# 1. Project configuration
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "mobilenetv2_finetuned_best.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"

IMAGE_SIZE = (224, 224)


# ==========================================
# 2. Load model and class names
# ==========================================

def load_model_and_classes():
    """Load the trained model and breed labels."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class names file not found: {CLASS_NAMES_PATH}"
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    if model.output_shape[-1] != len(class_names):
        raise ValueError(
            "The number of model outputs does not match "
            "the number of breed labels."
        )

    return model, class_names


# ==========================================
# 3. Prepare image
# ==========================================

def preprocess_image(image_path):
    """Load and prepare an image for MobileNetV2."""

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE,
        color_mode="rgb",
    )

    image_array = tf.keras.utils.img_to_array(image)

    # Keep pixel values in [0, 255].
    # The trained model already contains its Rescaling layer.
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# ==========================================
# 4. Predict breed
# ==========================================

def predict_breed(image_path, model, class_names, top_k=3):
    """Return the top breed predictions and confidence scores."""

    image_array = preprocess_image(image_path)

    probabilities = model.predict(
        image_array,
        verbose=0,
    )[0]

    top_k = min(top_k, len(class_names))

    top_indices = np.argsort(probabilities)[::-1][:top_k]

    predictions = []

    for index in top_indices:
        predictions.append({
            "breed": class_names[int(index)],
            "confidence": float(probabilities[index]),
        })

    return predictions


# ==========================================
# 5. Run a sample prediction
# ==========================================

if __name__ == "__main__":

    model, class_names = load_model_and_classes()

    print("Model loaded successfully.")
    print("Supported breeds:", class_names)

    image_path = input(
        "\nEnter the path of a cattle image: "
    ).strip().strip('"')

    try:
        predictions = predict_breed(
            image_path,
            model,
            class_names,
            top_k=3,
        )

        print("\n===== CATTLE BREED PREDICTION =====")

        for rank, result in enumerate(predictions, start=1):
            print(
                f"{rank}. {result['breed']} - "
                f"{result['confidence']:.2%}"
            )

        print("\nPredicted breed:", predictions[0]["breed"])
        print(
            "Prediction confidence:",
            f"{predictions[0]['confidence']:.2%}",
        )

    except (FileNotFoundError, ValueError) as error:
        print("Error:", error)
