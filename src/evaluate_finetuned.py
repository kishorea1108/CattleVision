import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_DIR = BASE_DIR / "dataset" / "splits" / "test"
MODEL_PATH = BASE_DIR / "models" / "mobilenetv2_finetuned_best.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"

# Load class names
with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
    saved_classes = json.load(file)

# Load test data
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(224, 224),
    batch_size=16,
    label_mode="int",
    shuffle=False,
)

if test_ds.class_names != saved_classes:
    raise ValueError(
        f"Class order mismatch: {test_ds.class_names}"
    )

# Load the best fine-tuned model
model = tf.keras.models.load_model(MODEL_PATH)

# Generate predictions
y_true = np.concatenate([
    labels.numpy() for _, labels in test_ds
])

probabilities = model.predict(test_ds, verbose=1)
y_pred = np.argmax(probabilities, axis=1)

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)

print("\n" + "=" * 45)
print(f"Fine-tuned Test Accuracy: {accuracy * 100:.2f}%")
print("=" * 45)

print("\nClassification Report:\n")
print(
    classification_report(
        y_true,
        y_pred,
        labels=list(range(len(saved_classes))),
        target_names=saved_classes,
        digits=4,
        zero_division=0,
    )
)

print("\nConfusion Matrix:\n")
print(
    confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(saved_classes))),
    )
)