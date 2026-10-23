import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# -----------------------------
# 1. Configuration
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

TEST_DIR = BASE_DIR / "dataset" / "splits" / "test"
MODEL_PATH = BASE_DIR / "models" / "mobilenetv2_best.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"
OUTPUT_DIR = BASE_DIR / "models"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

# -----------------------------
# 2. Check required files
# -----------------------------
if not TEST_DIR.exists():
    raise FileNotFoundError(f"Test dataset not found: {TEST_DIR}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# -----------------------------
# 3. Load class names
# -----------------------------
with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
    saved_class_names = json.load(file)

print("Saved class names:", saved_class_names)

# -----------------------------
# 4. Load test dataset
# -----------------------------
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

class_names = test_ds.class_names

if class_names != saved_class_names:
    raise ValueError(
        f"Class order mismatch!\n"
        f"Saved: {saved_class_names}\n"
        f"Test:  {class_names}"
    )

print("\nTest classes:", class_names)

# -----------------------------
# 5. Load trained model
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)

# -----------------------------
# 6. Generate predictions
# -----------------------------
y_true = np.concatenate([
    labels.numpy() for _, labels in test_ds
])

probabilities = model.predict(test_ds, verbose=1)
y_pred = np.argmax(probabilities, axis=1)

# -----------------------------
# 7. Calculate accuracy
# -----------------------------
accuracy = accuracy_score(y_true, y_pred)

print("\n" + "=" * 50)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print("=" * 50)

# -----------------------------
# 8. Classification report
# -----------------------------
print("\nClassification Report:\n")

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    digits=4,
    zero_division=0,
)

print(report)

report_dict = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)

# -----------------------------
# 9. Confusion matrix
# -----------------------------
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)

fig, ax = plt.subplots(figsize=(10, 8))

display.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45,
    values_format="d",
)

ax.set_title("MobileNetV2 - Test Confusion Matrix")
plt.tight_layout()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.savefig(
    OUTPUT_DIR / "mobilenetv2_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

# -----------------------------
# 10. Save evaluation results
# -----------------------------
results = {
    "model": "MobileNetV2",
    "test_accuracy": float(accuracy),
    "class_names": class_names,
    "classification_report": report_dict,
    "confusion_matrix": cm.tolist(),
    "test_images": int(len(y_true)),
}

with open(
    OUTPUT_DIR / "mobilenetv2_evaluation.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(results, file, indent=4)

print("\nEvaluation saved to:")
print(OUTPUT_DIR / "mobilenetv2_evaluation.json")
print(OUTPUT_DIR / "mobilenetv2_confusion_matrix.png")