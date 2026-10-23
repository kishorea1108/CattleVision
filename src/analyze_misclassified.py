
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw

# ==========================================
# 1. Configuration
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent

TEST_DIR = BASE_DIR / "dataset" / "splits" / "test"
MODEL_PATH = BASE_DIR / "models" / "mobilenetv2_finetuned_best.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"

OUTPUT_DIR = BASE_DIR / "models" / "misclassified_report"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

# ==========================================
# 2. Load class names
# ==========================================
with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)

# ==========================================
# 3. Load test dataset
# ==========================================
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

if test_ds.class_names != class_names:
    raise ValueError(
        f"Class order mismatch!\n"
        f"Saved: {class_names}\n"
        f"Test: {test_ds.class_names}"
    )

# ==========================================
# 4. Load model
# ==========================================
model = tf.keras.models.load_model(MODEL_PATH)

# ==========================================
# 5. Predict test images
# ==========================================
y_true = np.concatenate([
    labels.numpy() for _, labels in test_ds
])

probabilities = model.predict(test_ds, verbose=1)
y_pred = np.argmax(probabilities, axis=1)

# ==========================================
# 6. Collect test image paths
# ==========================================
image_paths = []

for class_name in class_names:
    class_dir = TEST_DIR / class_name

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    files = sorted(
        path for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )

    image_paths.extend(files)

if len(image_paths) != len(y_true):
    raise ValueError(
        f"Image count mismatch: {len(image_paths)} paths "
        f"but {len(y_true)} labels. Check dataset ordering."
    )

# ==========================================
# 7. Identify incorrect predictions
# ==========================================
misclassified = []

for index, (actual, predicted) in enumerate(zip(y_true, y_pred)):
    if actual != predicted:
        confidence = float(probabilities[index][predicted])

        misclassified.append({
            "image_path": str(image_paths[index].relative_to(BASE_DIR)),
            "actual_breed": class_names[int(actual)],
            "predicted_breed": class_names[int(predicted)],
            "predicted_confidence": confidence,
            "actual_class_probability": float(
                probabilities[index][actual]
            ),
        })

# ==========================================
# 8. Save CSV report
# ==========================================
csv_path = OUTPUT_DIR / "misclassified_images.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as file:
    fieldnames = [
        "image_path",
        "actual_breed",
        "predicted_breed",
        "predicted_confidence",
        "actual_class_probability",
    ]

    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(misclassified)

# ==========================================
# 9. Create visual contact sheet
# ==========================================
if misclassified:
    columns = 4
    rows = (len(misclassified) + columns - 1) // columns

    cell_width = 300
    image_height = 200
    text_height = 100

    sheet = Image.new(
        "RGB",
        (columns * cell_width, rows * (image_height + text_height)),
        "white",
    )

    draw = ImageDraw.Draw(sheet)

    for i, item in enumerate(misclassified):
        image_path = BASE_DIR / item["image_path"]

        with Image.open(image_path) as original:
            image = original.convert("RGB")
            image.thumbnail((cell_width - 10, image_height - 10))

        x = (i % columns) * cell_width
        y = (i // columns) * (image_height + text_height)

        image_x = x + (cell_width - image.width) // 2
        sheet.paste(image, (image_x, y))

        text = (
            f"Actual: {item['actual_breed']}\n"
            f"Predicted: {item['predicted_breed']}\n"
            f"Confidence: "
            f"{item['predicted_confidence']:.2%}"
        )

        draw.multiline_text(
            (x + 5, y + image_height),
            text,
            fill="black",
            spacing=5,
        )

    sheet_path = OUTPUT_DIR / "misclassified_contact_sheet.png"
    sheet.save(sheet_path)

    print("\nContact sheet saved:", sheet_path)
else:
    print("\nNo misclassified images found.")

# ==========================================
# 10. Summary
# ==========================================
print("\n" + "=" * 50)
print("MISCLASSIFICATION ANALYSIS")
print("=" * 50)

print("Total test images:", len(y_true))
print("Incorrect predictions:", len(misclassified))
print(
    "Test accuracy:",
    f"{(len(y_true) - len(misclassified)) / len(y_true) * 100:.2f}%"
)

print("\nMisclassifications by actual breed:")

for breed in class_names:
    count = sum(
        item["actual_breed"] == breed
        for item in misclassified
    )
    print(f"{breed}: {count}")

print("\nReports saved:")
print(csv_path)