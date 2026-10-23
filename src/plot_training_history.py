import json
from pathlib import Path

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
HISTORY_PATH = BASE_DIR / "models" / "training_history.json"
OUTPUT_DIR = BASE_DIR / "models"

with open(HISTORY_PATH, "r", encoding="utf-8") as file:
    history = json.load(file)

epochs = range(1, len(history["accuracy"]) + 1)

# Accuracy plot
plt.figure(figsize=(9, 5))
plt.plot(epochs, history["accuracy"], label="Training Accuracy")
plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MobileNetV2: Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "accuracy_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Loss plot
plt.figure(figsize=(9, 5))
plt.plot(epochs, history["loss"], label="Training Loss")
plt.plot(epochs, history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MobileNetV2: Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "loss_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Saved accuracy curve:", OUTPUT_DIR / "accuracy_curve.png")
print("Saved loss curve:", OUTPUT_DIR / "loss_curve.png")