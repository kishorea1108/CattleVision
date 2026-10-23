
from pathlib import Path
import json

import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# -----------------------------
# Configuration
# -----------------------------

DATA_DIR = Path("dataset/splits")
MODEL_DIR = Path("models")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15
SEED = 42

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "validation"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

tf.keras.utils.set_random_seed(SEED)


# -----------------------------
# Verify dataset directories
# -----------------------------

for folder in [TRAIN_DIR, VAL_DIR]:
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {folder.resolve()}"
        )


# -----------------------------
# Load datasets
# -----------------------------

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

class_names = train_ds.class_names

if len(class_names) != 5:
    raise ValueError(
        f"Expected 5 breeds, found {len(class_names)}: {class_names}"
    )

if val_ds.class_names != class_names:
    raise ValueError(
        "Training and validation class names do not match."
    )

print("Breed classes:", class_names)

# Optimize input loading without caching the full dataset in RAM.
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)


# -----------------------------
# Build MobileNetV2 model
# -----------------------------

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze pretrained feature-extraction layers.
base_model.trainable = False

inputs = layers.Input(shape=(224, 224, 3))

# MobileNetV2 expects its inputs to be preprocessed to [-1, 1].
x = layers.Rescaling(
    scale=1.0 / 127.5,
    offset=-1
)(inputs)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.25)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# -----------------------------
# Callbacks
# -----------------------------

best_model_path = MODEL_DIR / "mobilenetv2_best.keras"

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]


# -----------------------------
# Train
# -----------------------------

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


# -----------------------------
# Save final artifacts
# -----------------------------

model.save(MODEL_DIR / "mobilenetv2_final.keras")

with open(MODEL_DIR / "class_names.json", "w") as file:
    json.dump(class_names, file, indent=4)

history_data = {
    key: [float(value) for value in values]
    for key, values in history.history.items()
}

with open(MODEL_DIR / "training_history.json", "w") as file:
    json.dump(history_data, file, indent=4)


# -----------------------------
# Plot training history
# -----------------------------

plt.figure(figsize=(10, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MobileNetV2 Training and Validation Accuracy")
plt.legend()
plt.tight_layout()

plt.savefig(
    MODEL_DIR / "training_accuracy.png",
    dpi=150
)

plt.close()

print("\nTraining completed.")
print("Best model:", best_model_path)
print("Final model:", MODEL_DIR / "mobilenetv2_final.keras")
print("Class names:", class_names)
print("Training history saved.")

