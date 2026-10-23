import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam


# ---------------------------------------
# 1. Configuration
# ---------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "dataset" / "splits" / "train"
VAL_DIR = BASE_DIR / "dataset" / "splits" / "validation"
MODEL_DIR = BASE_DIR / "models"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 12
SEED = 42

INITIAL_MODEL = MODEL_DIR / "mobilenetv2_best.keras"
BEST_MODEL = MODEL_DIR / "mobilenetv2_finetuned_best.keras"
FINAL_MODEL = MODEL_DIR / "mobilenetv2_finetuned_final.keras"


# ---------------------------------------
# 2. Check files
# ---------------------------------------
if not INITIAL_MODEL.exists():
    raise FileNotFoundError(
        f"Initial model not found: {INITIAL_MODEL}"
    )


# ---------------------------------------
# 3. Load datasets
# ---------------------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

if train_ds.class_names != val_ds.class_names:
    raise ValueError("Training and validation class orders differ.")

class_names = train_ds.class_names
num_classes = len(class_names)

print("Classes:", class_names)


# ---------------------------------------
# 4. Optimize data loading
# ---------------------------------------
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)


# ---------------------------------------
# 5. Load existing trained model
# ---------------------------------------
model = tf.keras.models.load_model(INITIAL_MODEL)

print("\nLoaded baseline model.")
print("Output classes:", num_classes)


# ---------------------------------------
# 6. Locate MobileNetV2 backbone
# ---------------------------------------
backbone = None

for layer in model.layers:
    if isinstance(layer, tf.keras.Model):
        if "mobilenetv2" in layer.name.lower():
            backbone = layer
            break

if backbone is None:
    raise ValueError(
        "Could not locate the MobileNetV2 backbone. "
        "Inspect model.summary() before continuing."
    )

print("Backbone:", backbone.name)


# ---------------------------------------
# 7. Freeze most layers
# ---------------------------------------
backbone.trainable = True

# Fine-tune only the upper portion of the backbone.
fine_tune_from = int(len(backbone.layers) * 0.70)

for layer in backbone.layers[:fine_tune_from]:
    layer.trainable = False

for layer in backbone.layers[fine_tune_from:]:
    # Batch Normalization is kept frozen for stability.
    layer.trainable = not isinstance(
        layer, layers.BatchNormalization
    )

print(
    "Total backbone layers:",
    len(backbone.layers)
)
print(
    "Fine-tuning starts at layer:",
    fine_tune_from
)
print(
    "Trainable backbone layers:",
    sum(layer.trainable for layer in backbone.layers)
)


# ---------------------------------------
# 8. Add data augmentation
# ---------------------------------------
# Augmentation is inserted before the existing
# Rescaling and MobileNetV2 backbone.
augmentation = models.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
    ],
    name="fine_tune_augmentation",
)

# Rebuild the model while retaining the learned weights.
inputs = layers.Input(shape=(224, 224, 3))

x = augmentation(inputs)

# The original model's preprocessing layer converts
# pixels from [0, 255] to [-1, 1].
preprocessing_layer = model.get_layer("rescaling")
x = preprocessing_layer(x)

x = backbone(x)

# Retain the existing classifier weights.
for layer in model.layers:
    if layer.name in {
        "global_average_pooling2d",
        "dropout",
        "dense",
    }:
        x = layer(x)

model_finetuned = models.Model(
    inputs=inputs,
    outputs=x,
    name="mobilenetv2_finetuned",
)


# ---------------------------------------
# 9. Compile with a low learning rate
# ---------------------------------------
model_finetuned.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model_finetuned.summary()


# ---------------------------------------
# 10. Callbacks
# ---------------------------------------
callbacks = [
    ModelCheckpoint(
        filepath=str(BEST_MODEL),
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    ),
]


# ---------------------------------------
# 11. Fine-tune
# ---------------------------------------
history = model_finetuned.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)


# ---------------------------------------
# 12. Save results
# ---------------------------------------
model_finetuned.save(FINAL_MODEL)

with open(
    MODEL_DIR / "mobilenetv2_finetuned_history.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
        file,
        indent=4,
    )

with open(
    MODEL_DIR / "class_names.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(class_names, file, indent=4)

print("\nFine-tuning completed.")
print("Best model:", BEST_MODEL)
print("Final model:", FINAL_MODEL)