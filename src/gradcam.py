
import numpy as np
import tensorflow as tf
import cv2


def generate_gradcam(model, image_array, class_index=None):
    """
    Generate a Grad-CAM heatmap for the selected class.

    image_array:
        NumPy array with shape (1, 224, 224, 3).
        Pixel values must be in the range [0, 255].

    Returns:
        heatmap, predicted_index, probabilities
    """

    # Find the MobileNetV2 backbone.
    backbone = None

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            if "mobilenetv2" in layer.name.lower():
                backbone = layer
                break

    if backbone is None:
        raise ValueError(
            "MobileNetV2 backbone not found. "
            "Check model.summary() and model layer names."
        )

    # Find the final convolutional layer.
    last_conv_layer = None

    for layer in reversed(backbone.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        raise ValueError(
            "No Conv2D layer found in MobileNetV2."
        )

    # Build a feature extractor for the backbone.
    feature_extractor = tf.keras.Model(
        inputs=backbone.inputs,
        outputs=last_conv_layer.output,
        name="gradcam_feature_extractor",
    )

    image_tensor = tf.convert_to_tensor(
        image_array,
        dtype=tf.float32,
    )

    # Run the image through the original model graph.
    # The GradientTape records operations needed for gradients.
    with tf.GradientTape() as tape:

        # Apply the original preprocessing layer.
        preprocessing_layer = model.get_layer("rescaling")
        preprocessed = preprocessing_layer(image_tensor)

        # Obtain the backbone's input tensor.
        backbone_input = preprocessed

        # Extract convolutional features.
        conv_outputs = feature_extractor(
            backbone_input,
            training=False,
        )

        # Reproduce the original classifier head.
        x = conv_outputs

        for layer in backbone.layers:
            if layer.name == last_conv_layer.name:
                found = True
            else:
                found = False

        # Use the backbone's remaining layers after the
        # selected convolutional layer.
        passed_conv = False

        for layer in backbone.layers:
            if layer.name == last_conv_layer.name:
                passed_conv = True
                continue

            if not passed_conv:
                continue

            x = layer(x, training=False)

        # Apply the remaining classification layers.
        # This assumes the classifier head consists of
        # global average pooling, dropout, and dense layers.
        for layer in model.layers:
            if layer.name in (
                "rescaling",
                backbone.name,
            ):
                continue

            if isinstance(layer, tf.keras.layers.InputLayer):
                continue

            x = layer(x, training=False)

        predictions = x

        predicted_index = int(
            tf.argmax(predictions[0]).numpy()
        )

        if class_index is None:
            class_index = predicted_index

        target_score = predictions[:, class_index]

    # Calculate gradients.
    gradients = tape.gradient(
        target_score,
        conv_outputs,
    )

    if gradients is None:
        raise RuntimeError(
            "Gradients could not be calculated. "
            "The feature extractor and classifier may not "
            "be connected through the same model graph."
        )

    # Average gradients across spatial dimensions.
    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2),
    )

    # Calculate weighted feature-map combination.
    features = conv_outputs[0]
    weights = weights[0]

    cam = tf.reduce_sum(
        features * weights[tf.newaxis, tf.newaxis, :],
        axis=-1,
    )

    cam = tf.nn.relu(cam)

    # Normalize heatmap.
    max_value = tf.reduce_max(cam)

    heatmap = tf.where(
        max_value > 0,
        cam / tf.maximum(
            max_value,
            tf.keras.backend.epsilon(),
        ),
        tf.zeros_like(cam),
    )

    return (
        heatmap.numpy(),
        predicted_index,
        predictions[0].numpy(),
    )


def create_gradcam_overlay(
    original_image,
    heatmap,
    alpha=0.4,
):
    """
    Overlay the heatmap on the original RGB image.
    """

    original = np.asarray(
        original_image.convert("RGB")
    )

    height, width = original.shape[:2]

    resized_heatmap = cv2.resize(
        heatmap,
        (width, height),
    )

    heatmap_uint8 = np.uint8(
        255 * np.clip(resized_heatmap, 0, 1)
    )

    colored_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET,
    )

    colored_heatmap = cv2.cvtColor(
        colored_heatmap,
        cv2.COLOR_BGR2RGB,
    )

    overlay = cv2.addWeighted(
        original,
        1 - alpha,
        colored_heatmap,
        alpha,
        0,
    )

    return overlay
