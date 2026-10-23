
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf

from src.cattle_detector import detect_cattle

from PIL import Image, ImageOps, UnidentifiedImageError


# =========================================================
# 1. PROJECT CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

MODEL_PATH = (
    BASE_DIR / "models" / "mobilenetv2_finetuned_best.keras"
)

CLASS_NAMES_PATH = (
    BASE_DIR / "models" / "class_names.json"
)

IMAGE_SIZE = (224, 224)
DETECTION_CONFIDENCE = 0.35

# Make modules inside src importable.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cattle_detector import detect_cattle
from gradcam import (
    generate_gradcam,
    create_gradcam_overlay,
)


# =========================================================
# 2. STREAMLIT PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CattleVision | Breed Recognition",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 3. CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f7fb;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    .hero {
        background: linear-gradient(120deg, #12372a, #246b50);
        padding: 2rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        color: white;
        font-size: 2.35rem;
        margin-bottom: 0.5rem;
    }

    .hero p {
        color: #e1f3e8;
        font-size: 1.05rem;
        margin-bottom: 0;
    }

    .result-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem 0 1rem 0;
    }

    .result-label {
        color: #64748b;
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
    }

    .breed-name {
        color: #166534;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .section-heading {
        color: #173b2d;
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
    }

    .footer {
        color: #64748b;
        text-align: center;
        font-size: 0.85rem;
        padding-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 4. LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class names file not found: {CLASS_NAMES_PATH}"
        )

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# =========================================================
# 5. APPLICATION HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <h1>🐄 CattleVision</h1>
        <p>
            AI-powered cattle breed recognition using
            YOLO detection, MobileNetV2 classification,
            and Grad-CAM explainability.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 6. SIDEBAR
# =========================================================

with st.sidebar:
    st.title("Project Information")

    st.markdown("### AI Components")
    st.write("• YOLO — cattle detection")
    st.write("• MobileNetV2 — breed classification")
    st.write("• Grad-CAM — visual explanation")

    st.divider()

    st.markdown("### Supported Breeds")

    st.write("1. Ayrshire cattle")
    st.write("2. Brown Swiss cattle")
    st.write("3. Holstein Friesian cattle")
    st.write("4. Jersey cattle")
    st.write("5. Red Dane cattle")

    st.divider()

    st.markdown("### How to Use")

    st.write("1. Upload a cattle photograph.")
    st.write("2. Wait for the detection stage.")
    st.write("3. Review the predicted breed.")
    st.write("4. Compare the top-three predictions.")
    st.write("5. Inspect the Grad-CAM visualization.")

    st.warning(
        "The classifier supports five cattle breeds only. "
        "Buffalo breed recognition is not implemented. "
        "YOLO detection does not guarantee that every "
        "non-cattle image will be rejected."
    )


# =========================================================
# 7. INITIALIZE MODELS
# =========================================================

try:
    model = load_model()
    class_names = load_class_names()

    if model.output_shape[-1] != len(class_names):
        st.error(
            "The number of model outputs does not match "
            "the number of breed labels."
        )
        st.stop()

except Exception as error:
    st.error(
        f"Unable to initialize the application: {error}"
    )
    st.stop()


# =========================================================
# 8. IMAGE UPLOAD INTERFACE
# =========================================================

left_column, right_column = st.columns(
    [1, 1.15],
    gap="large",
)

with left_column:

    st.markdown(
        '<div class="section-heading">Upload an Image</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Upload a clear cattle photograph. "
        "An image containing a single animal is preferable."
    )

    uploaded_file = st.file_uploader(
        "Select an image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supported formats: JPG, JPEG, PNG, and WEBP.",
    )

    st.caption(
        "Use a clear photograph similar to the images "
        "used to train the classification model."
    )


# =========================================================
# 9. PROCESS UPLOADED IMAGE
# =========================================================

if uploaded_file is not None:

    try:
        # -----------------------------------------------
        # 9.1 Read and validate the uploaded image
        # -----------------------------------------------

        image = Image.open(uploaded_file)

        image = ImageOps.exif_transpose(image).convert("RGB")

        if image.width < 32 or image.height < 32:
            st.error(
                "The uploaded image is too small. "
                "Please upload a larger image."
            )
            st.stop()

        # Display image preview.
        with left_column:

            st.markdown("#### Image Preview")

            st.image(
                image,
                use_container_width=True,
            )

            st.caption(
                f"Image dimensions: "
                f"{image.width} × {image.height}"
            )

        # -----------------------------------------------
        # 9.2 Prepare the image for MobileNetV2
        # -----------------------------------------------

        resized_image = image.resize(IMAGE_SIZE)

        image_array = np.asarray(
            resized_image,
            dtype=np.float32,
        )

        image_array = np.expand_dims(
            image_array,
            axis=0,
        )

        # The trained model already contains its
        # Rescaling layer. Do not normalize twice.

        # -----------------------------------------------
        # 9.3 Detect cattle using YOLO
        # -----------------------------------------------

        st.markdown(
            '<div class="section-heading">'
            'Cattle Detection</div>',
            unsafe_allow_html=True,
        )

        detection_result = None

        with st.spinner("Checking whether a cow is present..."):

            # YOLO detector accepts an image file path.
            # Save a temporary JPEG and remove it afterward.
            with tempfile.NamedTemporaryFile(
                suffix=".jpg",
                delete=False,
            ) as temp_file:

                temp_path = Path(temp_file.name)

            try:
                image.save(
                    temp_path,
                    format="JPEG",
                )

                detection_result = detect_cattle(
                    temp_path,
                    confidence_threshold=DETECTION_CONFIDENCE,
                )

            finally:
                temp_path.unlink(missing_ok=True)

        # -----------------------------------------------
        # 9.4 Reject images when no cow is detected
        # -----------------------------------------------

        if not detection_result["cattle_detected"]:

            st.error(
                "🐄 No cow detected in this image. "
                "Please upload a clear cattle photograph."
            )

            st.info(
                "The breed classifier was not run because "
                "the detection stage did not find a cow."
            )

            st.stop()

        st.success("Cow detected successfully.")

        st.caption(
            f"Detected cow instances: "
            f"{len(detection_result['detections'])}"
        )

        # -----------------------------------------------
        # 9.5 Predict cattle breed
        # -----------------------------------------------

        st.markdown(
            '<div class="section-heading">'
            'Breed Classification</div>',
            unsafe_allow_html=True,
        )

        with st.spinner("Predicting cattle breed..."):

            probabilities = model.predict(
                image_array,
                verbose=0,
            )[0]

        top_indices = np.argsort(probabilities)[::-1][:3]

        predicted_index = int(top_indices[0])

        predicted_breed = class_names[predicted_index]

        confidence = float(
            probabilities[predicted_index]
        )

        # -----------------------------------------------
        # 9.6 Display classification results
        # -----------------------------------------------

        with right_column:

            st.markdown(
                '<div class="section-heading">'
                'Prediction Results</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        Predicted cattle breed
                    </div>
                    <div class="breed-name">
                        {predicted_breed}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            metric_col1, metric_col2 = st.columns(2)

            with metric_col1:

                st.metric(
                    "Prediction score",
                    f"{confidence:.2%}",
                )

            with metric_col2:

                st.metric(
                    "Supported breeds",
                    len(class_names),
                )

            st.caption(
                "The score is the model's predicted probability "
                "for the selected class. It is not a guarantee "
                "that the prediction is correct."
            )

            st.markdown("#### Top 3 Predictions")

            for rank, index in enumerate(
                top_indices,
                start=1,
            ):

                breed = class_names[int(index)]

                score = float(probabilities[index])

                st.write(
                    f"**{rank}. {breed}** — {score:.2%}"
                )

                st.progress(
                    min(max(score, 0.0), 1.0)
                )

        # -----------------------------------------------
        # 9.7 Generate Grad-CAM
        # -----------------------------------------------

        st.divider()

        st.markdown(
            '<div class="section-heading">'
            'Explainable AI — Grad-CAM</div>',
            unsafe_allow_html=True,
        )

        st.write(
            "The heatmap highlights image regions that "
            "influence the selected breed prediction."
        )

        try:

            with st.spinner(
                "Generating Grad-CAM visualization..."
            ):

                heatmap, _, _ = generate_gradcam(
                    model,
                    image_array,
                    class_index=predicted_index,
                )

                overlay = create_gradcam_overlay(
                    image,
                    heatmap,
                    alpha=0.4,
                )

            cam_col1, cam_col2 = st.columns(
                2,
                gap="medium",
            )

            with cam_col1:

                st.markdown("#### Original Image")

                st.image(
                    image,
                    use_container_width=True,
                )

            with cam_col2:

                st.markdown("#### Grad-CAM Overlay")

                st.image(
                    overlay,
                    use_container_width=True,
                )

            st.caption(
                "Grad-CAM explains model activation. "
                "It does not prove that the highlighted "
                "regions contain breed-specific features."
            )

        except Exception as error:

            st.warning(
                f"Grad-CAM visualization is unavailable: {error}"
            )

    # =====================================================
    # 10. ERROR HANDLING
    # =====================================================

    except (UnidentifiedImageError, OSError):

        st.error(
            "The uploaded file could not be read as an image. "
            "Please select a valid image file."
        )

    except Exception as error:

        st.error(
            f"An unexpected error occurred: {error}"
        )

else:

    # =====================================================
    # 11. EMPTY STATE
    # =====================================================

    with right_column:

        st.markdown(
            '<div class="section-heading">'
            'Ready to Analyze</div>',
            unsafe_allow_html=True,
        )

        st.info(
            "Upload a cattle image to view the predicted breed, "
            "confidence score, top-three predictions, "
            "and Grad-CAM visualization."
        )

        st.markdown("#### Application Features")

        st.write("✓ Cattle detection using YOLO")
        st.write("✓ Five-class breed classification")
        st.write("✓ Confidence scores and alternative predictions")
        st.write("✓ Grad-CAM visualization")
        st.write("✓ Invalid-image error handling")


# =========================================================
# 12. FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        CattleVision | Image-Based Cattle Breed Recognition
        <br>
        Powered by TensorFlow, MobileNetV2, YOLO, Streamlit,
        and Grad-CAM
    </div>
    """,
    unsafe_allow_html=True,
)
