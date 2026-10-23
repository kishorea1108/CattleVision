from ultralytics import YOLO
from pathlib import Path

# YOLO will download pretrained weights on first use.
MODEL_NAME = "yolo11n.pt"

# COCO class ID 19 corresponds to cow.
COW_CLASS_ID = 19

_detector = None


def load_detector():
    """Load the pretrained YOLO object detector."""
    global _detector

    if _detector is None:
        _detector = YOLO(MODEL_NAME)

    return _detector


def detect_cattle(image_path, confidence_threshold=0.35):
    """
    Detect cows in an image.

    Returns:
        {
            "cattle_detected": bool,
            "detections": list
        }
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    detector = load_detector()

    results = detector.predict(
        source=str(image_path),
        conf=confidence_threshold,
        classes=[COW_CLASS_ID],
        verbose=False,
    )

    detections = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            detections.append({
                "confidence": float(box.conf[0].item()),
                "bounding_box": [
                    float(value)
                    for value in box.xyxy[0].tolist()
                ],
            })

    return {
        "cattle_detected": len(detections) > 0,
        "detections": detections,
    }