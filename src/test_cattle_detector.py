from cattle_detector import detect_cattle

image_path = input(
    "Enter the image path: "
).strip().strip('"')

try:
    result = detect_cattle(image_path)

    if result["cattle_detected"]:
        print("\nCattle detected!")

        for index, detection in enumerate(
            result["detections"],
            start=1,
        ):
            print(f"\nDetection {index}")
            print(
                "Detection confidence:",
                f"{detection['confidence']:.2%}",
            )
            print(
                "Bounding box:",
                detection["bounding_box"],
            )

    else:
        print("\nNo cattle detected.")
        print("Please upload a clear cattle photograph.")

except Exception as error:
    print("Detection failed:", error)