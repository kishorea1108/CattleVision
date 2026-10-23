
from pathlib import Path
from PIL import Image
import pandas as pd

# Update this path if your dataset is stored elsewhere.
DATASET_DIR = Path("dataset/Cattle Breeds")

# Image extensions to inspect
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}


def inspect_dataset():
    if not DATASET_DIR.exists():
        print(f"Dataset folder not found: {DATASET_DIR.resolve()}")
        return

    records = []

    breed_folders = sorted(
        folder
        for folder in DATASET_DIR.iterdir()
        if folder.is_dir()
    )

    if not breed_folders:
        print("No breed folders found.")
        return

    for breed_folder in breed_folders:
        image_paths = sorted(
            path
            for path in breed_folder.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        valid_count = 0
        invalid_count = 0
        total_width = 0
        total_height = 0

        for image_path in image_paths:
            try:
                with Image.open(image_path) as image:
                    image.verify()

                # Reopen after verify() before reading dimensions.
                with Image.open(image_path) as image:
                    width, height = image.size

                valid_count += 1
                total_width += width
                total_height += height

            except Exception as error:
                invalid_count += 1
                print(f"Invalid image: {image_path} | {error}")

        average_width = (
            round(total_width / valid_count, 2)
            if valid_count else 0
        )

        average_height = (
            round(total_height / valid_count, 2)
            if valid_count else 0
        )

        records.append({
            "Breed": breed_folder.name,
            "Total Images": len(image_paths),
            "Valid Images": valid_count,
            "Invalid Images": invalid_count,
            "Average Width": average_width,
            "Average Height": average_height
        })

    report = pd.DataFrame(records)

    print("\nDATASET INSPECTION REPORT")
    print("=" * 70)
    print(report.to_string(index=False))

    print("\nTotal valid images:", report["Valid Images"].sum())
    print("Total invalid images:", report["Invalid Images"].sum())
    print("Number of breed classes:", len(report))

    report.to_csv("dataset_report.csv", index=False)
    print("\nReport saved to dataset_report.csv")


if __name__ == "__main__":
    inspect_dataset()

