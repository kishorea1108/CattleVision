
from pathlib import Path
import random
import shutil

from sklearn.model_selection import train_test_split


# Dataset containing the original breed folders
SOURCE_DIR = Path("dataset/Cattle Breeds")

# Destination for the split datasets
OUTPUT_DIR = Path("dataset/splits")

# Reproducibility
RANDOM_STATE = 42

# Image extensions
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}


def split_dataset():

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {SOURCE_DIR.resolve()}"
        )

    breed_folders = sorted(
        folder
        for folder in SOURCE_DIR.iterdir()
        if folder.is_dir()
    )

    if len(breed_folders) != 5:
        raise ValueError(
            f"Expected 5 breed folders, found {len(breed_folders)}."
        )

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"{OUTPUT_DIR} already contains files. "
            "Inspect or back up the existing splits before rerunning."
        )

    random.seed(RANDOM_STATE)

    for breed_folder in breed_folders:

        images = sorted(
            path
            for path in breed_folder.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        if len(images) < 3:
            raise ValueError(
                f"Not enough images in {breed_folder.name}"
            )

        # 70% training, 30% temporary
        train_images, remaining_images = train_test_split(
            images,
            test_size=0.30,
            random_state=RANDOM_STATE,
            shuffle=True
        )

        # Split the remaining 30% equally:
        # 15% validation, 15% testing
        validation_images, test_images = train_test_split(
            remaining_images,
            test_size=0.50,
            random_state=RANDOM_STATE,
            shuffle=True
        )

        splits = {
            "train": train_images,
            "validation": validation_images,
            "test": test_images
        }

        for split_name, image_paths in splits.items():

            destination = OUTPUT_DIR / split_name / breed_folder.name
            destination.mkdir(parents=True, exist_ok=True)

            for image_path in image_paths:
                shutil.copy2(
                    image_path,
                    destination / image_path.name
                )

        print(f"\nBreed: {breed_folder.name}")
        print(f"Training:   {len(train_images)}")
        print(f"Validation: {len(validation_images)}")
        print(f"Testing:    {len(test_images)}")

    print("\nDataset splitting completed.")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    split_dataset()
