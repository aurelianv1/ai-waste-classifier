from pathlib import Path
import random
import shutil

SOURCE_DIR = Path("data/dataset-resized")
OUTPUT_DIR = Path("data/split")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

random.seed(SEED)

classes = [
    folder.name
    for folder in SOURCE_DIR.iterdir()
    if folder.is_dir()
]

for class_name in classes:
    class_dir = SOURCE_DIR / class_name

    images = [
        file
        for file in class_dir.iterdir()
        if file.is_file()
    ]

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    splits = {
        "train": train_images,
        "val": val_images,
        "test": test_images
    }

    for split_name, split_images in splits.items():

        destination = OUTPUT_DIR / split_name / class_name
        destination.mkdir(parents=True, exist_ok=True)

        for image in split_images:
            shutil.copy2(image, destination / image.name)

    print(
        f"{class_name}: "
        f"train={len(train_images)}, "
        f"val={len(val_images)}, "
        f"test={len(test_images)}"
    )