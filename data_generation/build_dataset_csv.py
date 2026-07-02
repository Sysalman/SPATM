"""
build_dataset_csv.py
Scans the generated image folders and builds a CSV dataset file
in the format expected by train_spatm.py:

    profession,attribute,class,filename

Usage:
    python build_dataset_csv.py --base_dir ./dataset/gender_dataset/train
                                --attribute gender
                                --output ./dataset/gender_dataset.txt
"""

import os
import argparse
import glob


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir", type=str, required=True,
        help="Base directory containing profession/class subfolders"
    )
    parser.add_argument(
        "--attribute", type=str, required=True,
        choices=["gender", "race", "age"]
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output CSV path, e.g. ./dataset/gender_dataset.txt"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    rows = []

    # Expected folder structure:
    # base_dir/
    #   construction_worker/
    #     male/
    #       images/
    #         000.jpg ...
    #     female/
    #       images/
    #         000.jpg ...

    for profession_dir in sorted(os.listdir(args.base_dir)):
        profession_path = os.path.join(args.base_dir, profession_dir)
        if not os.path.isdir(profession_path):
            continue

        profession = profession_dir.replace("_", " ")

        for class_dir in sorted(os.listdir(profession_path)):
            class_path = os.path.join(profession_path, class_dir)
            if not os.path.isdir(class_path):
                continue

            attribute_class = class_dir.replace("_", " ")
            images_path = os.path.join(class_path, "images")

            if not os.path.exists(images_path):
                print(f"  [SKIP] No images folder: {images_path}")
                continue

            images = sorted(glob.glob(os.path.join(images_path, "*.jpg")))
            images += sorted(glob.glob(os.path.join(images_path, "*.png")))

            for img_path in images:
                # Use forward slashes for cross-platform compatibility
                img_path_clean = img_path.replace("\\", "/")
                rows.append(
                    f"{profession},{args.attribute},{attribute_class},{img_path_clean}"
                )

            print(f"  {profession} / {attribute_class}: {len(images)} images")

    # Write CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write("profession,attribute,class,filename\n")
        for row in rows:
            f.write(row + "\n")

    print(f"\nDataset CSV written: {args.output}")
    print(f"Total images: {len(rows)}")
