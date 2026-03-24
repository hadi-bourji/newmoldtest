# CVAT doesn't create a labels file for images with no annotations, 
# this script is meant to fix that by creating empty text files for every image 
# Works only with YOLO format

import os
import argparse
def generate_empty_labels(base_dir):
    for file in os.listdir(os.path.join(base_dir, "images", "train")):
        file_name = os.path.splitext(file)[0]
        label_path = os.path.join(base_dir, "labels", "train", f"{file_name}.txt")
        if file.lower().endswith(".jpg") and not os.path.exists(label_path):
            print(label_path)
            open(label_path, "w").close()

    for file in os.listdir(os.path.join(base_dir, "images", "validation")):
        file_name = os.path.splitext(file)[0]
        label_path = os.path.join(base_dir, "labels", "validation", f"{file_name}.txt")
        if not os.path.exists(label_path):
            print(label_path)
            open(label_path, "w").close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    generate_empty_labels(args.path)