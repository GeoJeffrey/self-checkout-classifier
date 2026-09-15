"""
Convert a single YOLO-format detection dataset into a classification dataset.

Expected input structure (standard Roboflow YOLO export):

    SRC_DIR/
        train/
            images/  *.jpg
            labels/  *.txt   (class_id x_center y_center width height, normalized)
        valid/
            images/
            labels/
        test/            (optional)
            images/
            labels/
        data.yaml        (contains class names)

What this script does:
    - Reads class names from data.yaml
    - For every image, reads its YOLO label file
    - Crops out each bounding box
    - Saves the crop into classification-style folders:

        OUT_DIR/
            train/
                class_a/ crop1.jpg crop2.jpg ...
                class_b/ ...
            val/
                class_a/ ...
                class_b/ ...

This output structure is what torchvision.datasets.ImageFolder expects,
and is what 02_eda.py / 03_train_classifier.py consume.
"""

import os
import shutil
import yaml
import cv2

# ----------------------- CONFIG (edit this) -----------------------
CONFIG = {
    # Root of your single YOLO dataset (the folder that contains
    # train/, valid/ (or val/), and data.yaml)
    "SRC_DIR": "path/to/your/yolo_dataset",

    # Where to write the classification-style dataset
    "OUT_DIR": "classification_dataset",

    # Only keep these class names (must match names in data.yaml exactly).
    # Set to None to keep ALL classes found in data.yaml.
    "KEEP_CLASSES": None,   # e.g. ["apple", "milk", "bread", "soap", "noodles"]

    # Minimum crop size (pixels) to keep — filters out tiny/degenerate boxes
    "MIN_CROP_SIZE": 10,

    # Padding added around each box before cropping (fraction of box size)
    "BOX_PADDING": 0.05,
}
# --------------------------------------------------------------------


def load_class_names(src_dir):
    yaml_path = os.path.join(src_dir, "data.yaml")
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"data.yaml not found at {yaml_path}")
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    names = data["names"]
    # names can be a list or a dict {id: name}
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names.keys())]
    return names


def find_split_dirs(src_dir):
    """
    Detect which split folders exist (train / valid / val / test) and
    map them to output split names ('train' or 'val').
    """
    candidates = {
        "train": ["train"],
        "val": ["valid", "val", "validation"],
    }
    found = {}
    for out_name, options in candidates.items():
        for opt in options:
            p = os.path.join(src_dir, opt)
            if os.path.isdir(os.path.join(p, "images")):
                found[out_name] = p
                break
    if "train" not in found:
        raise FileNotFoundError(
            f"Could not find a train/images folder under {src_dir}"
        )
    return found


def yolo_to_pixel_box(x_c, y_c, w, h, img_w, img_h, padding):
    box_w = w * img_w
    box_h = h * img_h
    box_w *= (1 + padding)
    box_h *= (1 + padding)

    cx = x_c * img_w
    cy = y_c * img_h

    x1 = int(max(0, cx - box_w / 2))
    y1 = int(max(0, cy - box_h / 2))
    x2 = int(min(img_w, cx + box_w / 2))
    y2 = int(min(img_h, cy + box_h / 2))
    return x1, y1, x2, y2


def process_split(split_dir, out_split_dir, class_names, keep_classes, min_size, padding):
    images_dir = os.path.join(split_dir, "images")
    labels_dir = os.path.join(split_dir, "labels")

    image_files = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    os.makedirs(out_split_dir, exist_ok=True)
    counts = {}
    skipped_no_label = 0
    skipped_bad_box = 0

    for img_name in image_files:
        img_path = os.path.join(images_dir, img_name)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_name)

        if not os.path.isfile(label_path):
            skipped_no_label += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            skipped_bad_box += 1
            continue
        img_h, img_w = img.shape[:2]

        with open(label_path, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_id, x_c, y_c, w, h = parts
            cls_id = int(float(cls_id))
            x_c, y_c, w, h = map(float, (x_c, y_c, w, h))

            if cls_id < 0 or cls_id >= len(class_names):
                continue
            cls_name = class_names[cls_id]

            if keep_classes is not None and cls_name not in keep_classes:
                continue

            x1, y1, x2, y2 = yolo_to_pixel_box(x_c, y_c, w, h, img_w, img_h, padding)
            if (x2 - x1) < min_size or (y2 - y1) < min_size:
                skipped_bad_box += 1
                continue

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                skipped_bad_box += 1
                continue

            cls_out_dir = os.path.join(out_split_dir, cls_name)
            os.makedirs(cls_out_dir, exist_ok=True)

            crop_name = f"{os.path.splitext(img_name)[0]}_{i}.jpg"
            cv2.imwrite(os.path.join(cls_out_dir, crop_name), crop)

            counts[cls_name] = counts.get(cls_name, 0) + 1

    return counts, skipped_no_label, skipped_bad_box


def main():
    src_dir = CONFIG["SRC_DIR"]
    out_dir = CONFIG["OUT_DIR"]
    keep_classes = CONFIG["KEEP_CLASSES"]
    min_size = CONFIG["MIN_CROP_SIZE"]
    padding = CONFIG["BOX_PADDING"]

    if not os.path.isdir(src_dir):
        raise FileNotFoundError(
            f"SRC_DIR '{src_dir}' does not exist. Edit CONFIG['SRC_DIR'] at the top of this script."
        )

    class_names = load_class_names(src_dir)
    print(f"Classes found in data.yaml: {class_names}")

    if keep_classes is not None:
        missing = [c for c in keep_classes if c not in class_names]
        if missing:
            raise ValueError(f"KEEP_CLASSES contains names not in data.yaml: {missing}")
        print(f"Restricting to: {keep_classes}")

    if os.path.isdir(out_dir):
        print(f"Output dir '{out_dir}' already exists — removing and recreating.")
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    splits = find_split_dirs(src_dir)
    print(f"Detected splits: {list(splits.keys())}")

    all_class_names_written = set()
    for out_split_name, split_path in splits.items():
        out_split_dir = os.path.join(out_dir, out_split_name)
        counts, no_label, bad_box = process_split(
            split_path, out_split_dir, class_names, keep_classes, min_size, padding
        )
        all_class_names_written.update(counts.keys())

        print(f"\n[{out_split_name}] from {split_path}")
        for cls, n in sorted(counts.items()):
            print(f"  {cls}: {n} crops")
        print(f"  skipped (no label file): {no_label}")
        print(f"  skipped (bad/tiny box): {bad_box}")

    # If there was no separate val split, carve one out of train
    if "val" not in splits:
        print("\nNo val/valid split found — creating one from 15% of train images.")
        make_val_split_from_train(out_dir, val_fraction=0.15)

    # Write class_names.txt for downstream scripts (EDA, training, app)
    final_classes = sorted(all_class_names_written)
    with open(os.path.join(out_dir, "class_names.txt"), "w") as f:
        for c in final_classes:
            f.write(c + "\n")

    print(f"\nDone. Classification dataset written to: {out_dir}")
    print(f"Final classes ({len(final_classes)}): {final_classes}")


def make_val_split_from_train(out_dir, val_fraction=0.15):
    import random
    random.seed(42)

    train_dir = os.path.join(out_dir, "train")
    val_dir = os.path.join(out_dir, "val")
    os.makedirs(val_dir, exist_ok=True)

    for cls_name in os.listdir(train_dir):
        cls_train_dir = os.path.join(train_dir, cls_name)
        if not os.path.isdir(cls_train_dir):
            continue
        files = os.listdir(cls_train_dir)
        random.shuffle(files)
        n_val = max(1, int(len(files) * val_fraction))
        val_files = files[:n_val]

        cls_val_dir = os.path.join(val_dir, cls_name)
        os.makedirs(cls_val_dir, exist_ok=True)
        for fname in val_files:
            shutil.move(
                os.path.join(cls_train_dir, fname),
                os.path.join(cls_val_dir, fname),
            )
        print(f"  moved {n_val}/{len(files)} images to val/{cls_name}")


if __name__ == "__main__":
    main()
