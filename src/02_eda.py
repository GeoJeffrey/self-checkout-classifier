"""
Exploratory Data Analysis on the classification dataset produced by
01_yolo_to_classification.py.

Produces:
    - class_distribution.png   (bar chart of images per class, train vs val)
    - sample_grid.png          (grid of sample images, a few per class)
    - Console summary stats

Run this after 01_yolo_to_classification.py.
"""

import os
import random
import math
import matplotlib.pyplot as plt
from PIL import Image

# ----------------------- CONFIG (edit if needed) -----------------------
CONFIG = {
    "DATA_DIR": "classification_dataset",   # output of script 1
    "SAMPLES_PER_CLASS": 3,                 # images per class in the grid
}
# ------------------------------------------------------------------------


def count_images_per_class(split_dir):
    counts = {}
    if not os.path.isdir(split_dir):
        return counts
    for cls_name in sorted(os.listdir(split_dir)):
        cls_dir = os.path.join(split_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        n = len([
            f for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        counts[cls_name] = n
    return counts


def plot_class_distribution(train_counts, val_counts, out_path):
    classes = sorted(set(train_counts) | set(val_counts))
    train_vals = [train_counts.get(c, 0) for c in classes]
    val_vals = [val_counts.get(c, 0) for c in classes]

    x = range(len(classes))
    width = 0.35

    plt.figure(figsize=(max(8, len(classes) * 0.9), 5))
    plt.bar([i - width / 2 for i in x], train_vals, width, label="train")
    plt.bar([i + width / 2 for i in x], val_vals, width, label="val")
    plt.xticks(list(x), classes, rotation=45, ha="right")
    plt.ylabel("Number of images (crops)")
    plt.title("Class distribution: train vs val")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_sample_grid(train_dir, classes, samples_per_class, out_path):
    n_classes = len(classes)
    fig, axes = plt.subplots(
        n_classes, samples_per_class,
        figsize=(samples_per_class * 2.2, n_classes * 2.2)
    )

    # normalize axes shape when there's only one row/col
    if n_classes == 1:
        axes = [axes]
    if samples_per_class == 1:
        axes = [[ax] for ax in axes]

    for row, cls_name in enumerate(classes):
        cls_dir = os.path.join(train_dir, cls_name)
        files = [
            f for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ] if os.path.isdir(cls_dir) else []
        random.shuffle(files)
        chosen = files[:samples_per_class]

        for col in range(samples_per_class):
            ax = axes[row][col]
            ax.axis("off")
            if col < len(chosen):
                img_path = os.path.join(cls_dir, chosen[col])
                try:
                    img = Image.open(img_path).convert("RGB")
                    ax.imshow(img)
                except Exception:
                    pass
            if col == 0:
                ax.set_title(cls_name, fontsize=9, loc="left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def main():
    random.seed(42)
    data_dir = CONFIG["DATA_DIR"]
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"'{train_dir}' not found. Run 01_yolo_to_classification.py first."
        )

    train_counts = count_images_per_class(train_dir)
    val_counts = count_images_per_class(val_dir)

    print("=== Class distribution ===")
    classes = sorted(set(train_counts) | set(val_counts))
    total_train, total_val = 0, 0
    for cls in classes:
        t, v = train_counts.get(cls, 0), val_counts.get(cls, 0)
        total_train += t
        total_val += v
        print(f"  {cls:20s} train={t:5d}  val={v:5d}")
    print(f"\nTotal: train={total_train}, val={total_val}, classes={len(classes)}")

    # Flag class imbalance
    if classes:
        train_vals = [train_counts.get(c, 0) for c in classes]
        if max(train_vals) > 0:
            ratio = max(train_vals) / max(1, min(v for v in train_vals if v > 0))
            print(f"Max/min class imbalance ratio (train): {ratio:.2f}x")

    plot_class_distribution(
        train_counts, val_counts,
        os.path.join(data_dir, "class_distribution.png")
    )
    plot_sample_grid(
        train_dir, classes, CONFIG["SAMPLES_PER_CLASS"],
        os.path.join(data_dir, "sample_grid.png")
    )


if __name__ == "__main__":
    main()
