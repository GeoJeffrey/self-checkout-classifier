"""
05_generate_results.py

Complete results generator for the self-checkout classifier project.
Matches this project layout exactly:

    self_checkout_project/
      classification_dataset/
        train/
        val/
        class_names.txt
      models/
        best_model.pth
        class_names.txt
        price_dict.txt
      results/            <- outputs go here
      src/
        01_yolo_to_classification.py
        02_eda.py
        03_train_classifier_v2.py
        04_pca_eval_compare.py
        05_generate_results.py   <- put THIS file here

Run from the PROJECT ROOT (not from src/):
    cd "D:\\Machine Learning Project\\self_checkout_project\\self_checkout_project"
    python src\\05_generate_results.py

Outputs written to results/:
    class_distribution.png
    sample_grid.png
    confusion_matrix.png
    classification_report.txt
    classification_report.csv
    pca_scatter.png
    pca_svm_comparison.txt
"""

import os
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, ConfusionMatrixDisplay, accuracy_score
)

# ----------------------------- CONFIG -----------------------------
# All paths are relative to the PROJECT ROOT, matching your folder tree.
CONFIG = {
    "DATA_DIR": "classification_dataset",          # has train/ and val/
    "MODEL_PATH": "models/best_model.pth",
    "CLASS_NAMES_PATH": "models/class_names.txt",   # trained-on class list
    "RESULTS_DIR": "results",
    "IMG_SIZE": 224,
    "BATCH_SIZE": 32,
    "SAMPLES_PER_CLASS_GRID": 4,
    "MAX_CLASSES_IN_GRID": 12,
    "SEED": 42,
}
# --------------------------------------------------------------------

random.seed(CONFIG["SEED"])
np.random.seed(CONFIG["SEED"])
torch.manual_seed(CONFIG["SEED"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)


def load_class_names():
    path = CONFIG["CLASS_NAMES_PATH"]
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Could not find {path}. Check CONFIG['CLASS_NAMES_PATH'] at the top "
            f"of this script matches your actual project layout."
        )
    with open(path, "r") as f:
        names = [line.strip() for line in f if line.strip()]
    return names


def build_model(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(CONFIG["MODEL_PATH"], map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ---------- 1. Class distribution ----------
def plot_class_distribution(class_names):
    print("\n[1/5] Class distribution...")
    counts = {"train": {}, "val": {}}
    for split in ["train", "val"]:
        split_dir = os.path.join(CONFIG["DATA_DIR"], split)
        for cls in class_names:
            cls_dir = os.path.join(split_dir, cls)
            n = len(os.listdir(cls_dir)) if os.path.isdir(cls_dir) else 0
            counts[split][cls] = n

    train_counts = [counts["train"][c] for c in class_names]
    val_counts = [counts["val"][c] for c in class_names]

    x = np.arange(len(class_names))
    width = 0.4
    fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 0.5), 6))
    ax.bar(x - width / 2, train_counts, width, label="train")
    ax.bar(x + width / 2, val_counts, width, label="val")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=75, ha="right")
    ax.set_ylabel("Image count")
    ax.set_title("Class distribution (train vs val)")
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(CONFIG["RESULTS_DIR"], "class_distribution.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    nonzero_train = [c for c in train_counts if c > 0]
    if not nonzero_train:
        print(f"  WARNING: every class has 0 images in train/ under {CONFIG['DATA_DIR']}.")
        print(f"  Check that CONFIG['CLASS_NAMES_PATH'] matches the folder names in "
              f"{CONFIG['DATA_DIR']}/train — a mismatch here is the usual cause.")
    else:
        imbalance_ratio = max(nonzero_train) / min(nonzero_train)
        print(f"  Saved {out_path}")
        print(f"  Train imbalance ratio (max/min class count): {imbalance_ratio:.2f}")


# ---------- 2. Sample grid ----------
def plot_sample_grid(class_names):
    print("\n[2/5] Sample image grid...")
    from PIL import Image

    classes_to_show = class_names[: CONFIG["MAX_CLASSES_IN_GRID"]]
    n_per_class = CONFIG["SAMPLES_PER_CLASS_GRID"]

    fig, axes = plt.subplots(
        len(classes_to_show), n_per_class,
        figsize=(n_per_class * 2, len(classes_to_show) * 2)
    )
    if len(classes_to_show) == 1:
        axes = np.expand_dims(axes, 0)

    for row, cls in enumerate(classes_to_show):
        cls_dir = os.path.join(CONFIG["DATA_DIR"], "train", cls)
        files = os.listdir(cls_dir) if os.path.isdir(cls_dir) else []
        random.shuffle(files)
        for col in range(n_per_class):
            ax = axes[row][col]
            ax.axis("off")
            if col < len(files):
                img = Image.open(os.path.join(cls_dir, files[col])).convert("RGB")
                ax.imshow(img)
            if col == 0:
                ax.set_ylabel(cls, fontsize=8)
                ax.axis("on")
                ax.set_xticks([])
                ax.set_yticks([])

    plt.tight_layout()
    out_path = os.path.join(CONFIG["RESULTS_DIR"], "sample_grid.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")
    if len(class_names) > CONFIG["MAX_CLASSES_IN_GRID"]:
        print(f"  (showing first {CONFIG['MAX_CLASSES_IN_GRID']} of {len(class_names)} classes)")


# ---------- Shared: load val set ----------
def get_val_loader():
    tf = transforms.Compose([
        transforms.Resize((CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_dir = os.path.join(CONFIG["DATA_DIR"], "val")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"{val_dir} not found. Check CONFIG['DATA_DIR'] matches your project layout."
        )
    val_ds = datasets.ImageFolder(val_dir, transform=tf)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=False)
    return val_ds, val_loader


def extract_predictions_and_features(model, val_loader):
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()

    all_labels, all_preds, all_feats = [], [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            feats = feature_extractor(imgs).squeeze(-1).squeeze(-1)
            logits = model.fc(feats)
            preds = logits.argmax(dim=1).cpu().numpy()

            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_feats.append(feats.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.concatenate(all_feats, axis=0),
    )


# ---------- 3. Confusion matrix + classification report ----------
def plot_confusion_and_report(y_true, y_pred, class_names):
    print("\n[3/5] Confusion matrix + classification report...")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.5), max(8, len(class_names) * 0.5)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=75, cmap="Blues", colorbar=True)
    ax.set_title("Confusion Matrix (val set)")
    plt.tight_layout()
    out_path = os.path.join(CONFIG["RESULTS_DIR"], "confusion_matrix.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path} (overwrote any existing file)")

    report_txt = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    report_path_txt = os.path.join(CONFIG["RESULTS_DIR"], "classification_report.txt")
    with open(report_path_txt, "w") as f:
        f.write(report_txt)
    print(f"  Saved {report_path_txt}")

    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True
    )
    report_path_csv = os.path.join(CONFIG["RESULTS_DIR"], "classification_report.csv")
    with open(report_path_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1-score", "support"])
        for cls, metrics in report_dict.items():
            if isinstance(metrics, dict):
                writer.writerow([
                    cls, f"{metrics['precision']:.4f}", f"{metrics['recall']:.4f}",
                    f"{metrics['f1-score']:.4f}", metrics["support"]
                ])
    print(f"  Saved {report_path_csv}")

    overall_acc = accuracy_score(y_true, y_pred)
    print(f"  Overall val accuracy: {overall_acc * 100:.1f}%")
    return overall_acc


# ---------- 4/5. PCA scatter + SVM-on-PCA comparison ----------
def pca_and_svm(features, y_true, cnn_acc, class_names):
    print("\n[4/5] PCA scatter...")
    pca = PCA(n_components=2, random_state=CONFIG["SEED"])
    feats_2d = pca.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(feats_2d[:, 0], feats_2d[:, 1], c=y_true, cmap="tab20", s=15)
    ax.set_title("PCA (2D) of penultimate-layer features — val set")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")

    if len(class_names) <= 20:
        handles, _ = scatter.legend_elements(num=len(class_names))
        ax.legend(handles, class_names, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)

    plt.tight_layout()
    out_path = os.path.join(CONFIG["RESULTS_DIR"], "pca_scatter.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")
    print(f"  Total variance explained by 2 PCs: {sum(pca.explained_variance_ratio_)*100:.1f}%")

    print("\n[5/5] SVM on 2D PCA features vs CNN...")
    X_train, X_test, y_train, y_test = train_test_split(
        feats_2d, y_true, test_size=0.3, random_state=CONFIG["SEED"], stratify=y_true
    )
    svm = SVC(kernel="rbf")
    svm.fit(X_train, y_train)
    svm_preds = svm.predict(X_test)
    svm_acc = accuracy_score(y_test, svm_preds)

    comparison_txt = (
        f"CNN (ResNet18, 512-dim features) accuracy on full val set: {cnn_acc*100:.2f}%\n"
        f"SVM (RBF kernel) on 2D PCA-reduced features, held-out split: {svm_acc*100:.2f}%\n\n"
        f"Interpretation: the CNN uses the full 512-dim feature space and heavily "
        f"outperforms the SVM restricted to only 2 PCA dimensions, which is expected — "
        f"PCA-to-2D is used here for visualization and to demonstrate dimensionality "
        f"reduction, not as the production classifier.\n"
    )
    out_path = os.path.join(CONFIG["RESULTS_DIR"], "pca_svm_comparison.txt")
    with open(out_path, "w") as f:
        f.write(comparison_txt)
    print(f"  Saved {out_path}")
    print("  " + comparison_txt.replace("\n", "\n  "))


def main():
    class_names = load_class_names()
    print(f"Loaded {len(class_names)} classes from {CONFIG['CLASS_NAMES_PATH']}.")

    plot_class_distribution(class_names)
    plot_sample_grid(class_names)

    model = build_model(len(class_names))
    val_ds, val_loader = get_val_loader()

    if val_ds.classes != class_names:
        print("\n  WARNING: val folder class order != class_names.txt order/content.")
        print("  This usually means the model's class list doesn't match the current")
        print("  classification_dataset/val folders. Using val_ds.classes (the actual")
        print("  folder names) so results are at least internally consistent — but if")
        print("  best_model.pth was trained on a DIFFERENT class set, these results")
        print("  will be meaningless. Retrain with src\\03_train_classifier_v2.py first")
        print("  if class_names.txt in models/ and classification_dataset/ don't match.")
        class_names = val_ds.classes

    y_true, y_pred, feats = extract_predictions_and_features(model, val_loader)
    cnn_acc = plot_confusion_and_report(y_true, y_pred, class_names)
    pca_and_svm(feats, y_true, cnn_acc, class_names)

    print(f"\nAll results saved to ./{CONFIG['RESULTS_DIR']}/")


if __name__ == "__main__":
    main()
