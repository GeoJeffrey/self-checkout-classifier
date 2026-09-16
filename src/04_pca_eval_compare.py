"""
Evaluation script for the classifier trained by 03_train_classifier.py.

Does:
    1. Extracts penultimate-layer features for the val set using the trained ResNet18.
    2. PCA (2D) projection + scatter plot of features, colored by class.
    3. Classification report (precision/recall/F1) + confusion matrix for the CNN.
    4. Trains an SVM on the PCA-reduced features as a second, comparison model.

Produces:
    - pca_plot.png
    - confusion_matrix.png
    - metrics_report.txt   (classification report + SVM accuracy, for your write-up)

Run this after 03_train_classifier.py.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ----------------------- CONFIG (edit if needed) -----------------------
CONFIG = {
    "DATA_DIR": "classification_dataset",
    "MODEL_PATH": "models/best_model.pth",
    "CLASS_NAMES_PATH": "models/class_names.txt",
    "OUT_DIR": "results",
    "IMG_SIZE": 224,
    "BATCH_SIZE": 32,
}
# ------------------------------------------------------------------------


def load_class_names(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def build_model_for_inference(num_classes, model_path, device):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def get_val_loader(data_dir, img_size, batch_size):
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])
    val_dir = os.path.join(data_dir, "val")
    val_ds = datasets.ImageFolder(val_dir, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return val_loader, val_ds


def extract_features_and_preds(model, val_loader, device):
    """
    Extract penultimate-layer (pre-fc) features by hooking into avgpool output,
    plus the model's own class predictions and true labels.
    """
    features = []
    preds = []
    labels = []

    # Hook the avgpool layer (output right before the final fc layer)
    activation = {}

    def hook(module, inp, out):
        activation["feat"] = out

    handle = model.avgpool.register_forward_hook(hook)

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            out = model(x)
            feat = activation["feat"].squeeze(-1).squeeze(-1)  # (B, 512) for resnet18

            features.append(feat.cpu().numpy())
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(y.numpy())

    handle.remove()
    features = np.vstack(features)
    return features, np.array(preds), np.array(labels)


def plot_pca(features, labels, class_names, out_path):
    pca = PCA(n_components=2)
    proj = pca.fit_transform(features)

    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(class_names):
        idx = labels == i
        if idx.sum() == 0:
            continue
        plt.scatter(proj[idx, 0], proj[idx, 1], label=cls, s=15)
    plt.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.title("PCA (2D) of classifier features — val set")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")

    return proj


def plot_confusion_matrix(labels, preds, class_names, out_path):
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.6), max(6, len(class_names) * 0.6)))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        xticks_rotation=45, ax=ax, colorbar=False
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")
    return cm


def run_svm_comparison(proj, labels, report_lines):
    """
    Train/test an SVM on the same PCA-reduced (2D) features, as a
    classical-ML comparison point against the CNN.
    """
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2 or counts.min() < 2:
        msg = ("Skipping SVM comparison: need at least 2 samples per class "
               "in the val set for a train/test split.")
        print(msg)
        report_lines.append(msg)
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        proj, labels, test_size=0.2, random_state=42, stratify=labels
    )
    svm = SVC(kernel="rbf")
    svm.fit(X_train, y_train)
    acc = svm.score(X_test, y_test)
    print(f"SVM on PCA features — test accuracy: {acc:.4f}")
    report_lines.append(f"SVM (RBF kernel) on 2D PCA features — test accuracy: {acc:.4f}")
    return acc


def main():
    data_dir = CONFIG["DATA_DIR"]
    out_dir = CONFIG["OUT_DIR"]
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    class_names = load_class_names(CONFIG["CLASS_NAMES_PATH"])
    print(f"Classes ({len(class_names)}): {class_names}")

    model = build_model_for_inference(len(class_names), CONFIG["MODEL_PATH"], device)
    val_loader, val_ds = get_val_loader(data_dir, CONFIG["IMG_SIZE"], CONFIG["BATCH_SIZE"])

    # Make sure class order in the val ImageFolder matches class_names.txt
    if val_ds.classes != class_names:
        print("WARNING: val_ds.classes does not match class_names.txt order.")
        print(f"  val_ds.classes:   {val_ds.classes}")
        print(f"  class_names.txt:  {class_names}")
        print("  Using val_ds.classes for this run to stay consistent with the model's val loader.")
        class_names = val_ds.classes

    features, preds, labels = extract_features_and_preds(model, val_loader, device)

    report_lines = []

    # --- Classification report ---
    report = classification_report(labels, preds, target_names=class_names, zero_division=0)
    print("\n=== Classification report (CNN) ===")
    print(report)
    report_lines.append("=== Classification report (CNN, ResNet18 transfer learning) ===")
    report_lines.append(report)

    # --- Confusion matrix ---
    plot_confusion_matrix(labels, preds, class_names, os.path.join(out_dir, "confusion_matrix.png"))

    # --- PCA plot ---
    proj = plot_pca(features, labels, class_names, os.path.join(out_dir, "pca_plot.png"))

    # --- SVM comparison on PCA features ---
    report_lines.append("\n=== Model comparison: CNN vs SVM (on 2D PCA features) ===")
    svm_acc = run_svm_comparison(proj, labels, report_lines)

    cnn_acc = (preds == labels).mean()
    report_lines.append(f"CNN (ResNet18) val accuracy (full feature space): {cnn_acc:.4f}")
    if svm_acc is not None:
        report_lines.append(
            f"Note: SVM was trained on only 2 PCA dimensions, so this is not a fully "
            f"fair comparison — it's meant to show a classical-ML baseline on reduced "
            f"features, not to outperform the CNN."
        )

    # --- Save text report ---
    report_path = os.path.join(out_dir, "metrics_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nSaved {report_path}")


if __name__ == "__main__":
    main()
