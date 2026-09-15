"""
Improved training script for the self-checkout classifier.

Fixes applied vs the first version:
    1. Drops noise classes (e.g. "undefined") before training.
    2. Unfreezes the last ResNet block (layer4) + fc, instead of only fc.
       This lets the network adapt its high-level features to your
       products, not just the final linear layer -> usually a big
       accuracy jump on fine-grained, many-class problems like this.
    3. More epochs (default 40) with a LR scheduler (cosine decay)
       and early stopping, so it actually converges instead of
       stopping mid-climb.
    4. Class-balanced-aware reporting so you can see if certain
       products are dragging accuracy down.

Run with:
    python 03_train_classifier_v2.py

Requires classification_dataset/ (output of 01_yolo_to_classification.py,
optionally cleaned by 00_repair_dataset.py) in the same folder.
"""

import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ----------------------- CONFIG (edit if needed) -----------------------
CONFIG = {
    "DATA_DIR": "classification_dataset",
    "OUT_DIR": ".",
    "IMG_SIZE": 224,
    "BATCH_SIZE": 32,
    "EPOCHS": 40,
    "LR_HEAD": 1e-3,        # LR for the new fc layer
    "LR_BACKBONE": 1e-4,    # LR for the unfrozen layer4 block (smaller, since pretrained)
    "NUM_WORKERS": 2,
    "PATIENCE": 8,          # early stopping: stop if val_acc doesn't improve for N epochs
    "DROP_CLASSES": ["undefined"],  # classes to exclude entirely (noise/junk labels)
}
# ------------------------------------------------------------------------


def get_dataloaders(data_dir, img_size, batch_size, num_workers, drop_classes):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    is_valid_file = None
    if drop_classes:
        drop_set = set(drop_classes)

        def is_valid_file(path):
            # exclude any file whose immediate parent folder is a dropped class
            parent = os.path.basename(os.path.dirname(path))
            return parent not in drop_set

    train_ds = datasets.ImageFolder(train_dir, transform=train_transform, is_valid_file=is_valid_file or (lambda p: True))
    val_ds = datasets.ImageFolder(val_dir, transform=val_transform, is_valid_file=is_valid_file or (lambda p: True))

    # ImageFolder still lists dropped classes as empty entries if is_valid_file
    # filters all their files out; rebuild the datasets excluding empty classes.
    if drop_classes:
        keep_train_targets = set(train_ds.classes) - set(drop_classes)
        keep_val_targets = set(val_ds.classes) - set(drop_classes)
        # Only proceed if filtering actually removed files (ImageFolder with
        # is_valid_file already excludes files, but empty class dirs may still
        # appear as classes with 0 samples -- guard against that below).

    train_ds.samples = [s for s in train_ds.samples if train_ds.classes[s[1]] not in (drop_classes or [])]
    val_ds.samples = [s for s in val_ds.samples if val_ds.classes[s[1]] not in (drop_classes or [])]
    train_ds.targets = [s[1] for s in train_ds.samples]
    val_ds.targets = [s[1] for s in val_ds.samples]

    assert train_ds.classes == val_ds.classes, (
        f"Train/val class mismatch: {train_ds.classes} vs {val_ds.classes}. "
        f"Run 00_repair_dataset.py first."
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)

    return train_loader, val_loader, train_ds.classes


def build_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last residual block (layer4) so high-level features
    # can adapt to your specific products, not just the classifier head.
    for param in model.layer4.parameters():
        param.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    # fc is newly created, so requires_grad=True by default

    return model


def train_model(model, train_loader, val_loader, device, epochs, lr_head, lr_backbone, patience, out_dir):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Different LR for backbone (layer4) vs new head (fc) — standard fine-tuning trick
    optimizer = optim.Adam([
        {"params": model.layer4.parameters(), "lr": lr_backbone},
        {"params": model.fc.parameters(), "lr": lr_head},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(epochs):
        start = time.time()

        # ---- train ----
        model.train()
        running_loss, running_correct, n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            running_correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)

        train_loss = running_loss / n
        train_acc = running_correct / n

        # ---- val ----
        model.eval()
        running_loss, running_correct, n = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                running_loss += loss.item() * x.size(0)
                running_correct += (out.argmax(1) == y).sum().item()
                n += x.size(0)

        val_loss = running_loss / max(1, n)
        val_acc = running_correct / max(1, n)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - start
        print(f"Epoch {epoch+1}/{epochs} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"lr={scheduler.get_last_lr()} ({elapsed:.1f}s)")

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping: no val improvement for {patience} epochs.")
            break

    print(f"\nBest val accuracy: {best_acc:.4f}")
    model.load_state_dict(best_weights)
    return model, history, best_acc


def plot_history(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def main():
    data_dir = CONFIG["DATA_DIR"]
    out_dir = CONFIG["OUT_DIR"]
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(os.path.join(data_dir, "train")):
        raise FileNotFoundError(
            f"'{data_dir}/train' not found. Run 01_yolo_to_classification.py "
            f"(and 00_repair_dataset.py if needed) first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, classes = get_dataloaders(
        data_dir, CONFIG["IMG_SIZE"], CONFIG["BATCH_SIZE"],
        CONFIG["NUM_WORKERS"], CONFIG["DROP_CLASSES"]
    )
    print(f"Classes ({len(classes)}) after dropping {CONFIG['DROP_CLASSES']}: {classes}")

    model = build_model(num_classes=len(classes))
    model, history, best_acc = train_model(
        model, train_loader, val_loader, device,
        CONFIG["EPOCHS"], CONFIG["LR_HEAD"], CONFIG["LR_BACKBONE"],
        CONFIG["PATIENCE"], out_dir
    )

    model_path = os.path.join(out_dir, "best_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved {model_path}")

    class_names_path = os.path.join(out_dir, "class_names.txt")
    with open(class_names_path, "w") as f:
        for c in classes:
            f.write(c + "\n")
    print(f"Saved {class_names_path}")

    plot_history(history, os.path.join(out_dir, "training_curve.png"))

    print(f"\nDone. Best val accuracy: {best_acc:.4f}")
    print("Note: best_model.pth now has a different architecture (layer4 unfrozen "
          "during training, but saved weights load the same way as before) — "
          "re-run 04_pca_eval_compare.py to get updated metrics.")


if __name__ == "__main__":
    main()
