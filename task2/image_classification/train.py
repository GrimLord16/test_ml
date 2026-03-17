"""
train.py - Train ResNet-18 for Mammals-45 Image Classification
Usage:
    python train.py --data_dir ./mammals45 --output_dir ./animal_classifier
"""

import argparse
import json
import os
import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split


# ImageNet normalization statistics (used because we start from ImageNet weights)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ResNet-18 on the Mammals-45 dataset"
    )
    parser.add_argument(
        "--data_dir",
        default="./mammals45",
        help="Root directory of the Mammals-45 dataset (default: ./mammals45)",
    )
    parser.add_argument(
        "--output_dir",
        default="./animal_classifier",
        help="Directory to save model weights and class mapping (default: ./animal_classifier)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training and validation (default: 32)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate for Adam optimizer (default: 1e-4)",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Image resize dimension (default: 224)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transforms(img_size: int) -> dict:
    """Return train and validation image transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomRotation(degrees=20),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return {"train": train_transform, "val": val_transform}


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def build_datasets(data_dir: str, img_size: int):
    """
    Load full dataset with ImageFolder, then split indices into
    train and validation subsets with their respective transforms.

    Returns (train_subset, val_subset, idx_to_class)
    where idx_to_class maps integer index → class name string.
    """
    tfms = get_transforms(img_size)

    # Load once (no transform) to get class list and targets for stratified split
    full_dataset = datasets.ImageFolder(root=data_dir)

    class_names = full_dataset.classes
    idx_to_class = {i: name for i, name in enumerate(class_names)}
    num_classes = len(class_names)

    # Stratified train/val split (80/20)
    targets = full_dataset.targets
    indices = list(range(len(full_dataset)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=targets,
        random_state=42,
    )

    # Create separate dataset copies with appropriate transforms
    train_dataset = datasets.ImageFolder(root=data_dir, transform=tfms["train"])
    val_dataset   = datasets.ImageFolder(root=data_dir, transform=tfms["val"])

    train_subset = Subset(train_dataset, train_idx)
    val_subset   = Subset(val_dataset,   val_idx)

    print(f"Classes ({num_classes}): {class_names}")
    print(f"Train samples: {len(train_subset)}, Validation samples: {len(val_subset)}")
    return train_subset, val_subset, idx_to_class


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(num_classes: int) -> nn.Module:
    """
    Load a pre-trained ResNet-18 and replace the final fully-connected layer
    with a new Linear(512, num_classes) head.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features  # 512 for ResNet-18
    model.fc = nn.Linear(in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    output_dir: str,
    device: torch.device,
) -> dict:
    """
    Train the model, track best validation accuracy, and save best weights.

    Returns a dict of training history.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_val_acc = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    os.makedirs(output_dir, exist_ok=True)

    print("\nStarting training...")
    print("=" * 65)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # ---- Training phase ----
        model.train()
        running_loss, running_correct, n_train = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            running_loss += loss.item() * inputs.size(0)
            running_correct += (preds == labels).sum().item()
            n_train += inputs.size(0)

        train_loss = running_loss / n_train
        train_acc = running_correct / n_train

        # ---- Validation phase ----
        model.eval()
        val_loss, val_correct, n_val = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                preds = outputs.argmax(dim=1)
                val_loss += loss.item() * inputs.size(0)
                val_correct += (preds == labels).sum().item()
                n_val += inputs.size(0)

        val_loss /= n_val
        val_acc = val_correct / n_val

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(best_model_weights,
                       os.path.join(output_dir, "best_model.pth"))

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch:02d}/{num_epochs:02d}]  "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  |  "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}  "
            f"({elapsed:.1f}s)"
        )

    print("=" * 65)
    print(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")

    model.load_state_dict(best_model_weights)
    return history




def main():
    args = parse_args()

    print("=" * 65)
    print("Animal Image Classifier — Training Configuration")
    print("=" * 65)
    for k, v in vars(args).items():
        print(f"  {k:25s}: {v}")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    train_subset, val_subset, idx_to_class = build_datasets(
        args.data_dir, args.img_size
    )
    num_classes = len(idx_to_class)

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = build_model(num_classes=num_classes).to(device)
    print(f"Model: ResNet-18 (modified FC → {num_classes} classes)\n")

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        device=device,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    class_mapping_path = os.path.join(args.output_dir, "class_mapping.json")
    with open(class_mapping_path, "w") as f:
        json.dump(idx_to_class, f, indent=2)
    print(f"Class mapping saved to '{class_mapping_path}'.")

    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to '{history_path}'.")
    print("Done.")


if __name__ == "__main__":
    main()
