"""
Train the final ResNet18 scene-recognition model.

Usage:
    python train.py
"""

import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights
from torchvision.transforms import v2

# ---- Final hyperparameters (from probe sweep + hyperparameter tuning stage) ----
SEED = 0
TRAIN_ROOT = "data/train"
TEST_ROOT = "data/test"
IMG_SIZE = 224
BATCH_SIZE = 32
VAL_FRACTION = 0.20
NUM_WORKERS = 4

DROPOUT = 0.3
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
MIXUP_CUTMIX_ALPHA = 1.0
EPOCHS = 70
PATIENCE = 10

CHECKPOINT_OUT = "checkpoints/resnet18_final.pth"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_resnet18(num_classes, dropout_rate):
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    if dropout_rate > 0:
        model.fc = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(in_features, num_classes))
    else:
        model.fc = nn.Linear(in_features, num_classes)
    return model


@torch.inference_mode()
def evaluate(model, loader, device, criterion):
    model.eval()
    correct, total, running_loss = 0, 0, 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Using device: {device}")
    print(f"Seed: {SEED}")

    set_random_seed(SEED)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    full_train_aug = datasets.ImageFolder(TRAIN_ROOT, transform=train_transform)
    full_train_eval = datasets.ImageFolder(TRAIN_ROOT, transform=eval_transform)
    class_names = full_train_aug.classes
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    val_size = int(round(len(full_train_aug) * VAL_FRACTION))
    train_size = len(full_train_aug) - val_size
    generator = torch.Generator().manual_seed(SEED)
    train_subset, val_subset = random_split(full_train_aug, [train_size, val_size], generator=generator)
    val_subset = torch.utils.data.Subset(full_train_eval, val_subset.indices)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=pin_memory)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=pin_memory)

    model = build_resnet18(num_classes, DROPOUT).to(device)

    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    cutmix = v2.CutMix(num_classes=num_classes, alpha=MIXUP_CUTMIX_ALPHA)
    mixup = v2.MixUp(num_classes=num_classes, alpha=MIXUP_CUTMIX_ALPHA)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    epochs_no_improve = 0
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, total_seen = 0.0, 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            images, labels = cutmix_or_mixup(images, labels)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_seen += images.size(0)

        scheduler.step()

        train_loss = total_loss / total_seen
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        elapsed = time.time() - start
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:02d}/{EPOCHS} | train loss {train_loss:.4f} | "
              f"val loss {val_loss:.4f} | val acc {val_acc:.4f} | lr {current_lr:.6f} | {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping after {epoch} epochs (no improvement for {PATIENCE} epochs).")
            break

    model.load_state_dict(best_state)
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    out_path = Path(CHECKPOINT_OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_state,
        "class_names": class_names,
        "num_classes": num_classes,
        "dropout": DROPOUT,
        "img_size": IMG_SIZE,
        "best_val_acc": best_val_acc,
        "seed": SEED,
    }, out_path)
    print(f"Saved checkpoint to {out_path}")


if __name__ == "__main__":
    main()
