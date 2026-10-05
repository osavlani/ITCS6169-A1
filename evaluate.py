"""
Evaluate a trained checkpoint on the test set.

Usage:
    python evaluate.py --checkpoint checkpoints/resnet18_final.pth
"""

import argparse

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

TEST_ROOT = "data/test"
BATCH_SIZE = 32
NUM_WORKERS = 4

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_resnet18(num_classes, dropout_rate):
    model = models.resnet18(weights=None)  # weights get overwritten by the checkpoint
    in_features = model.fc.in_features
    if dropout_rate > 0:
        model.fc = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(in_features, num_classes))
    else:
        model.fc = nn.Linear(in_features, num_classes)
    return model


@torch.inference_mode()
def evaluate_model(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Path to a .pth checkpoint saved by train.py.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Using device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    class_names = ckpt["class_names"]
    num_classes = ckpt["num_classes"]
    dropout = ckpt["dropout"]
    img_size = ckpt["img_size"]
    seed = ckpt["seed"]

    print(f"Seed used for training: {seed}")
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Reported best validation accuracy at train time: {ckpt['best_val_acc']:.4f}")

    model = build_resnet18(num_classes, dropout).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    test_dataset = datasets.ImageFolder(TEST_ROOT, transform=eval_transform)

    if test_dataset.classes != class_names:
        raise ValueError(
            f"Test set classes do not match the classes the checkpoint was trained on.\n"
            f"Checkpoint: {class_names}\nTest folder: {test_dataset.classes}\n"
            f"Check for a missing or misnamed class folder."
        )

    pin_memory = torch.cuda.is_available()
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin_memory)

    test_loss, test_acc = evaluate_model(model, test_loader, device)
    print(f"Final test loss: {test_loss:.4f}")
    print(f"Final test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
