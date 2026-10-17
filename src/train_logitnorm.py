
from pathlib import Path
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.models import resnet18


ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KNOWN = ["airplane", "automobile", "bird", "cat", "deer"]
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
KNOWN_IDS = [CLASSES.index(c) for c in KNOWN]
LABEL_MAP = {old: new for new, old in enumerate(KNOWN_IDS)}

BATCH_SIZE = 64
EPOCHS = 5
TAU = 0.04
LEARNING_RATE = 0.00002
SEED = 42

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class ImageDataset(Dataset):
    def __init__(self, images, labels, indices, training=False):
        self.images = images
        self.labels = labels
        self.indices = np.asarray(indices)

        ops = [transforms.ToPILImage()]
        if training:
            ops += [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        ops += [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
        self.transform = transforms.Compose(ops)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = int(self.indices[index])
        image = self.transform(self.images[i])
        label = LABEL_MAP[int(self.labels[i])]
        return image, label


def logitnorm_loss(logits, labels):
    norms = torch.linalg.vector_norm(
        logits, ord=2, dim=1, keepdim=True
    ).clamp_min(1e-7)
    normalized_logits = logits / (norms * TAU)
    return F.cross_entropy(normalized_logits, labels)


def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(images)

            # Evaluate ordinary classification accuracy.
            total_loss += F.cross_entropy(
                logits, labels, reduction="sum"
            ).item()
            correct += (logits.argmax(1) == labels).sum().item()
            count += labels.size(0)

    return total_loss / count, correct / count


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    split_file = ROOT / "data/processed/dataset_splits.npz"
    data_dir = ROOT / "data/raw"
    original_path = ROOT / "models/known_classifier_resnet18.pth"
    output_path = ROOT / "models/known_classifier_logitnorm.pth"

    if not original_path.exists():
        raise FileNotFoundError(original_path)

    splits = np.load(split_file)
    train_raw = CIFAR10(
        root=str(data_dir), train=True, download=False
    )

    train_set = ImageDataset(
        train_raw.data, train_raw.targets,
        splits["train_indices"], training=True
    )
    val_set = ImageDataset(
        train_raw.data, train_raw.targets,
        splits["val_indices"], training=False
    )

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    checkpoint = torch.load(
        original_path, map_location=DEVICE, weights_only=True
    )

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(KNOWN))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)

    # Fine-tune the last residual stage and classifier only.
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    optimizer = torch.optim.AdamW(
        [
            {"params": model.layer4.parameters(), "lr": LEARNING_RATE},
            {"params": model.fc.parameters(), "lr": LEARNING_RATE},
        ],
        weight_decay=0.0001,
    )

    best_accuracy = -1.0
    best_state = None
    history = []
    start = time.time()

    print("Device:", DEVICE)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("Training images:", len(train_set))
    print("Validation images:", len(val_set))
    print("LogitNorm tau:", TAU)
    print("Starting from the existing ResNet-18 checkpoint.")
    print("Original checkpoint will NOT be modified.")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        count = 0

        for images, labels in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = logitnorm_loss(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            count += labels.size(0)

        train_loss = total_loss / count
        train_accuracy = correct / count
        val_loss, val_accuracy = evaluate(model, val_loader)

        history.append([
            epoch + 1, train_loss, train_accuracy,
            val_loss, val_accuracy
        ])

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_accuracy:.4f} | "
            f"val_acc={val_accuracy:.4f}"
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_state = copy.deepcopy(model.state_dict())

    # Save a distinct experimental checkpoint.
    torch.save({
        "model_state_dict": best_state,
        "known_classes": KNOWN,
        "class_ids": KNOWN_IDS,
        "architecture": "ResNet18_LogitNorm_finetuned",
        "best_val_accuracy": best_accuracy,
        "tau": TAU,
        "epochs": EPOCHS,
        "source_checkpoint": "known_classifier_resnet18.pth",
    }, output_path)

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    np.save(
        results_dir / "logitnorm_training_history.npy",
        np.asarray(history, dtype=np.float64)
    )

    print("\nTraining complete.")
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print(f"Experimental checkpoint: {output_path}")
    print("Original ResNet-18 checkpoint remains unchanged.")
    print(f"Elapsed time: {(time.time() - start) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
