
from pathlib import Path
import copy
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

KNOWN = ["airplane", "automobile", "bird", "cat", "deer"]
CLASS_IDS = [CLASSES.index(c) for c in KNOWN]
LABEL_MAP = {old: new for new, old in enumerate(CLASS_IDS)}

BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.001
SEED = 42

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.247, 0.243, 0.261)


class ImageDataset(Dataset):
    def __init__(self, images, labels, indices, training=False):
        self.images = images
        self.labels = labels
        self.indices = indices

        ops = [transforms.ToPILImage()]

        if training:
            ops += [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),

                # NEW: mild colour variation during training only
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.15,
                    hue=0.02,
                ),
            ]

        ops += [
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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        return self.relu(out + residual)


class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.features = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        return self.classifier(x.flatten(1))


def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    count = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            count += labels.size(0)

    return total_loss / count, correct / count


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    split_file = ROOT / "data" / "processed" / "dataset_splits.npz"
    data_dir = ROOT / "data" / "raw"

    if not split_file.exists():
        raise FileNotFoundError(
            "Dataset split missing. Run src/prepare_dataset.py first."
        )

    splits = np.load(split_file)

    train_raw = CIFAR10(
        root=str(data_dir),
        train=True,
        download=False,
    )

    train_set = ImageDataset(
        train_raw.data,
        train_raw.targets,
        splits["train_indices"],
        training=True,
    )

    val_set = ImageDataset(
        train_raw.data,
        train_raw.targets,
        splits["val_indices"],
        training=False,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = ImprovedCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0005,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    print("Device:", DEVICE)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("Experiment: ImprovedCNN + mild ColorJitter")
    print("Training classes:", KNOWN)
    print("Training images:", len(train_set))
    print("Validation images:", len(val_set))
    print("Epochs:", EPOCHS)

    best_val_acc = 0.0
    best_state = None
    history = []
    start_time = time.time()

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
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            count += labels.size(0)

        train_loss = total_loss / count
        train_acc = correct / count

        val_loss, val_acc = evaluate(
            model, val_loader, criterion
        )

        scheduler.step()

        history.append([
            epoch + 1,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        ])

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.6f}"
        )

        if best_state is None or val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model_dir = ROOT / "models"
    model_dir.mkdir(exist_ok=True)

    if best_state is None:
        raise RuntimeError("No best model checkpoint was produced.")

    checkpoint = {
        "model_state_dict": best_state,
        "known_classes": KNOWN,
        "class_ids": CLASS_IDS,
        "architecture": "ImprovedCNN",
        "best_val_accuracy": best_val_acc,
        "epochs": EPOCHS,
        "augmentation": [
            "RandomCrop(32, padding=4)",
            "RandomHorizontalFlip",
            "ColorJitter(brightness=0.15, contrast=0.15, "
            "saturation=0.15, hue=0.02)",
        ],
    }

    # Save separately: preserve the original baseline checkpoint.
    torch.save(
        checkpoint,
        model_dir / "known_classifier_augmented.pth",
    )

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    np.save(
        results_dir / "augmented_training_history.npy",
        np.array(history, dtype=np.float32),
    )

    print("\nAugmented training complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print("Saved model: models/known_classifier_augmented.pth")
    print("Saved history: results/augmented_training_history.npy")
    print(
        f"Elapsed time: {(time.time() - start_time) / 60:.1f} minutes"
    )


if __name__ == "__main__":
    main()
