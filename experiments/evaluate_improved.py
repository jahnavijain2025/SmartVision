
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
KNOWN = ["airplane", "automobile", "bird", "cat", "deer"]
CLASS_IDS = [CLASSES.index(c) for c in KNOWN]
LABEL_MAP = {old: new for new, old in enumerate(CLASS_IDS)}

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.247, 0.243, 0.261)


class ImageDataset(Dataset):
    def __init__(self, images, labels, indices):
        self.images = images
        self.labels = labels
        self.indices = indices
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

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
            in_channels, out_channels, 3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, 1,
                    stride=stride, bias=False
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


def main():
    split_path = ROOT / "data" / "processed" / "dataset_splits.npz"
    checkpoint_path = ROOT / "models" / "known_classifier_augmented.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model not found: {checkpoint_path}")

    splits = np.load(split_path)

    raw = CIFAR10(
        root=str(ROOT / "data" / "raw"),
        train=False,
        download=False,
    )

    known_set = ImageDataset(
        raw.data,
        raw.targets,
        splits["known_test_indices"],
    )

    loader = DataLoader(
        known_set,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )

    if checkpoint["known_classes"] != KNOWN:
        raise ValueError("Checkpoint classes don't match evaluation classes.")

    model = ImprovedCNN().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            logits = model(images)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)

    report = classification_report(
        all_labels,
        all_preds,
        labels=list(range(len(KNOWN))),
        target_names=KNOWN,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        all_labels,
        all_preds,
        labels=list(range(len(KNOWN))),
    )

    print("\n=== AUGMENTED MODEL: KNOWN TEST SET ===")
    print("Device:", DEVICE)
    print("Test images:", len(known_set))
    print(f"Known test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\nPer-class metrics:")
    print(classification_report(
        all_labels,
        all_preds,
        labels=list(range(len(KNOWN))),
        target_names=KNOWN,
        zero_division=0,
    ))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(matrix)

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    np.savez(
        out_dir / "augmented_known_test_results.npz",
        y_true=np.asarray(all_labels),
        y_pred=np.asarray(all_preds),
        accuracy=np.asarray(accuracy),
        confusion_matrix=matrix,
    )

    with open(out_dir / "augmented_known_test_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\nSaved results to results/augmented_known_test_results.npz")
    print("Saved report to results/augmented_known_test_report.json")


if __name__ == "__main__":
    main()
