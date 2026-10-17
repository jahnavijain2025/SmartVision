
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score

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
    def __init__(self, images, labels, indices, known_only=False):
        self.images = images
        self.labels = labels
        self.indices = np.asarray(indices)
        self.known_only = known_only
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
        label = int(self.labels[i])

        if self.known_only:
            label = LABEL_MAP[label]

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

    def extract_features(self, x):
        x = self.stem(x)
        x = self.features(x)
        return x.flatten(1)


def extract(model, loader):
    features, labels = [], []
    model.eval()

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(DEVICE)
            feats = model.extract_features(images)
            features.append(feats.cpu().numpy())
            labels.append(targets.numpy())

    return np.concatenate(features), np.concatenate(labels)


def minimum_mahalanobis(features, means, precision):
    scores = []

    for feature in features:
        differences = feature[None, :] - means
        distances = np.einsum(
            "ij,jk,ik->i", differences, precision, differences
        )
        scores.append(np.min(distances))

    return np.asarray(scores)


def main():
    splits = np.load(ROOT / "data" / "processed" / "dataset_splits.npz")
    checkpoint_path = ROOT / "models" / "known_classifier_improved.pth"

    train_raw = CIFAR10(
        root=str(ROOT / "data" / "raw"),
        train=True, download=False
    )
    test_raw = CIFAR10(
        root=str(ROOT / "data" / "raw"),
        train=False, download=False
    )

    train_set = ImageDataset(
        train_raw.data, train_raw.targets,
        splits["train_indices"], known_only=True
    )
    val_set = ImageDataset(
        train_raw.data, train_raw.targets,
        splits["val_indices"], known_only=True
    )
    known_set = ImageDataset(
        test_raw.data, test_raw.targets,
        splits["known_test_indices"], known_only=True
    )
    unknown_set = ImageDataset(
        test_raw.data, test_raw.targets,
        splits["unknown_test_indices"], known_only=False
    )

    def loader(dataset):
        return DataLoader(
            dataset, batch_size=256, shuffle=False,
            num_workers=0, pin_memory=torch.cuda.is_available()
        )

    checkpoint = torch.load(
        checkpoint_path, map_location=DEVICE, weights_only=True
    )
    if checkpoint["known_classes"] != KNOWN:
        raise ValueError("Checkpoint classes do not match.")

    model = ImprovedCNN().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("Device:", DEVICE)
    print("Extracting training features...")
    train_features, train_labels = extract(model, loader(train_set))

    print("Extracting validation features...")
    val_features, _ = extract(model, loader(val_set))

    print("Extracting test features...")
    known_features, _ = extract(model, loader(known_set))
    unknown_features, _ = extract(model, loader(unknown_set))

    # Calculate one mean vector for each known class.
    means = np.stack([
        train_features[train_labels == class_id].mean(axis=0)
        for class_id in range(len(KNOWN))
    ])

    # Estimate shared within-class covariance with regularization.
    residuals = np.concatenate([
        train_features[train_labels == class_id]
        - means[class_id]
        for class_id in range(len(KNOWN))
    ])
    covariance = np.cov(residuals, rowvar=False)

    # Shrink covariance slightly toward the identity for stability.
    scale = np.trace(covariance) / covariance.shape[0]
    covariance = 0.9 * covariance + 0.1 * scale * np.eye(
        covariance.shape[0]
    )
    precision = np.linalg.pinv(covariance)

    val_scores = minimum_mahalanobis(val_features, means, precision)
    known_scores = minimum_mahalanobis(known_features, means, precision)
    unknown_scores = minimum_mahalanobis(
        unknown_features, means, precision
    )

    # Select threshold from known validation images only.
    threshold = float(np.percentile(val_scores, 95))

    known_reject = float(np.mean(known_scores > threshold))
    unknown_recall = float(np.mean(unknown_scores > threshold))

    y_true = np.concatenate([
        np.zeros(len(known_scores), dtype=int),
        np.ones(len(unknown_scores), dtype=int),
    ])
    all_scores = np.concatenate([known_scores, unknown_scores])

    auroc = roc_auc_score(y_true, all_scores)
    aupr = average_precision_score(y_true, all_scores)

    print("\n=== IMPROVED MAHALANOBIS OOD RESULTS ===")
    print(f"Validation threshold (95th percentile): {threshold:.4f}")
    print(f"Known-image rejection rate: {known_reject:.4f}")
    print(f"Unknown detection recall: {unknown_recall:.4f}")
    print(f"OOD AUROC: {auroc:.4f}")
    print(f"OOD AUPR: {aupr:.4f}")

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    np.savez(
        out_dir / "improved_mahalanobis_results.npz",
        known_scores=known_scores,
        unknown_scores=unknown_scores,
        threshold=np.asarray(threshold),
        known_rejection_rate=np.asarray(known_reject),
        unknown_recall=np.asarray(unknown_recall),
        auroc=np.asarray(auroc),
        aupr=np.asarray(aupr),
    )
    print("Saved: results/improved_mahalanobis_results.npz")


if __name__ == "__main__":
    main()
