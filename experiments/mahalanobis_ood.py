
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALL_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
KNOWN = ["airplane", "automobile", "bird", "cat", "deer"]
KNOWN_IDS = [ALL_CLASSES.index(c) for c in KNOWN]
LABEL_MAP = {old: new for new, old in enumerate(KNOWN_IDS)}


# Must match the architecture used in train.py
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, 5)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


class ImageDataset(Dataset):
    def __init__(self, images, labels, indices):
        self.images = images
        self.labels = np.asarray(labels)
        self.indices = np.asarray(indices)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.247, 0.243, 0.261),
            ),
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = int(self.indices[index])
        return self.transform(self.images[i]), int(self.labels[i])


def extract_features(model, loader):
    features, labels = [], []
    model.eval()

    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(DEVICE)
            # Extract 128-dimensional learned representations.
            z = model.features(images).flatten(1)
            features.append(z.cpu().numpy())
            labels.append(batch_labels.numpy())

    return np.concatenate(features), np.concatenate(labels)


def main():
    split_file = ROOT / "data" / "processed" / "dataset_splits.npz"
    model_file = ROOT / "models" / "known_classifier.pth"

    splits = np.load(split_file)
    checkpoint = torch.load(
        model_file, map_location=DEVICE, weights_only=True
    )

    if checkpoint["known_classes"] != KNOWN:
        raise ValueError("Checkpoint classes do not match.")

    model = SmallCNN().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    data_dir = ROOT / "data" / "raw"
    train_data = CIFAR10(
        root=str(data_dir), train=True, download=False
    )
    test_data = CIFAR10(
        root=str(data_dir), train=False, download=False
    )

    def loader(images, labels, indices):
        ds = ImageDataset(images, labels, indices)
        return DataLoader(
            ds, batch_size=128, shuffle=False, num_workers=0
        )

    train_loader = loader(
        train_data.data, train_data.targets, splits["train_indices"]
    )
    val_loader = loader(
        train_data.data, train_data.targets, splits["val_indices"]
    )
    known_loader = loader(
        test_data.data, test_data.targets,
        splits["known_test_indices"]
    )
    unknown_loader = loader(
        test_data.data, test_data.targets,
        splits["unknown_test_indices"]
    )

    print("Extracting image features...")
    train_z, train_y = extract_features(model, train_loader)
    val_z, _ = extract_features(model, val_loader)
    known_z, known_y = extract_features(model, known_loader)
    unknown_z, _ = extract_features(model, unknown_loader)

    # Convert original CIFAR labels to the five-class label space.
    train_y = np.array([LABEL_MAP[int(y)] for y in train_y])
    known_y = np.array([LABEL_MAP[int(y)] for y in known_y])

    # Estimate the average feature vector for each known class.
    means = np.stack([
        train_z[train_y == c].mean(axis=0)
        for c in range(len(KNOWN))
    ])

    # Estimate shared within-class covariance.
    residuals = np.concatenate([
        train_z[train_y == c] - means[c]
        for c in range(len(KNOWN))
    ])

    covariance = np.cov(residuals, rowvar=False)

    # Regularization makes covariance inversion more stable.
    covariance += np.eye(covariance.shape[0]) * 0.01

    inverse_covariance = np.linalg.pinv(covariance)

    def distance_scores(features):
        # Return the smallest class-wise Mahalanobis distance.
        distances = []
        for mean in means:
            delta = features - mean
            d2 = np.einsum(
                "bi,ij,bj->b",
                delta, inverse_covariance, delta
            )
            distances.append(np.maximum(d2, 0))

        return np.min(np.stack(distances, axis=1), axis=1)

    # Fit the rejection threshold using known validation data only.
    val_distances = distance_scores(val_z)
    threshold = float(np.quantile(val_distances, 0.95))

    known_distances = distance_scores(known_z)
    unknown_distances = distance_scores(unknown_z)

    known_rejected = known_distances > threshold
    unknown_rejected = unknown_distances > threshold

    # AUROC: higher distance should indicate greater likelihood of OOD.
    scores = np.concatenate([known_distances, unknown_distances])
    ood_labels = np.concatenate([
        np.zeros(len(known_distances)),
        np.ones(len(unknown_distances)),
    ])

    print("\n=== MAHALANOBIS OOD DETECTION ===")
    print("Distance threshold:", round(threshold, 4))
    print("Known test rejection rate:",
          round(known_rejected.mean(), 4))
    print("Unknown detection recall:",
          round(unknown_rejected.mean(), 4))
    print("OOD AUROC:", round(roc_auc_score(ood_labels, scores), 4))

    output_dir = ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    np.savez(
        output_dir / "mahalanobis_results.npz",
        threshold=threshold,
        known_distances=known_distances,
        unknown_distances=unknown_distances,
    )
    print("Saved results to results/mahalanobis_results.npz")


if __name__ == "__main__":
    main()
