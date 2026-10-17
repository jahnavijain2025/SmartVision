
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from sklearn.metrics import roc_auc_score, average_precision_score


ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
K_VALUES = [10, 50, 200]

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
KNOWN = ["airplane", "automobile", "bird", "cat", "deer"]
KNOWN_IDS = [CLASSES.index(c) for c in KNOWN]
LABEL_MAP = {old: new for new, old in enumerate(KNOWN_IDS)}

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class ImageDataset(Dataset):
    def __init__(self, images, labels, indices, known_only=False):
        self.images = images
        self.labels = labels
        self.indices = np.asarray(indices)
        self.known_only = known_only

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = int(self.indices[index])
        image = TRANSFORM(self.images[i])
        original_label = int(self.labels[i])

        if self.known_only:
            label = LABEL_MAP[original_label]
        else:
            label = int(original_label not in KNOWN_IDS)

        return image, label


def extract_features(model, loader, description):
    features = []
    labels = []
    model.eval()

    start = time.time()
    with torch.no_grad():
        for batch_idx, (images, batch_labels) in enumerate(loader):
            images = images.to(DEVICE, non_blocking=True)
            feats = model(images)

            features.append(feats.cpu().numpy())
            labels.append(batch_labels.numpy())

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"{description}: "
                    f"{batch_idx + 1}/{len(loader)} batches"
                )

    features = np.concatenate(features).astype(np.float32)
    labels = np.concatenate(labels)

    print(
        f"{description}: extracted {len(labels)} feature vectors "
        f"in {(time.time() - start) / 60:.1f} minutes"
    )
    return features, labels


def normalize_features(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-12)


def knn_scores(reference_features, query_features, k):
    """Higher cosine distance to the k-th neighbor = more unknown."""
    reference = torch.from_numpy(
        normalize_features(reference_features)
    ).to(DEVICE)

    query = torch.from_numpy(
        normalize_features(query_features)
    ).to(DEVICE)

    all_scores = []

    with torch.no_grad():
        for start in range(0, len(query), 128):
            batch = query[start:start + 128]

            # Cosine similarity because features are L2-normalized.
            similarities = batch @ reference.T

            # The k-th nearest neighbor has the k-th largest similarity.
            topk = torch.topk(
                similarities, k=k, dim=1, largest=True
            ).values

            kth_similarity = topk[:, -1]
            distance = 1.0 - kth_similarity

            all_scores.append(distance.cpu().numpy())

    return np.concatenate(all_scores)


def fit_mahalanobis(train_features, train_labels):
    """
    Estimate class means and a shared covariance of within-class
    residuals. Regularize the covariance for numerical stability.
    """
    means = np.stack([
        train_features[train_labels == c].mean(axis=0)
        for c in range(len(KNOWN))
    ])

    residuals = np.stack([
        train_features[i] - means[train_labels[i]]
        for i in range(len(train_features))
    ])

    # Shrinkage gives a more stable covariance estimate than
    # directly inverting a noisy covariance matrix.
    covariance = np.cov(residuals, rowvar=False)
    shrinkage = 0.05
    diagonal = np.diag(np.diag(covariance))
    covariance = (
        (1 - shrinkage) * covariance
        + shrinkage * diagonal
        + 1e-4 * np.eye(covariance.shape[0])
    )

    precision = np.linalg.pinv(covariance)
    return means, precision


def mahalanobis_scores(features, means, precision):
    """Higher minimum class-conditional distance = more unknown."""
    scores = []

    for mean in means:
        diff = features - mean
        distances = np.einsum(
            "bi,ij,bj->b", diff, precision, diff
        )
        scores.append(distances)

    return np.min(np.stack(scores, axis=1), axis=1)


def report_method(name, val_scores, known_scores, unknown_scores):
    targets = np.concatenate([
        np.zeros(len(known_scores), dtype=int),
        np.ones(len(unknown_scores), dtype=int),
    ])
    test_scores = np.concatenate([known_scores, unknown_scores])

    auroc = roc_auc_score(targets, test_scores)
    aupr = average_precision_score(targets, test_scores)

    print(f"\n{name}")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  AUPR:  {aupr:.4f}")

    for target in [0.05, 0.10, 0.20, 0.30]:
        threshold = np.percentile(
            val_scores, 100 * (1 - target)
        )
        known_rejection = np.mean(known_scores > threshold)
        unknown_recall = np.mean(unknown_scores > threshold)

        print(
            f"  Target rejection {target:.0%}: "
            f"actual known rejection={known_rejection:.4f}, "
            f"unknown recall={unknown_recall:.4f}"
        )

    return {
        "auroc": auroc,
        "aupr": aupr,
        "val_scores": val_scores,
        "known_scores": known_scores,
        "unknown_scores": unknown_scores,
    }


def main():
    split_file = ROOT / "data" / "processed" / "dataset_splits.npz"
    model_file = ROOT / "models" / "known_classifier_resnet18.pth"
    data_dir = ROOT / "data" / "raw"

    if not split_file.exists():
        raise FileNotFoundError(f"Missing splits: {split_file}")
    if not model_file.exists():
        raise FileNotFoundError(f"Missing checkpoint: {model_file}")

    splits = np.load(split_file)

    train_raw = CIFAR10(
        root=str(data_dir), train=True, download=False
    )
    test_raw = CIFAR10(
        root=str(data_dir), train=False, download=False
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
            dataset, batch_size=BATCH_SIZE,
            shuffle=False, num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

    print("Device:", DEVICE)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    checkpoint = torch.load(
        model_file, map_location=DEVICE, weights_only=True
    )

    # Load the exact trained ResNet-18, then expose its 512-D
    # penultimate representation instead of its five-class logits.
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(KNOWN))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.fc = nn.Identity()
    model.to(DEVICE)

    train_f, train_y = extract_features(
        model, loader(train_set), "Known train"
    )
    val_f, _ = extract_features(
        model, loader(val_set), "Known validation"
    )
    known_f, _ = extract_features(
        model, loader(known_set), "Known test"
    )
    unknown_f, _ = extract_features(
        model, loader(unknown_set), "Unknown test"
    )

    results = {}

    # Energy remains our baseline. Recover its values from the
    # existing saved evaluation instead of recalculating logits.
    baseline_file = ROOT / "results" / "resnet18_msp_energy_comparison.npz"
    if baseline_file.exists():
        baseline = np.load(baseline_file)
        results["Energy"] = {
            "auroc": float(baseline["energy_auroc"])
            if "energy_auroc" in baseline.files else np.nan,
            "aupr": float(baseline["energy_aupr"])
            if "energy_aupr" in baseline.files else np.nan,
        }

    # kNN reference is the known training feature set only.
    # Query thresholds are selected from known validation features.
    for k in K_VALUES:
        print(f"\nCalculating kNN scores (k={k})...")
        val_scores = knn_scores(train_f, val_f, k)
        known_scores = knn_scores(train_f, known_f, k)
        unknown_scores = knn_scores(train_f, unknown_f, k)

        results[f"kNN_{k}"] = report_method(
            f"kNN (k={k})",
            val_scores, known_scores, unknown_scores
        )

    print("\nFitting Mahalanobis model...")
    means, precision = fit_mahalanobis(train_f, train_y)

    maha_val = mahalanobis_scores(val_f, means, precision)
    maha_known = mahalanobis_scores(known_f, means, precision)
    maha_unknown = mahalanobis_scores(unknown_f, means, precision)

    results["Mahalanobis"] = report_method(
        "Mahalanobis distance",
        maha_val, maha_known, maha_unknown
    )

    output = ROOT / "results" / "resnet18_feature_ood_results.npz"
    save_data = {}

    for name, result in results.items():
        for key in ["auroc", "aupr"]:
            if key in result:
                save_data[f"{name}_{key}"] = result[key]

        for key in ["val_scores", "known_scores", "unknown_scores"]:
            if key in result:
                save_data[f"{name}_{key}"] = result[key]

    save_data["train_features"] = train_f
    save_data["train_labels"] = train_y
    save_data["val_features"] = val_f
    save_data["known_test_features"] = known_f
    save_data["unknown_test_features"] = unknown_f

    np.savez_compressed(output, **save_data)

    print("\nSaved:", output)
    print("Feature analysis complete. No model was retrained.")


if __name__ == "__main__":
    main()
