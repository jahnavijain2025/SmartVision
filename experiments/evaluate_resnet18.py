

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.models import resnet18

from sklearn.metrics import roc_auc_score, average_precision_score


ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KNOWN = ["airplane", "automobile", "bird", "cat", "deer"]
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
CLASS_IDS = [CLASSES.index(c) for c in KNOWN]
LABEL_MAP = {old: new for new, old in enumerate(CLASS_IDS)}

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
BATCH_SIZE = 64


class ImageDataset(Dataset):
    def __init__(self, images, labels, indices, known_only=False):
        self.images = images
        self.labels = labels
        self.indices = np.asarray(indices)
        self.known_only = known_only

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = int(self.indices[index])
        image = self.transform(self.images[i])
        original_label = int(self.labels[i])

        if self.known_only:
            label = LABEL_MAP[original_label]
        else:
            # 1 = unknown, 0 = known
            label = int(original_label not in CLASS_IDS)

        return image, label


def collect_scores(model, loader, known_only):
    all_logits = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            logits = model(images)

            all_logits.append(logits.cpu())
            all_labels.append(labels)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()

    # Higher energy means more likely to be unknown.
    energy = -torch.logsumexp(logits, dim=1).numpy()

    if known_only:
        accuracy = (logits.argmax(dim=1).numpy() == labels).mean()
        return logits.numpy(), labels, energy, accuracy

    return logits.numpy(), labels, energy, None


def main():
    split_file = ROOT / "data" / "processed" / "dataset_splits.npz"
    data_dir = ROOT / "data" / "raw"
    checkpoint_file = ROOT / "models" / "known_classifier_resnet18.pth"

    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Missing model: {checkpoint_file}")

    splits = np.load(split_file)

    test_raw = CIFAR10(
        root=str(data_dir), train=False, download=False
    )

    known_set = ImageDataset(
        test_raw.data, test_raw.targets,
        splits["known_test_indices"], known_only=True
    )
    unknown_set = ImageDataset(
        test_raw.data, test_raw.targets,
        splits["unknown_test_indices"], known_only=False
    )

    known_loader = DataLoader(
        known_set, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0
    )
    unknown_loader = DataLoader(
        unknown_set, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0
    )

    checkpoint = torch.load(
        checkpoint_file, map_location=DEVICE, weights_only=True
    )

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(KNOWN))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)

    print("Device:", DEVICE)
    print("Best validation accuracy:", checkpoint["best_val_accuracy"])

    known_logits, known_labels, known_energy, known_acc = collect_scores(
        model, known_loader, known_only=True
    )
    unknown_logits, unknown_labels, unknown_energy, _ = collect_scores(
        model, unknown_loader, known_only=False
    )

    print(f"\nKnown test accuracy: {known_acc:.4f}")
    print(f"Known test images: {len(known_labels)}")
    print(f"Unknown test images: {len(unknown_labels)}")

    # Select threshold using ONLY known validation data.
    train_raw = CIFAR10(
        root=str(data_dir), train=True, download=False
    )
    val_set = ImageDataset(
        train_raw.data, train_raw.targets,
        splits["val_indices"], known_only=True
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0
    )

    _, _, val_energy, _ = collect_scores(
        model, val_loader, known_only=True
    )

    # A high energy indicates a possible unknown.
    threshold = float(np.percentile(val_energy, 95))

    known_reject_rate = float((known_energy > threshold).mean())
    unknown_recall = float((unknown_energy > threshold).mean())

    # AUROC/AUPR: higher energy should indicate unknown.
    scores = np.concatenate([known_energy, unknown_energy])
    targets = np.concatenate([
        np.zeros(len(known_energy), dtype=int),
        np.ones(len(unknown_energy), dtype=int),
    ])

    auroc = roc_auc_score(targets, scores)
    aupr = average_precision_score(targets, scores)

    print(f"\nValidation-derived energy threshold: {threshold:.4f}")
    print(f"Known test rejection rate: {known_reject_rate:.4f}")
    print(f"Unknown test recall: {unknown_recall:.4f}")
    print(f"Energy AUROC: {auroc:.4f}")
    print(f"Energy AUPR: {aupr:.4f}")

    # Evaluate the full trade-off without tuning on test data.
    for target_rejection in [0.05, 0.10, 0.20, 0.30]:
        cutoff = float(
            np.percentile(val_energy, 100 * (1 - target_rejection))
        )
        kr = float((known_energy > cutoff).mean())
        ur = float((unknown_energy > cutoff).mean())

        print(
            f"Target known rejection={target_rejection:.0%} | "
            f"actual known rejection={kr:.4f} | "
            f"unknown recall={ur:.4f}"
        )

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    np.savez(
        results_dir / "resnet18_ood_results.npz",
        known_energy=known_energy,
        unknown_energy=unknown_energy,
        val_energy=val_energy,
        threshold=threshold,
        known_accuracy=known_acc,
        auroc=auroc,
        aupr=aupr,
    )
    print("\nSaved: results/resnet18_ood_results.npz")


if __name__ == "__main__":
    main()
