
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64

KNOWN = ["airplane", "automobile", "bird", "cat", "deer"]
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
KNOWN_IDS = [CLASSES.index(c) for c in KNOWN]
LABEL_MAP = {old: new for new, old in enumerate(KNOWN_IDS)}

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class ImageDataset(Dataset):
    def __init__(self, images, labels, indices, known_only=True):
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
            label = int(original_label not in KNOWN_IDS)

        return image, label


def collect(model, loader, calculate_accuracy=False):
    all_energy = []
    all_labels = []
    correct = 0
    count = 0

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            logits = model(images)

            energy = -torch.logsumexp(logits, dim=1)
            all_energy.append(energy.cpu().numpy())
            all_labels.append(labels.numpy())

            if calculate_accuracy:
                labels_device = labels.to(DEVICE)
                correct += (
                    logits.argmax(dim=1) == labels_device
                ).sum().item()
                count += labels.size(0)

    scores = np.concatenate(all_energy)
    labels = np.concatenate(all_labels)

    accuracy = correct / count if calculate_accuracy else None
    return scores, labels, accuracy


def main():
    split_path = ROOT / "data/processed/dataset_splits.npz"
    data_dir = ROOT / "data/raw"
    checkpoint_path = ROOT / "models/known_classifier_logitnorm.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    splits = np.load(split_path)
    train_raw = CIFAR10(
        root=str(data_dir), train=True, download=False
    )
    test_raw = CIFAR10(
        root=str(data_dir), train=False, download=False
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

    checkpoint = torch.load(
        checkpoint_path, map_location=DEVICE, weights_only=True
    )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(KNOWN))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)

    print("Device:", DEVICE)
    print("Checkpoint:", checkpoint_path.name)
    print("Best training-run validation accuracy:",
          checkpoint["best_val_accuracy"])
    print("Scoring validation and test sets...")

    val_energy, _, _ = collect(model, loader(val_set))
    known_energy, known_labels, known_accuracy = collect(
        model, loader(known_set), calculate_accuracy=True
    )
    unknown_energy, _, _ = collect(model, loader(unknown_set))

    # Calibrate only on known validation data.
    threshold = float(np.quantile(val_energy, 0.95, method="higher"))

    known_rejection = float(np.mean(known_energy > threshold))
    unknown_recall = float(np.mean(unknown_energy > threshold))

    scores = np.concatenate([known_energy, unknown_energy])
    targets = np.concatenate([
        np.zeros(len(known_energy), dtype=int),
        np.ones(len(unknown_energy), dtype=int),
    ])

    auroc = roc_auc_score(targets, scores)
    aupr = average_precision_score(targets, scores)

    print("\nLOGITNORM + ENERGY RESULTS")
    print(f"Known test accuracy:       {known_accuracy:.4f}")
    print(f"Energy AUROC:              {auroc:.4f}")
    print(f"Energy AUPR:               {aupr:.4f}")
    print(f"Validation threshold:      {threshold:.6f}")
    print(f"Known test rejection:      {known_rejection:.4f}")
    print(f"Unknown test recall:       {unknown_recall:.4f}")

    print("\nOperating-point comparison:")
    for target_rejection in [0.05, 0.10, 0.20, 0.30]:
        cutoff = float(np.quantile(
            val_energy, 1 - target_rejection, method="higher"
        ))
        kr = float(np.mean(known_energy > cutoff))
        ur = float(np.mean(unknown_energy > cutoff))
        print(
            f"Target={target_rejection:.0%} | "
            f"known rejection={kr:.4f} | "
            f"unknown recall={ur:.4f}"
        )

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    np.savez(
        results_dir / "logitnorm_ood_results.npz",
        val_energy=val_energy,
        known_energy=known_energy,
        unknown_energy=unknown_energy,
        known_accuracy=known_accuracy,
        threshold=threshold,
        auroc=auroc,
        aupr=aupr,
    )
    print("\nSaved: results/logitnorm_ood_results.npz")


if __name__ == "__main__":
    main()
