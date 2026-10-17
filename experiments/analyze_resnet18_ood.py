
from pathlib import Path

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

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
KNOWN = ["airplane", "automobile", "bird", "cat", "deer"]
KNOWN_IDS = [CLASSES.index(c) for c in KNOWN]
KNOWN_MAP = {old: new for new, old in enumerate(KNOWN_IDS)}

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class ImageDataset(Dataset):
    def __init__(self, images, labels, indices):
        self.images = images
        self.labels = labels
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = int(self.indices[index])
        return TRANSFORM(self.images[i]), int(self.labels[i])


def get_scores(model, loader):
    logits_list, labels_list = [], []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            logits_list.append(model(images).cpu())
            labels_list.append(labels.numpy())

    logits = torch.cat(logits_list)
    labels = np.concatenate(labels_list)

    # Higher energy = more likely unknown.
    energy = -torch.logsumexp(logits, dim=1).numpy()

    # Higher score = less confident in the most likely known class.
    msp = torch.softmax(logits, dim=1).max(dim=1).values.numpy()
    msp_unknown_score = 1.0 - msp

    return logits.numpy(), labels, energy, msp_unknown_score


def metrics_for_detector(known_scores, unknown_scores, name):
    y_true = np.concatenate([
        np.zeros(len(known_scores), dtype=int),
        np.ones(len(unknown_scores), dtype=int),
    ])
    scores = np.concatenate([known_scores, unknown_scores])

    auroc = roc_auc_score(y_true, scores)
    aupr = average_precision_score(y_true, scores)

    # Threshold selected from known validation scores only.
    val_scores = VAL_SCORES[name]
    print(f"\n{name}")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  AUPR:  {aupr:.4f}")

    for target in [0.05, 0.10, 0.20, 0.30]:
        threshold = np.percentile(val_scores, 100 * (1 - target))
        known_rejection = np.mean(known_scores > threshold)
        unknown_recall = np.mean(unknown_scores > threshold)

        print(
            f"  Target rejection {target:.0%}: "
            f"actual known rejection={known_rejection:.4f}, "
            f"unknown recall={unknown_recall:.4f}"
        )

    return auroc, aupr


def main():
    split_file = ROOT / "data" / "processed" / "dataset_splits.npz"
    model_file = ROOT / "models" / "known_classifier_resnet18.pth"
    data_dir = ROOT / "data" / "raw"

    splits = np.load(split_file)

    train_raw = CIFAR10(
        root=str(data_dir), train=True, download=False
    )
    test_raw = CIFAR10(
        root=str(data_dir), train=False, download=False
    )

    val_loader = DataLoader(
        ImageDataset(
            train_raw.data, train_raw.targets,
            splits["val_indices"]
        ),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    known_loader = DataLoader(
        ImageDataset(
            test_raw.data, test_raw.targets,
            splits["known_test_indices"]
        ),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    unknown_loader = DataLoader(
        ImageDataset(
            test_raw.data, test_raw.targets,
            splits["unknown_test_indices"]
        ),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    checkpoint = torch.load(
        model_file, map_location=DEVICE, weights_only=True
    )
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(KNOWN))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)

    _, val_labels, val_energy, val_msp = get_scores(model, val_loader)
    known_logits, known_labels, known_energy, known_msp = get_scores(
        model, known_loader
    )
    unknown_logits, unknown_labels, unknown_energy, unknown_msp = get_scores(
        model, unknown_loader
    )

    known_mapped_labels = np.array([
        KNOWN_MAP[int(label)] for label in known_labels
    ])
    known_accuracy = np.mean(
        known_logits.argmax(axis=1) == known_mapped_labels
    )

    print("Device:", DEVICE)
    print(f"Known test accuracy: {known_accuracy:.4f}")
    print("Unknown test images:", len(unknown_labels))

    global VAL_SCORES
    VAL_SCORES = {
        "Energy": val_energy,
        "MSP": val_msp,
    }

    energy_metrics = metrics_for_detector(
        known_energy, unknown_energy, "Energy"
    )
    msp_metrics = metrics_for_detector(
        known_msp, unknown_msp, "MSP"
    )

    print("\nUNKNOWN CLASS BREAKDOWN")
    print("Recall uses each detector's validation-derived 5% threshold.")

    energy_threshold = np.percentile(val_energy, 95)
    msp_threshold = np.percentile(val_msp, 95)

    for class_id in range(len(KNOWN_IDS), len(CLASSES)):
        mask = unknown_labels == class_id
        class_energy = unknown_energy[mask]
        class_msp = unknown_msp[mask]

        energy_recall = np.mean(class_energy > energy_threshold)
        msp_recall = np.mean(class_msp > msp_threshold)

        energy_auc = roc_auc_score(
            np.concatenate([
                np.zeros(len(known_energy)),
                np.ones(len(class_energy)),
            ]),
            np.concatenate([known_energy, class_energy])
        )
        msp_auc = roc_auc_score(
            np.concatenate([
                np.zeros(len(known_msp)),
                np.ones(len(class_msp)),
            ]),
            np.concatenate([known_msp, class_msp])
        )

        print(
            f"{CLASSES[class_id]:10s} | "
            f"Energy recall={energy_recall:.4f}, AUROC={energy_auc:.4f} | "
            f"MSP recall={msp_recall:.4f}, AUROC={msp_auc:.4f}"
        )

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    np.savez(
        results_dir / "resnet18_msp_energy_comparison.npz",
        val_energy=val_energy,
        val_msp=val_msp,
        known_energy=known_energy,
        known_msp=known_msp,
        unknown_energy=unknown_energy,
        unknown_msp=unknown_msp,
        unknown_labels=unknown_labels,
        energy_auroc=energy_metrics[0],
        energy_aupr=energy_metrics[1],
        msp_auroc=msp_metrics[0],
        msp_aupr=msp_metrics[1],
    )

    print("\nSaved: results/resnet18_msp_energy_comparison.npz")


if __name__ == "__main__":
    VAL_SCORES = {}
    main()
