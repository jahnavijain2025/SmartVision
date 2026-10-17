
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64

KNOWN = ["airplane", "automobile", "bird", "cat", "deer"]
ALL_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
KNOWN_IDS = [ALL_CLASSES.index(c) for c in KNOWN]
LABEL_MAP = {old: new for new, old in enumerate(KNOWN_IDS)}

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class ImageDataset(Dataset):
    def __init__(self, images, labels, indices, known_only):
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


def collect(model, loader):
    all_logits, all_labels = [], []
    model.eval()

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(DEVICE, non_blocking=True))
            all_logits.append(logits.cpu())
            all_labels.append(labels)

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    energy = -np.logaddexp.reduce(logits, axis=1)
    predictions = logits.argmax(axis=1)

    return energy, predictions, labels


def make_loader(dataset):
    return DataLoader(
        dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0
    )


def main():
    split_path = ROOT / "data/processed/dataset_splits.npz"
    data_dir = ROOT / "data/raw"
    checkpoint_path = ROOT / "models/known_classifier_resnet18.pth"

    splits = np.load(split_path)
    train_raw = CIFAR10(
        root=str(data_dir), train=True, download=False
    )
    test_raw = CIFAR10(
        root=str(data_dir), train=False, download=False
    )

    val_loader = make_loader(ImageDataset(
        train_raw.data, train_raw.targets,
        splits["val_indices"], known_only=True
    ))
    known_loader = make_loader(ImageDataset(
        test_raw.data, test_raw.targets,
        splits["known_test_indices"], known_only=True
    ))
    unknown_loader = make_loader(ImageDataset(
        test_raw.data, test_raw.targets,
        splits["unknown_test_indices"], known_only=False
    ))

    checkpoint = torch.load(
        checkpoint_path, map_location=DEVICE, weights_only=True
    )
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(KNOWN))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)

    val_energy, val_pred, _ = collect(model, val_loader)
    known_energy, known_pred, known_labels = collect(model, known_loader)
    unknown_energy, unknown_pred, _ = collect(model, unknown_loader)

    # Baseline: one global threshold, calibrated on known validation images.
    global_threshold = np.quantile(val_energy, 0.95, method="higher")

    # Proposed method: one threshold per predicted class.
    # Each threshold uses only known validation images predicted as that class.
    class_thresholds = np.empty(len(KNOWN), dtype=float)

    print("Validation counts by predicted class:")
    for c, name in enumerate(KNOWN):
        scores = val_energy[val_pred == c]
        if len(scores) < 20:
            raise RuntimeError(
                f"Too few validation predictions for {name}: {len(scores)}"
            )
        class_thresholds[c] = np.quantile(
            scores, 0.95, method="higher"
        )
        print(
            f"  {name:10s}: n={len(scores):4d}, "
            f"threshold={class_thresholds[c]:.4f}"
        )

    # Reject if energy exceeds the relevant threshold.
    global_known_reject = known_energy > global_threshold
    global_unknown_reject = unknown_energy > global_threshold

    conditional_known_reject = (
        known_energy > class_thresholds[known_pred]
    )
    conditional_unknown_reject = (
        unknown_energy > class_thresholds[unknown_pred]
    )

    print("\nTest comparison")
    print("-" * 65)
    print(
        f"Global Energy:             known rejection="
        f"{global_known_reject.mean():.4f}, "
        f"unknown recall={global_unknown_reject.mean():.4f}"
    )
    print(
        f"Class-conditional Energy:  known rejection="
        f"{conditional_known_reject.mean():.4f}, "
        f"unknown recall={conditional_unknown_reject.mean():.4f}"
    )

    print("\nPer-class unknown recall using class-conditional thresholds:")
    unknown_labels_original = test_raw.targets
    unknown_indices = splits["unknown_test_indices"]

    original_labels = np.asarray(unknown_labels_original)[unknown_indices]
    for original_id in sorted(set(original_labels)):
        mask = original_labels == original_id
        recall = conditional_unknown_reject[mask].mean()
        print(
            f"  {ALL_CLASSES[original_id]:10s}: "
            f"{recall:.4f} ({recall * 100:.1f}%)"
        )

    print("\nKnown test accuracy (unchanged by rejection):",
          f"{(known_pred == known_labels).mean():.4f}")
    print("No model was retrained or checkpoint modified.")


if __name__ == "__main__":
    main()
