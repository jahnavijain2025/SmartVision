
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
    def __init__(self, images, labels, indices, known_labels=False):
        self.images = images
        self.labels = labels
        self.indices = np.asarray(indices)
        self.known_labels = known_labels
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

        if self.known_labels:
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
            out_channels, out_channels, 3, padding=1, bias=False
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


def get_scores(model, loader):
    logits_list = []
    labels_list = []
    model.eval()

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(DEVICE, non_blocking=True))
            logits_list.append(logits.cpu())
            labels_list.append(labels)

    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list).numpy()

    probabilities = torch.softmax(logits, dim=1)
    msp = probabilities.max(dim=1).values.numpy()
    energy = -torch.logsumexp(logits, dim=1).numpy()
    predictions = logits.argmax(dim=1).numpy()

    return msp, energy, predictions, labels


def main():
    split_path = ROOT / "data" / "processed" / "dataset_splits.npz"
    checkpoint_path = ROOT / "models" / "known_classifier_augmented.pth"

    if not split_path.exists():
        raise FileNotFoundError(f"Dataset splits not found: {split_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    splits = np.load(split_path)

    train_raw = CIFAR10(
        root=str(ROOT / "data" / "raw"),
        train=True, download=False
    )
    test_raw = CIFAR10(
        root=str(ROOT / "data" / "raw"),
        train=False, download=False
    )

    val_set = ImageDataset(
        train_raw.data, train_raw.targets,
        splits["val_indices"], known_labels=True
    )
    known_set = ImageDataset(
        test_raw.data, test_raw.targets,
        splits["known_test_indices"], known_labels=True
    )
    unknown_set = ImageDataset(
        test_raw.data, test_raw.targets,
        splits["unknown_test_indices"], known_labels=False
    )

    def make_loader(dataset):
        return DataLoader(
            dataset, batch_size=256, shuffle=False,
            num_workers=0, pin_memory=torch.cuda.is_available()
        )

    checkpoint = torch.load(
        checkpoint_path, map_location=DEVICE, weights_only=True
    )
    if checkpoint["known_classes"] != KNOWN:
        raise ValueError("Checkpoint known classes do not match.")

    model = ImprovedCNN().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Device:", DEVICE)
    print("Evaluating augmented model...")

    val_msp, val_energy, _, _ = get_scores(model, make_loader(val_set))
    known_msp, known_energy, known_preds, known_labels = get_scores(
        model, make_loader(known_set)
    )
    unknown_msp, unknown_energy, _, unknown_labels = get_scores(
        model, make_loader(unknown_set)
    )

    # AUROC/AUPR: higher anomaly score should indicate an unknown image.
    y_true = np.concatenate([
        np.zeros(len(known_msp), dtype=int),
        np.ones(len(unknown_msp), dtype=int),
    ])
    msp_anomaly = np.concatenate([-known_msp, -unknown_msp])
    energy_anomaly = np.concatenate([known_energy, unknown_energy])

    msp_auc = roc_auc_score(y_true, msp_anomaly)
    energy_auc = roc_auc_score(y_true, energy_anomaly)
    msp_aupr = average_precision_score(y_true, msp_anomaly)
    energy_aupr = average_precision_score(y_true, energy_anomaly)

    print("\n=== AUGMENTED MODEL: OPEN-SET EVALUATION ===")
    print("Known test images:", len(known_msp))
    print("Unknown test images:", len(unknown_msp))
    print(f"Known-test classification accuracy: "
          f"{np.mean(known_preds == known_labels):.4f}")

    print("\n--- Overall ranking metrics ---")
    print(f"MSP OOD AUROC: {msp_auc:.4f}")
    print(f"MSP OOD AUPR:  {msp_aupr:.4f}")
    print(f"Energy OOD AUROC: {energy_auc:.4f}")
    print(f"Energy OOD AUPR:  {energy_aupr:.4f}")

    # Select thresholds using known validation scores ONLY.
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    threshold_rows = []
    for target_rejection in (5, 10, 20, 30):
        # Use the upper tail of validation energy scores.
        energy_threshold = float(
            np.percentile(val_energy, 100 - target_rejection)
        )
        msp_threshold = float(
            np.percentile(val_msp, target_rejection)
        )

        energy_known_reject = float(
            np.mean(known_energy > energy_threshold)
        )
        energy_unknown_recall = float(
            np.mean(unknown_energy > energy_threshold)
        )
        msp_known_reject = float(
            np.mean(known_msp < msp_threshold)
        )
        msp_unknown_recall = float(
            np.mean(unknown_msp < msp_threshold)
        )

        threshold_rows.append([
            target_rejection,
            energy_threshold,
            energy_known_reject,
            energy_unknown_recall,
            msp_threshold,
            msp_known_reject,
            msp_unknown_recall,
        ])

        print(
            f"\nTarget known rejection: {target_rejection}%"
            f"\n  Energy threshold: {energy_threshold:.4f}"
            f" | actual known rejection: {energy_known_reject:.4f}"
            f" | unknown recall: {energy_unknown_recall:.4f}"
            f"\n  MSP threshold: {msp_threshold:.4f}"
            f" | actual known rejection: {msp_known_reject:.4f}"
            f" | unknown recall: {msp_unknown_recall:.4f}"
        )

    # Also save the conventional 5% validation thresholds and all scores.
    np.savez(
        out_dir / "augmented_ood_results.npz",
        val_msp=val_msp,
        val_energy=val_energy,
        known_msp=known_msp,
        unknown_msp=unknown_msp,
        known_energy=known_energy,
        unknown_energy=unknown_energy,
        known_labels=known_labels,
        known_predictions=known_preds,
        unknown_original_labels=unknown_labels,
        msp_auroc=np.asarray(msp_auc),
        energy_auroc=np.asarray(energy_auc),
        msp_aupr=np.asarray(msp_aupr),
        energy_aupr=np.asarray(energy_aupr),
        threshold_comparison=np.asarray(threshold_rows),
    )

    print("\nSaved results to results/augmented_ood_results.npz")


if __name__ == "__main__":
    main()
