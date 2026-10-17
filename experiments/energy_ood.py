
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

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
KNOWN = ["airplane", "automobile", "bird", "cat", "deer"]
CLASS_IDS = [CLASSES.index(c) for c in KNOWN]
LABEL_MAP = {old: new for new, old in enumerate(CLASS_IDS)}
BATCH_SIZE = 128


class ImageDataset(Dataset):
    def __init__(self, images, labels, indices):
        self.images = images
        self.labels = labels
        self.indices = indices
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.247, 0.243, 0.261)
            )
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = int(self.indices[index])
        image = self.transform(self.images[i])
        return image, int(self.labels[i])


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
        x = self.features(x)
        return self.classifier(x.flatten(1))


def get_logits(model, loader):
    all_logits = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(DEVICE))
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_logits), np.concatenate(all_labels)


def energy_score(logits, temperature=1.0):
    """
    Negative log-sum-exp energy.
    Higher scores indicate less compatibility with known classes.
    """
    logits = torch.as_tensor(logits, dtype=torch.float64)
    return (
        -temperature
        * torch.logsumexp(logits / temperature, dim=1)
    ).numpy()


def main():
    split_file = ROOT / "data" / "processed" / "dataset_splits.npz"
    model_file = ROOT / "models" / "known_classifier.pth"
    data_dir = ROOT / "data" / "raw"

    if not split_file.exists() or not model_file.exists():
        raise FileNotFoundError(
            "Dataset splits or model missing. "
            "Run prepare_dataset.py and train.py first."
        )

    splits = np.load(split_file)
    checkpoint = torch.load(
        model_file, map_location=DEVICE, weights_only=True
    )

    if checkpoint["known_classes"] != KNOWN:
        raise ValueError("Model classes do not match.")

    model = SmallCNN().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    train_data = CIFAR10(
        root=str(data_dir), train=True, download=False
    )
    test_data = CIFAR10(
        root=str(data_dir), train=False, download=False
    )

    def make_loader(images, labels, indices):
        dataset = ImageDataset(images, labels, indices)
        return DataLoader(
            dataset, batch_size=BATCH_SIZE,
            shuffle=False, num_workers=0
        )

    val_loader = make_loader(
        train_data.data, train_data.targets,
        splits["val_indices"]
    )
    known_loader = make_loader(
        test_data.data, test_data.targets,
        splits["known_test_indices"]
    )
    unknown_loader = make_loader(
        test_data.data, test_data.targets,
        splits["unknown_test_indices"]
    )

    print("Extracting logits...")
    val_logits, _ = get_logits(model, val_loader)
    known_logits, _ = get_logits(model, known_loader)
    unknown_logits, _ = get_logits(model, unknown_loader)

    val_scores = energy_score(val_logits)
    known_scores = energy_score(known_logits)
    unknown_scores = energy_score(unknown_logits)

    # Choose threshold from known validation data only.
    # Higher energy indicates a more likely unknown input.
    threshold = float(np.quantile(val_scores, 0.95))

    known_rejected = known_scores > threshold
    unknown_rejected = unknown_scores > threshold

    labels = np.concatenate([
        np.zeros(len(known_scores)),
        np.ones(len(unknown_scores))
    ])
    scores = np.concatenate([known_scores, unknown_scores])
    auc = roc_auc_score(labels, scores)

    print("\n=== ENERGY-BASED OOD DETECTION ===")
    print("Device:", DEVICE)
    print("Energy threshold:", round(threshold, 4))
    print("Known test rejection rate:",
          round(known_rejected.mean(), 4))
    print("Unknown detection recall:",
          round(unknown_rejected.mean(), 4))
    print("OOD AUROC:", round(auc, 4))

    output_dir = ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    np.savez(
        output_dir / "energy_results.npz",
        threshold=threshold,
        known_scores=known_scores,
        unknown_scores=unknown_scores,
        auroc=auc,
    )

    print("Saved results to results/energy_results.npz")


if __name__ == "__main__":
    main()
