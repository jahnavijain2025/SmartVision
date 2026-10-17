
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALL_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
KNOWN = ["airplane", "automobile", "bird", "cat", "deer"]
KNOWN_IDS = [ALL_CLASSES.index(c) for c in KNOWN]
LABEL_MAP = {old: new for new, old in enumerate(KNOWN_IDS)}


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


class EvalDataset(Dataset):
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
        image = self.transform(self.images[i])
        original_label = int(self.labels[i])
        return image, original_label


def collect_outputs(model, loader):
    scores, labels = [], []
    model.eval()

    with torch.no_grad():
        for images, batch_labels in loader:
            logits = model(images.to(DEVICE))
            probabilities = torch.softmax(logits, dim=1)
            scores.append(probabilities.cpu().numpy())
            labels.append(batch_labels.numpy())

    return np.concatenate(scores), np.concatenate(labels)


def main():
    split_path = ROOT / "data" / "processed" / "dataset_splits.npz"
    checkpoint_path = ROOT / "models" / "known_classifier.pth"

    if not split_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(
            "Dataset splits or trained model missing. "
            "Run prepare_dataset.py and train.py first."
        )

    splits = np.load(split_path)
    checkpoint = torch.load(
        checkpoint_path, map_location=DEVICE, weights_only=True
    )

    if checkpoint["known_classes"] != KNOWN:
        raise ValueError("Model classes do not match evaluation classes.")

    model = SmallCNN().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    train_data = CIFAR10(
        root=str(ROOT / "data" / "raw"),
        train=True, download=False
    )
    test_data = CIFAR10(
        root=str(ROOT / "data" / "raw"),
        train=False, download=False
    )

    # Validation images come only from known categories.
    val_loader = DataLoader(
        EvalDataset(
            train_data.data,
            train_data.targets,
            splits["val_indices"],
        ),
        batch_size=128, shuffle=False,
    )

    known_loader = DataLoader(
        EvalDataset(
            test_data.data,
            test_data.targets,
            splits["known_test_indices"],
        ),
        batch_size=128, shuffle=False,
    )

    unknown_loader = DataLoader(
        EvalDataset(
            test_data.data,
            test_data.targets,
            splits["unknown_test_indices"],
        ),
        batch_size=128, shuffle=False,
    )

    val_probs, val_labels = collect_outputs(model, val_loader)
    known_probs, known_labels = collect_outputs(model, known_loader)
    unknown_probs, unknown_labels = collect_outputs(model, unknown_loader)

    # Convert labels into the model's five-class label space.
    known_targets = np.array([LABEL_MAP[int(y)] for y in known_labels])
    known_predictions = known_probs.argmax(axis=1)

    print("\n=== NORMAL CLASSIFIER: KNOWN TEST ===")
    print("Known-class accuracy:",
          round(accuracy_score(known_targets, known_predictions), 4))
    print(classification_report(
        known_targets, known_predictions,
        labels=list(range(5)), target_names=KNOWN, zero_division=0
    ))

    # Set a threshold using only known validation images.
    # This targets approximately 5% validation rejection.
    val_msp = val_probs.max(axis=1)
    threshold = float(np.quantile(val_msp, 0.05))

    known_msp = known_probs.max(axis=1)
    unknown_msp = unknown_probs.max(axis=1)

    known_accepted = known_msp >= threshold
    unknown_rejected = unknown_msp < threshold

    known_accepted_accuracy = (
        accuracy_score(
            known_targets[known_accepted],
            known_predictions[known_accepted],
        )
        if known_accepted.any() else float("nan")
    )

    print("\n=== UNCERTAINTY-AWARE CLASSIFIER ===")
    print("Threshold:", round(threshold, 4))
    print("Known test accuracy before rejection:",
          round(accuracy_score(known_targets, known_predictions), 4))
    print("Known test rejection rate:",
          round(1 - known_accepted.mean(), 4))
    print("Accuracy on accepted known images:",
          round(known_accepted_accuracy, 4))
    print("Unknown detection recall:",
          round(unknown_rejected.mean(), 4))

    # Higher score means more likely to be unknown.
    all_msp = np.concatenate([known_msp, unknown_msp])
    ood_labels = np.concatenate([
        np.zeros(len(known_msp)),
        np.ones(len(unknown_msp)),
    ])

    print("OOD AUROC:",
          round(roc_auc_score(ood_labels, -all_msp), 4))

    output_dir = ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    np.savez(
        output_dir / "evaluation_results.npz",
        threshold=threshold,
        known_msp=known_msp,
        unknown_msp=unknown_msp,
        known_targets=known_targets,
        known_predictions=known_predictions,
        unknown_labels=unknown_labels,
    )

    print("\nResults saved to results/evaluation_results.npz")


if __name__ == "__main__":
    main()
