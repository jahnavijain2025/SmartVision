
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms, models
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_BINS = 15

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
    def __init__(self, images, labels, indices):
        self.images = images
        self.labels = labels
        self.indices = np.asarray(indices)

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
        label = LABEL_MAP[original_label]
        return image, label


def load_logits(model, data_loader):
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(DEVICE)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels)

    return torch.cat(all_logits), torch.cat(all_labels).long()


def expected_calibration_error(probs, labels, num_bins=15):
    """Top-label ECE with equally spaced confidence bins."""
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1)
    edges = torch.linspace(0, 1, num_bins + 1)

    for i in range(num_bins):
        # Include confidence=1 in the final bin.
        if i == num_bins - 1:
            mask = (confidences >= edges[i]) & (
                confidences <= edges[i + 1]
            )
        else:
            mask = (confidences >= edges[i]) & (
                confidences < edges[i + 1]
            )

        if mask.any():
            bin_accuracy = accuracies[mask].float().mean()
            bin_confidence = confidences[mask].mean()
            ece += mask.float().mean() * torch.abs(
                bin_accuracy - bin_confidence
            )

    return ece.item()


def brier_score(probs, labels, num_classes=5):
    targets = F.one_hot(labels, num_classes=num_classes).float()
    # Multiclass Brier score: mean sum of squared class-probability errors.
    return torch.sum((probs - targets) ** 2, dim=1).mean().item()


def metrics(logits, labels, temperature=1.0):
    probs = torch.softmax(logits / temperature, dim=1)
    accuracy = (logits.argmax(dim=1) == labels).float().mean().item()

    return {
        "accuracy": accuracy,
        "ece": expected_calibration_error(probs, labels, NUM_BINS),
        "brier": brier_score(probs, labels),
    }


def fit_temperature(val_logits, val_labels):
    """Fit one scalar temperature using validation NLL only."""
    logits = val_logits.to(DEVICE)
    labels = val_labels.to(DEVICE)

    log_temperature = torch.nn.Parameter(
        torch.zeros(1, device=DEVICE)
    )
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=50,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(0.05, 20.0)
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    return float(
        torch.exp(log_temperature).clamp(0.05, 20.0).detach().cpu()
    )


def plot_reliability(ax, logits, labels, temperature, title):
    probs = torch.softmax(logits / temperature, dim=1)
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels).float()

    edges = torch.linspace(0, 1, NUM_BINS + 1)
    bin_confidences = []
    bin_accuracies = []

    for i in range(NUM_BINS):
        if i == NUM_BINS - 1:
            mask = (confidences >= edges[i]) & (
                confidences <= edges[i + 1]
            )
        else:
            mask = (confidences >= edges[i]) & (
                confidences < edges[i + 1]
            )

        if mask.any():
            bin_confidences.append(confidences[mask].mean().item())
            bin_accuracies.append(accuracies[mask].mean().item())

    ax.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    ax.plot(
        bin_confidences, bin_accuracies,
        marker="o", label=title
    )
    ax.set_title(title)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def main():
    split_path = ROOT / "data/processed/dataset_splits.npz"
    data_dir = ROOT / "data/raw"
    results_dir = ROOT / "results"
    plots_dir = results_dir / "plots"
    results_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_paths = {
        "Original ResNet-18": ROOT / "models/known_classifier_resnet18.pth",
        "LogitNorm ResNet-18": ROOT / "models/known_classifier_logitnorm.pth",
    }

    for path in checkpoint_paths.values():
        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}\n"
                "Check the model filename in the script."
            )

    splits = np.load(split_path)
    train_raw = CIFAR10(
        root=str(data_dir), train=True, download=False
    )
    test_raw = CIFAR10(
        root=str(data_dir), train=False, download=False
    )

    val_set = ImageDataset(
        train_raw.data, train_raw.targets, splits["val_indices"]
    )
    test_set = ImageDataset(
        test_raw.data, test_raw.targets, splits["known_test_indices"]
    )

    def make_loader(dataset):
        return DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    val_loader = make_loader(val_set)
    test_loader = make_loader(test_set)

    all_results = {}
    logits_by_model = {}
    labels_for_test = None
    val_logits_by_model = {}

    print("Device:", DEVICE)
    print("Loading checkpoints and collecting logits...")

    for name, path in checkpoint_paths.items():
        checkpoint = torch.load(
            path, map_location=DEVICE, weights_only=True
        )

        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(KNOWN))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(DEVICE)

        val_logits, val_labels = load_logits(model, val_loader)
        test_logits, test_labels = load_logits(model, test_loader)

        val_logits_by_model[name] = (val_logits, val_labels)
        logits_by_model[name] = test_logits
        labels_for_test = test_labels

        print(f"Loaded: {name}")

    # Fit temperature on validation data only, for LogitNorm.
    logitnorm_val_logits, val_labels = val_logits_by_model[
        "LogitNorm ResNet-18"
    ]
    temperature = fit_temperature(logitnorm_val_logits, val_labels)

    print(f"\nValidation-fitted temperature: {temperature:.4f}")

    # Evaluate all configurations on the same held-out known test set.
    configurations = [
        ("Original ResNet-18", "Original ResNet-18", 1.0),
        ("LogitNorm ResNet-18", "LogitNorm ResNet-18", 1.0),
        (
            "LogitNorm + Temperature Scaling",
            "LogitNorm ResNet-18",
            temperature,
        ),
    ]

    print("\nKNOWN TEST SET CALIBRATION RESULTS")
    print("-" * 76)
    print(f"{'Model':38s} {'Accuracy':>10s} {'ECE':>10s} {'Brier':>10s}")
    print("-" * 76)

    for display_name, source_name, temp in configurations:
        result = metrics(
            logits_by_model[source_name],
            labels_for_test,
            temperature=temp,
        )
        all_results[display_name] = result
        print(
            f"{display_name:38s} "
            f"{result['accuracy']:10.4f} "
            f"{result['ece']:10.4f} "
            f"{result['brier']:10.4f}"
        )

    # Reliability diagram.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    plot_reliability(
        axes[0],
        logits_by_model["Original ResNet-18"],
        labels_for_test,
        1.0,
        "Original ResNet-18",
    )
    plot_reliability(
        axes[1],
        logits_by_model["LogitNorm ResNet-18"],
        labels_for_test,
        1.0,
        "LogitNorm ResNet-18",
    )
    plot_reliability(
        axes[2],
        logits_by_model["LogitNorm ResNet-18"],
        labels_for_test,
        temperature,
        "LogitNorm + Temperature Scaling",
    )

    fig.suptitle("Confidence Calibration on Known-Class Test Set")
    fig.tight_layout()

    plot_path = plots_dir / "calibration_reliability_diagram.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Save machine-readable results.
    output_path = results_dir / "calibration_results.npz"
    np.savez(
        output_path,
        temperature=temperature,
        labels=labels_for_test.numpy(),
        **{
            f"{name}_logits": logits_by_model[source_name].numpy()
            for name, source_name, _ in configurations
        },
        **{
            f"{name}_accuracy": np.array(result["accuracy"])
            for name, result in all_results.items()
        },
        **{
            f"{name}_ece": np.array(result["ece"])
            for name, result in all_results.items()
        },
        **{
            f"{name}_brier": np.array(result["brier"])
            for name, result in all_results.items()
        },
    )

    print(f"\nSaved results: {output_path.relative_to(ROOT)}")
    print(f"Saved plot:    {plot_path.relative_to(ROOT)}")
    print("\nLower ECE and Brier scores indicate better calibration.")
    print("Accuracy is reported separately from calibration.")


if __name__ == "__main__":
    main()
