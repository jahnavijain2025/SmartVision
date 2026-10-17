
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.models import resnet18
from sklearn.metrics import roc_auc_score, average_precision_score


ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
TEMPERATURE = 10.0
EPSILON = 0.0014

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
KNOWN = ["airplane", "automobile", "bird", "cat", "deer"]
KNOWN_IDS = [CLASSES.index(c) for c in KNOWN]
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

        label = (
            LABEL_MAP[original_label]
            if self.known_only
            else int(original_label not in KNOWN_IDS)
        )
        return image, label


def get_logits(model, images):
    return model(images)


def collect_odin_scores(model, loader):
    scores, labels = [], []
    model.eval()

    mean = torch.tensor(MEAN, device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=DEVICE).view(1, 3, 1, 1)

    for images, batch_labels in loader:
        images = images.to(DEVICE).detach()
        images.requires_grad_(True)

        # First pass: temperature-scaled confidence.
        logits = get_logits(model, images)
        scaled_logits = logits / TEMPERATURE
        predicted = scaled_logits.argmax(dim=1)

        # ODIN perturbs the input to increase the predicted class's
        # temperature-scaled probability.
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(scaled_logits, predicted)
        loss.backward()

        # Input is ImageNet-normalized. Convert the pixel-space
        # epsilon into normalized-coordinate units.
        grad_sign = images.grad.sign()
        perturbation = (EPSILON / std) * grad_sign

        perturbed = (images - perturbation).detach()

        with torch.no_grad():
            perturbed_logits = get_logits(model, perturbed)
            scaled = perturbed_logits / TEMPERATURE

            # Higher score = more likely unknown.
            # Negative maximum softmax probability is the ODIN score.
            probabilities = F.softmax(scaled, dim=1)
            uncertainty = -probabilities.max(dim=1).values

        scores.append(uncertainty.cpu().numpy())
        labels.append(batch_labels.numpy())

    return np.concatenate(scores), np.concatenate(labels)


def collect_energy_scores(model, loader):
    scores, labels = [], []
    model.eval()

    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(DEVICE)
            logits = model(images)
            score = -torch.logsumexp(logits, dim=1)
            scores.append(score.cpu().numpy())
            labels.append(batch_labels.numpy())

    return np.concatenate(scores), np.concatenate(labels)


def evaluate(name, val_scores, known_scores, unknown_scores):
    # Threshold uses known validation images only.
    threshold = float(np.quantile(val_scores, 0.95, method="higher"))

    known_rejection = float(np.mean(known_scores > threshold))
    unknown_recall = float(np.mean(unknown_scores > threshold))

    all_scores = np.concatenate([known_scores, unknown_scores])
    targets = np.concatenate([
        np.zeros(len(known_scores), dtype=int),
        np.ones(len(unknown_scores), dtype=int),
    ])

    auroc = roc_auc_score(targets, all_scores)
    aupr = average_precision_score(targets, all_scores)

    print(f"\n{name}")
    print(f"  Validation threshold: {threshold:.6f}")
    print(f"  Known test rejection: {known_rejection:.4f}")
    print(f"  Unknown test recall:  {unknown_recall:.4f}")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  AUPR:  {aupr:.4f}")

    return {
        "threshold": threshold,
        "known_rejection": known_rejection,
        "unknown_recall": unknown_recall,
        "auroc": auroc,
        "aupr": aupr,
    }


def main():
    split_path = ROOT / "data" / "processed" / "dataset_splits.npz"
    data_dir = ROOT / "data" / "raw"
    checkpoint_path = ROOT / "models" / "known_classifier_resnet18.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if not split_path.exists():
        raise FileNotFoundError(split_path)

    splits = np.load(split_path)
    train_raw = CIFAR10(root=str(data_dir), train=True, download=False)
    test_raw = CIFAR10(root=str(data_dir), train=False, download=False)

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

    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    known_loader = DataLoader(
        known_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    unknown_loader = DataLoader(
        unknown_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    checkpoint = torch.load(
        checkpoint_path, map_location=DEVICE, weights_only=True
    )
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(KNOWN))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print("Device:", DEVICE)
    print("ODIN temperature:", TEMPERATURE)
    print("ODIN epsilon:", EPSILON)
    print("No retraining; checkpoint remains unchanged.")

    print("\nScoring known validation images...")
    val_odin, _ = collect_odin_scores(model, val_loader)
    val_energy, _ = collect_energy_scores(model, val_loader)

    print("Scoring known test images...")
    known_odin, known_labels = collect_odin_scores(model, known_loader)
    known_energy, _ = collect_energy_scores(model, known_loader)

    print("Scoring unknown test images...")
    unknown_odin, unknown_labels = collect_odin_scores(model, unknown_loader)
    unknown_energy, _ = collect_energy_scores(model, unknown_loader)

    odin_results = evaluate(
        "ODIN (temperature-scaled MSP uncertainty)",
        val_odin, known_odin, unknown_odin
    )
    energy_results = evaluate(
        "Energy baseline",
        val_energy, known_energy, unknown_energy
    )

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    np.savez(
        results_dir / "odin_results.npz",
        val_odin=val_odin,
        known_odin=known_odin,
        unknown_odin=unknown_odin,
        val_energy=val_energy,
        known_energy=known_energy,
        unknown_energy=unknown_energy,
        odin_auroc=odin_results["auroc"],
        odin_aupr=odin_results["aupr"],
        energy_auroc=energy_results["auroc"],
        energy_aupr=energy_results["aupr"],
        temperature=TEMPERATURE,
        epsilon=EPSILON,
    )
    print("\nSaved: results/odin_results.npz")


if __name__ == "__main__":
    main()
