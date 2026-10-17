
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
BATCH_SIZE = 64

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


def make_loader(dataset):
    return DataLoader(
        dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0,
        pin_memory=torch.cuda.is_available()
    )


def collect(model, loader):
    features_all, logits_all, labels_all = [], [], []
    model.eval()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)

            # Exact ResNet-18 forward path up to avgpool.
            x = model.conv1(images)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)       # Important: preserve maxpool
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            features = model.avgpool(x)
            features = torch.flatten(features, 1)
            logits = model.fc(features)

            features_all.append(features.cpu().numpy())
            logits_all.append(logits.cpu().numpy())
            labels_all.append(labels.numpy())

    return (
        np.concatenate(features_all),
        np.concatenate(logits_all),
        np.concatenate(labels_all),
    )


def energy(logits):
    return -np.logaddexp.reduce(logits, axis=1)


def report(name, val_scores, known_scores, unknown_scores):
    # Threshold selected only using known validation scores.
    threshold = np.quantile(val_scores, 0.95, method="higher")
    known_rejection = np.mean(known_scores > threshold)
    unknown_recall = np.mean(unknown_scores > threshold)

    scores = np.concatenate([known_scores, unknown_scores])
    targets = np.concatenate([
        np.zeros(len(known_scores), dtype=int),
        np.ones(len(unknown_scores), dtype=int),
    ])

    print(f"\n{name}")
    print(f"  Validation threshold: {threshold:.6f}")
    print(f"  Known test rejection: {known_rejection:.4f}")
    print(f"  Unknown test recall:  {unknown_recall:.4f}")
    print(f"  AUROC: {roc_auc_score(targets, scores):.4f}")
    print(f"  AUPR:  {average_precision_score(targets, scores):.4f}")

    return threshold, known_rejection, unknown_recall


def main():
    split_path = ROOT / "data/processed/dataset_splits.npz"
    data_dir = ROOT / "data/raw"
    checkpoint_path = ROOT / "models/known_classifier_resnet18.pth"

    splits = np.load(split_path)
    train_raw = CIFAR10(root=str(data_dir), train=True, download=False)
    test_raw = CIFAR10(root=str(data_dir), train=False, download=False)

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

    checkpoint = torch.load(
        checkpoint_path, map_location=DEVICE, weights_only=True
    )
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(KNOWN))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print("Device:", DEVICE)
    print("Extracting features through the standard ResNet-18 path...")

    train_features, train_logits, _ = collect(model, make_loader(train_set))
    val_features, val_logits, _ = collect(model, make_loader(val_set))
    known_features, known_logits, known_labels = collect(
        model, make_loader(known_set)
    )
    unknown_features, unknown_logits, _ = collect(
        model, make_loader(unknown_set)
    )

    # Baseline energy, computed from the exact forward-path logits.
    base_val = energy(val_logits)
    base_known = energy(known_logits)
    base_unknown = energy(unknown_logits)

    # ReAct clipping value: one scalar estimated from training activations.
    # This is a simple baseline variant, not a guaranteed optimal setting.
    clip_value = float(np.percentile(train_features, 99.5))
    print(f"ReAct clipping value: {clip_value:.6f}")

    classifier_weight = model.fc.weight.detach().cpu().numpy()
    classifier_bias = model.fc.bias.detach().cpu().numpy()

    def react_energy(features):
        clipped_features = np.minimum(features, clip_value)
        logits = clipped_features @ classifier_weight.T + classifier_bias
        return energy(logits)

    react_val = react_energy(val_features)
    react_known = react_energy(known_features)
    react_unknown = react_energy(unknown_features)

    base_result = report(
        "Standard Energy", base_val, base_known, base_unknown
    )
    react_result = report(
        "ReAct + Energy", react_val, react_known, react_unknown
    )

    # Save to a new file; never overwrite the original baseline.
    out = ROOT / "results/react_energy_fixed_results.npz"
    np.savez(
        out,
        clip_value=clip_value,
        baseline_val_energy=base_val,
        baseline_known_energy=base_known,
        baseline_unknown_energy=base_unknown,
        react_val_energy=react_val,
        react_known_energy=react_known,
        react_unknown_energy=react_unknown,
        baseline_threshold=base_result[0],
        react_threshold=react_result[0],
    )

    print(f"\nSaved: {out}")
    print("No model was retrained or checkpoint modified.")


if __name__ == "__main__":
    main()
