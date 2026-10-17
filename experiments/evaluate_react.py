
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

        if self.known_only:
            label = LABEL_MAP[original_label]
        else:
            label = int(original_label not in KNOWN_IDS)

        return image, label


def make_loader(dataset):
    return DataLoader(
        dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0
    )


def collect_features(model, loader):
    features_all = []
    labels_all = []
    logits_all = []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)

            # ResNet-18 penultimate features: shape [batch, 512].
            features = model.avgpool(
                model.layer4(
                    model.layer3(
                        model.layer2(
                            model.layer1(model.relu(model.bn1(model.conv1(images))))
                        )
                    )
                )
            )
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


def energy_from_logits(logits):
    # Higher energy means more likely unknown.
    return -np.logaddexp.reduce(logits, axis=1)


def main():
    split_file = ROOT / "data/processed/dataset_splits.npz"
    data_dir = ROOT / "data/raw"
    checkpoint_file = ROOT / "models/known_classifier_resnet18.pth"

    splits = np.load(split_file)
    train_raw = CIFAR10(root=str(data_dir), train=True, download=False)
    test_raw = CIFAR10(root=str(data_dir), train=False, download=False)

    # Training data is used only to estimate the activation clipping level.
    train_set = ImageDataset(
        train_raw.data, train_raw.targets,
        splits["train_indices"], known_only=True
    )
    val_set = ImageDataset(
        train_raw.data, train_raw.targets,
        splits["val_indices"], known_only=True
    )
    known_test_set = ImageDataset(
        test_raw.data, test_raw.targets,
        splits["known_test_indices"], known_only=True
    )
    unknown_test_set = ImageDataset(
        test_raw.data, test_raw.targets,
        splits["unknown_test_indices"], known_only=False
    )

    checkpoint = torch.load(
        checkpoint_file, map_location=DEVICE, weights_only=True
    )
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(KNOWN))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print("Device:", DEVICE)
    print("Extracting penultimate features...")
    train_features, _, _ = collect_features(model, make_loader(train_set))
    val_features, val_logits, _ = collect_features(model, make_loader(val_set))
    known_features, known_logits, known_labels = collect_features(
        model, make_loader(known_test_set)
    )
    unknown_features, unknown_logits, _ = collect_features(
        model, make_loader(unknown_test_set)
    )

    # Standard Energy baseline.
    baseline_val = energy_from_logits(val_logits)
    baseline_known = energy_from_logits(known_logits)
    baseline_unknown = energy_from_logits(unknown_logits)

    # ReAct clipping level estimated from training activations only.
    clip_value = float(np.percentile(train_features, 99.5))
    print(f"ReAct clipping value (training feature percentile): {clip_value:.6f}")

    def react_logits(features):
        clipped = np.minimum(features, clip_value)
        return clipped @ model.fc.weight.detach().cpu().numpy().T + \
            model.fc.bias.detach().cpu().numpy()

    # Note: ReAct clips features before the classifier.
    react_val = energy_from_logits(react_logits(val_features))
    react_known = energy_from_logits(react_logits(known_features))
    react_unknown = energy_from_logits(react_logits(unknown_features))

    # Thresholds calibrated on known validation data only.
    baseline_threshold = np.quantile(baseline_val, 0.95, method="higher")
    react_threshold = np.quantile(react_val, 0.95, method="higher")

    def report(name, val_scores, known_scores, unknown_scores, threshold):
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

    report(
        "Standard Energy",
        baseline_val, baseline_known, baseline_unknown,
        baseline_threshold
    )
    report(
        "ReAct + Energy",
        react_val, react_known, react_unknown,
        react_threshold
    )

    output = ROOT / "results/react_energy_results.npz"
    np.savez(
        output,
        clip_value=clip_value,
        baseline_threshold=baseline_threshold,
        react_threshold=react_threshold,
        baseline_val_energy=baseline_val,
        baseline_known_energy=baseline_known,
        baseline_unknown_energy=baseline_unknown,
        react_val_energy=react_val,
        react_known_energy=react_known,
        react_unknown_energy=react_unknown,
    )
    print(f"\nSaved experiment results: {output}")
    print("No model was retrained or checkpoint modified.")


if __name__ == "__main__":
    main()
