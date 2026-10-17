
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
KNOWN = ["airplane", "automobile", "bird", "cat", "deer"]
CLASS_IDS = [CLASSES.index(c) for c in KNOWN]
LABEL_MAP = {old: new for new, old in enumerate(CLASS_IDS)}


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


def main():
    checkpoint = torch.load(
        ROOT / "models" / "known_classifier.pth",
        map_location=DEVICE,
        weights_only=True,
    )

    model = SmallCNN().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test = CIFAR10(
        root=str(ROOT / "data" / "raw"),
        train=False,
        download=False,
    )

    split = np.load(
        ROOT / "data" / "processed" / "dataset_splits.npz"
    )
    energy = np.load(ROOT / "results" / "energy_results.npz")

    known_indices = split["known_test_indices"]
    unknown_indices = split["unknown_test_indices"]

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.247, 0.243, 0.261),
        ),
    ])

    def predict(indices):
        predictions, confidences = [], []
        with torch.no_grad():
            for start in range(0, len(indices), 128):
                batch_indices = indices[start:start + 128]
                images = torch.stack([
                    transform(test.data[int(i)])
                    for i in batch_indices
                ]).to(DEVICE)

                probabilities = torch.softmax(model(images), dim=1)
                predictions.extend(
                    probabilities.argmax(1).cpu().numpy()
                )
                confidences.extend(
                    probabilities.max(1).values.cpu().numpy()
                )

        return np.array(predictions), np.array(confidences)

    known_pred, known_conf = predict(known_indices)
    unknown_pred, unknown_conf = predict(unknown_indices)

    known_true = np.array([
        LABEL_MAP[int(test.targets[int(i)])]
        for i in known_indices
    ])

    # Energy scores are stored in exactly the same test order.
    unknown_energy = energy["unknown_scores"]
    threshold = float(energy["threshold"])

    known_wrong = np.flatnonzero(known_pred != known_true)
    unknown_accepted = np.flatnonzero(unknown_energy <= threshold)

    print("Known test images misclassified:", len(known_wrong))
    print("Unknown images accepted as known:", len(unknown_accepted))

    # Show up to five examples of each failure type.
    examples = []

    for j in known_wrong[:5]:
        examples.append((
            test.data[int(known_indices[j])],
            f"Known: {KNOWN[known_true[j]]}",
            f"Predicted: {KNOWN[known_pred[j]]}",
        ))

    for j in unknown_accepted[:5]:
        original = int(test.targets[int(unknown_indices[j])])
        examples.append((
            test.data[int(unknown_indices[j])],
            f"Unknown: {CLASSES[original]}",
            f"Accepted as: {KNOWN[unknown_pred[j]]}",
        ))

    if not examples:
        print("No examples to display.")
        return

    columns = 5
    rows = (len(examples) + columns - 1) // columns
    fig, axes = plt.subplots(
        rows, columns, figsize=(14, 3 * rows), squeeze=False
    )

    for ax in axes.flat:
        ax.axis("off")

    for ax, (image, actual, predicted) in zip(axes.flat, examples):
        ax.imshow(image)
        ax.set_title(f"{actual}\n{predicted}", fontsize=9)
        ax.axis("off")

    fig.suptitle("SmartVision Failure Cases")
    fig.tight_layout()

    output = ROOT / "results" / "plots"
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "failure_cases.png", dpi=150)
    plt.show()

    print("Saved failure examples to results/plots/failure_cases.png")


if __name__ == "__main__":
    main()
