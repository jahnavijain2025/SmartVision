
from pathlib import Path
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = ROOT / "data" / "processed" / "dataset_splits.npz"
MODEL_PATH = ROOT / "models" / "known_classifier_improved.pth"
OUTPUT_DIR = ROOT / "results" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KNOWN = ["airplane", "automobile", "bird", "cat", "deer"]
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.247, 0.243, 0.261)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3,
            padding=1, bias=False
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


def main():
    print("Device:", DEVICE)

    splits = np.load(DATA_PATH)
    print("Dataset keys:", splits.files)

    cifar = CIFAR10(
        root=str(ROOT / "data" / "raw"),
        train=False,
        download=False,
    )

    indices = splits["known_test_indices"].astype(np.int64)
    images = cifar.data[indices]
    original_labels = np.asarray(cifar.targets)[indices]

    # The known classes are CIFAR-10 labels 0 through 4.
    labels = original_labels.astype(np.int64)

    print("Known test images:", len(images))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    class KnownTestDataset(Dataset):
        def __len__(self):
            return len(images)

        def __getitem__(self, index):
            from PIL import Image
            image = transform(Image.fromarray(images[index]))
            return image, int(labels[index])

    loader = DataLoader(
        KnownTestDataset(),
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    model = ImprovedCNN().to(DEVICE)
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Checkpoint loaded successfully.")
    print("Checkpoint validation accuracy:",
          checkpoint.get("best_val_accuracy", "not recorded"))

    all_preds, all_probs, all_labels, all_images = [], [], [], []

    with torch.no_grad():
        for batch_images, batch_labels in loader:
            logits = model(batch_images.to(DEVICE))
            probabilities = torch.softmax(logits, dim=1)

            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
            all_labels.extend(batch_labels.numpy())
            all_images.extend(batch_images.numpy())

    preds = np.asarray(all_preds)
    probs = np.asarray(all_probs)
    labels = np.asarray(all_labels)
    images_norm = np.asarray(all_images)

    print(f"Known-test accuracy: {np.mean(preds == labels):.4%}")
    print("\nMisclassification counts:")

    for class_id, name in enumerate(KNOWN):
        mask = labels == class_id
        errors = np.sum(mask & (preds != labels))
        print(f"{name:10s}: {errors} wrong out of {mask.sum()}")

    cm = confusion_matrix(labels, preds, labels=range(5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=KNOWN,
    )
    display.plot(cmap="Blues", xticks_rotation=45)
    plt.title("ImprovedCNN — Known Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "improved_confusion_matrix.png",
        dpi=160,
    )
    plt.close()

    wrong_indices = np.flatnonzero(preds != labels)
    if len(wrong_indices):
        # Choose the 20 errors with the highest confidence.
        chosen = wrong_indices[
            np.argsort(probs[wrong_indices].max(axis=1))[::-1][:20]
        ]

        fig, axes = plt.subplots(4, 5, figsize=(14, 10))
        for ax in axes.flat:
            ax.axis("off")

        mean = np.asarray(MEAN).reshape(1, 1, 3)
        std = np.asarray(STD).reshape(1, 1, 3)

        for ax, idx in zip(axes.flat, chosen):
            img = images_norm[idx].transpose(1, 2, 0)
            img = np.clip(img * std + mean, 0, 1)
            ax.imshow(img)
            ax.set_title(
                f"True: {KNOWN[labels[idx]]}\n"
                f"Pred: {KNOWN[preds[idx]]} "
                f"({probs[idx, preds[idx]]:.1%})",
                fontsize=9,
            )
            ax.axis("off")

        plt.suptitle("Most Confident Misclassifications")
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / "improved_failure_cases.png",
            dpi=160,
        )
        plt.close()
    else:
        print("No misclassified images found.")

    print("\nSaved plots:")
    print(OUTPUT_DIR / "improved_confusion_matrix.png")
    print(OUTPUT_DIR / "improved_failure_cases.png")


if __name__ == "__main__":
    main()
