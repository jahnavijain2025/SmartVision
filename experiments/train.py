
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
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

BATCH_SIZE = 128
EPOCHS = 5

class ImageDataset(Dataset):
    def __init__(self, images, labels, indices, training=False):
        self.images = images
        self.labels = labels
        self.indices = indices
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip() if training
            else transforms.Lambda(lambda x: x),
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
        label = LABEL_MAP[int(self.labels[i])]
        return image, label


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


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    data_dir = ROOT / "data" / "raw"
    split_file = ROOT / "data" / "processed" / "dataset_splits.npz"

    if not split_file.exists():
        raise FileNotFoundError(
            "Run python src/prepare_dataset.py first."
        )

    splits = np.load(split_file)
    train_raw = CIFAR10(
        root=str(data_dir), train=True, download=False
    )
    test_raw = CIFAR10(
        root=str(data_dir), train=False, download=False
    )

    # Keep only known categories in training and validation.
    train_set = ImageDataset(
        train_raw.data, train_raw.targets,
        splits["train_indices"], training=True
    )
    val_set = ImageDataset(
        train_raw.data, train_raw.targets,
        splits["val_indices"], training=False
    )

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0
    )

    model = SmallCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("Device:", DEVICE)
    print("Training only on:", KNOWN)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        count = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            count += labels.size(0)

        model.eval()
        val_correct = 0
        val_count = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                predictions = model(images).argmax(1)
                val_correct += (predictions == labels).sum().item()
                val_count += labels.size(0)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"loss={total_loss / count:.4f} | "
            f"train_acc={correct / count:.3f} | "
            f"val_acc={val_correct / val_count:.3f}"
        )

    model_dir = ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "known_classes": KNOWN,
        "class_ids": CLASS_IDS,
    }, model_dir / "known_classifier.pth")

    print("Model saved to models/known_classifier.pth")


if __name__ == "__main__":
    main()
