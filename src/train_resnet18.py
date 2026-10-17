
from pathlib import Path
import copy
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights


ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
KNOWN = ["airplane", "automobile", "bird", "cat", "deer"]
CLASS_IDS = [CLASSES.index(c) for c in KNOWN]
LABEL_MAP = {old: new for new, old in enumerate(CLASS_IDS)}

BATCH_SIZE = 64
HEAD_EPOCHS = 3
FINETUNE_EPOCHS = 10
SEED = 42

# ImageNet normalization for pretrained ResNet-18.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class ImageDataset(Dataset):
    def __init__(self, images, labels, indices, training=False):
        self.images = images
        self.labels = labels
        self.indices = indices

        ops = [transforms.ToPILImage()]

        if training:
            ops += [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]

        ops += [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]

        self.transform = transforms.Compose(ops)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = int(self.indices[index])
        image = self.transform(self.images[i])
        label = LABEL_MAP[int(self.labels[i])]
        return image, label


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            count += labels.size(0)

    return total_loss / count, correct / count


def run_training_phase(
    model, train_loader, val_loader, criterion,
    optimizer, epochs, phase_name, history, best
):
    for epoch in range(epochs):
        model.train()

        # Keep frozen backbone BatchNorm statistics fixed during
        # the classifier-only phase.
        if phase_name == "Head":
            model.eval()
            model.fc.train()

        total_loss = 0.0
        correct = 0
        count = 0

        for images, labels in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            count += labels.size(0)

        train_loss = total_loss / count
        train_acc = correct / count
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        history.append([
            phase_name, epoch + 1,
            train_loss, train_acc, val_loss, val_acc
        ])

        print(
            f"{phase_name} epoch {epoch + 1:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc > best["accuracy"]:
            best["accuracy"] = val_acc
            best["state"] = copy.deepcopy(model.state_dict())


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    split_file = ROOT / "data" / "processed" / "dataset_splits.npz"
    data_dir = ROOT / "data" / "raw"

    if not split_file.exists():
        raise FileNotFoundError(
            "Dataset split missing. Run src/prepare_dataset.py first."
        )

    splits = np.load(split_file)

    train_raw = CIFAR10(
        root=str(data_dir), train=True, download=False
    )

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
        shuffle=True, num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    print("Device:", DEVICE)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("Known classes:", KNOWN)
    print("Training images:", len(train_set))
    print("Validation images:", len(val_set))
    print("Loading pretrained ResNet-18 weights...")

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(KNOWN))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    history = []
    best = {"accuracy": 0.0, "state": None}
    start_time = time.time()

    # Phase 1: train only the new five-class classifier.
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    print("\nPhase 1: training the classification head")
    optimizer = torch.optim.AdamW(
        model.fc.parameters(), lr=0.001, weight_decay=0.0005
    )

    run_training_phase(
        model, train_loader, val_loader, criterion,
        optimizer, HEAD_EPOCHS, "Head", history, best
    )

    # Phase 2: fine-tune layer4 and the classifier.
    for param in model.layer4.parameters():
        param.requires_grad = True

    print("\nPhase 2: fine-tuning layer4 and the classifier")
    optimizer = torch.optim.AdamW(
        [
            {"params": model.layer4.parameters(), "lr": 0.0001},
            {"params": model.fc.parameters(), "lr": 0.0005},
        ],
        weight_decay=0.0005
    )

    run_training_phase(
        model, train_loader, val_loader, criterion,
        optimizer, FINETUNE_EPOCHS, "FineTune", history, best
    )

    if best["state"] is None:
        raise RuntimeError("No checkpoint was produced.")

    model_dir = ROOT / "models"
    model_dir.mkdir(exist_ok=True)

    checkpoint = {
        "model_state_dict": best["state"],
        "known_classes": KNOWN,
        "class_ids": CLASS_IDS,
        "architecture": "ResNet18_ImageNet_Pretrained",
        "best_val_accuracy": best["accuracy"],
        "head_epochs": HEAD_EPOCHS,
        "finetune_epochs": FINETUNE_EPOCHS,
    }

    output_model = model_dir / "known_classifier_resnet18.pth"
    torch.save(checkpoint, output_model)

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    np.save(
        results_dir / "resnet18_training_history.npy",
        np.array(history, dtype=object)
    )

    print("\nTraining complete!")
    print(f"Best validation accuracy: {best['accuracy']:.4f}")
    print(f"Saved model: {output_model}")
    print("Saved history: results/resnet18_training_history.npy")
    print(f"Elapsed time: {(time.time() - start_time) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
