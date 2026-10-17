from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
SPLIT_FILE = ROOT / "data" / "processed" / "dataset_splits.npz"

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

dataset = CIFAR10(root=str(DATA_DIR), train=True, download=False)
splits = np.load(SPLIT_FILE)

train_indices = splits["train_indices"]
val_indices = splits["val_indices"]

print("Training images:", len(train_indices))
print("Validation images:", len(val_indices))
print("Known test images:", len(splits["known_test_indices"]))
print("Unknown test images:", len(splits["unknown_test_indices"]))

fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for class_id, ax in enumerate(axes.flat):
    indices = [
        i for i in train_indices
        if dataset.targets[i] == class_id
    ]

    if indices:
        idx = indices[0]
        ax.imshow(dataset.data[idx])
        ax.set_title(CLASS_NAMES[class_id])
    ax.axis("off")

plt.tight_layout()
plt.show()