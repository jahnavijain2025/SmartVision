from pathlib import Path
import sys

import numpy as np
from torchvision.datasets import CIFAR10

# Allow imports from the src folder
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_config import (
    KNOWN_CLASSES,
    UNKNOWN_CLASSES,
    RANDOM_SEED,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

def main():
    print("Downloading CIFAR-10...")
    train_data = CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=True,
    )

    test_data = CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=True,
    )

    train_labels = np.array(train_data.targets)
    test_labels = np.array(test_data.targets)

    known_ids = [CLASS_NAMES.index(c) for c in KNOWN_CLASSES]
    unknown_ids = [CLASS_NAMES.index(c) for c in UNKNOWN_CLASSES]

    # Known training examples only
    known_train_indices = np.where(
        np.isin(train_labels, known_ids)
    )[0]

    # Split known training examples into train/validation
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(known_train_indices)

    split = int(0.8 * len(known_train_indices))
    train_indices = known_train_indices[:split]
    val_indices = known_train_indices[split:]

    # Known test images and excluded-class test images
    known_test_indices = np.where(
        np.isin(test_labels, known_ids)
    )[0]

    unknown_test_indices = np.where(
        np.isin(test_labels, unknown_ids)
    )[0]

    output_dir = ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_dir / "dataset_splits.npz",
        train_indices=train_indices,
        val_indices=val_indices,
        known_test_indices=known_test_indices,
        unknown_test_indices=unknown_test_indices,
        train_labels=train_labels,
        test_labels=test_labels,
    )

    print("\nDataset preparation complete!")
    print("Training images:", len(train_indices))
    print("Validation images:", len(val_indices))
    print("Known test images:", len(known_test_indices))
    print("Unknown test images:", len(unknown_test_indices))
    print("Saved split metadata to:", output_dir)

if __name__ == "__main__":
    main()