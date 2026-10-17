
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score

data = np.load("results/resnet18_msp_energy_comparison.npz")

known_energy = np.asarray(data["known_energy"]).ravel()
unknown_energy = np.asarray(data["unknown_energy"]).ravel()
unknown_labels = np.asarray(data["unknown_labels"]).ravel()

# CIFAR-10 original label mapping
class_names = {
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}

output_dir = Path("results/plots")
output_dir.mkdir(parents=True, exist_ok=True)

# Threshold selected from known validation data to target 5% rejection.
val_energy = np.asarray(data["val_energy"]).ravel()
threshold = np.quantile(val_energy, 0.95, method="higher")

print(f"Energy threshold (known validation 95th percentile): {threshold:.6f}")
print("\nPer-class unknown detection")
print("-" * 65)

# Positive means unknown; higher energy means more likely unknown.
for label in sorted(class_names):
    mask = unknown_labels == label
    scores = unknown_energy[mask]

    recall = np.mean(scores >= threshold)

    # Compare this unknown class with all known test samples.
    y_true = np.concatenate([
        np.zeros(len(known_energy)),
        np.ones(len(scores)),
    ])
    combined_scores = np.concatenate([known_energy, scores])
    auroc = roc_auc_score(y_true, combined_scores)

    print(
        f"{class_names[label]:<8} | "
        f"images={len(scores):4d} | "
        f"unknown recall={recall:.4f} ({recall*100:.1f}%) | "
        f"AUROC={auroc:.4f}"
    )

# Plot 1: known vs all unknown energy distributions
plt.figure(figsize=(10, 6))
plt.hist(known_energy, bins=60, density=True, alpha=0.65,
         label="Known test images")
plt.hist(unknown_energy, bins=60, density=True, alpha=0.55,
         label="Unknown test images")
plt.axvline(threshold, linestyle="--",
            label="Validation-derived threshold")
plt.xlabel("Energy score (higher = more likely unknown)")
plt.ylabel("Density")
plt.title("ResNet-18: Known vs Unknown Energy Distributions")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "resnet18_energy_distribution.png", dpi=160)
plt.close()

# Plot 2: individual unknown-class distributions
plt.figure(figsize=(11, 6))
for label in sorted(class_names):
    scores = unknown_energy[unknown_labels == label]
    plt.hist(scores, bins=45, density=True, alpha=0.35,
             label=class_names[label])

plt.axvline(threshold, linestyle="--",
            label="Validation-derived threshold")
plt.xlabel("Energy score (higher = more likely unknown)")
plt.ylabel("Density")
plt.title("Energy Distributions by Unknown Class")
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "resnet18_unknown_class_distributions.png", dpi=160)
plt.close()

# Plot 3: unknown detection recall by class
names = []
recalls = []

for label in sorted(class_names):
    scores = unknown_energy[unknown_labels == label]
    names.append(class_names[label])
    recalls.append(np.mean(scores >= threshold))

plt.figure(figsize=(9, 5))
bars = plt.bar(names, np.array(recalls) * 100)
plt.ylabel("Unknown images detected (%)")
plt.xlabel("Unknown class")
plt.title("ResNet-18 Energy Detection by Unknown Class")
plt.ylim(0, 100)

for bar, recall in zip(bars, recalls):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{recall*100:.1f}%",
        ha="center",
        va="bottom",
    )

plt.tight_layout()
plt.savefig(output_dir / "resnet18_unknown_recall_by_class.png", dpi=160)
plt.close()

print("\nSaved plots:")
print(output_dir / "resnet18_energy_distribution.png")
print(output_dir / "resnet18_unknown_class_distributions.png")
print(output_dir / "resnet18_unknown_recall_by_class.png")
