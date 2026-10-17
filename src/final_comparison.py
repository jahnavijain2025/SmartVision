
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PLOT_DIR = RESULTS / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

softmax = np.load(RESULTS / "evaluation_results.npz")
maha = np.load(RESULTS / "mahalanobis_results.npz")
energy = np.load(RESULTS / "energy_results.npz")

methods = {
    "Softmax MSP": (
        -softmax["known_msp"],
        -softmax["unknown_msp"],
    ),
    "Mahalanobis": (
        maha["known_distances"],
        maha["unknown_distances"],
    ),
    "Energy": (
        energy["known_scores"],
        energy["unknown_scores"],
    ),
}

print("=== FINAL OOD COMPARISON ===")
print(f"{'Method':<18} {'AUROC':>8}")

for name, (known, unknown) in methods.items():
    labels = np.concatenate([
        np.zeros(len(known)),
        np.ones(len(unknown)),
    ])
    scores = np.concatenate([known, unknown])
    auc = roc_auc_score(labels, scores)
    print(f"{name:<18} {auc:>8.4f}")

# Plot score distributions for all three methods.
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (name, (known, unknown)) in zip(axes, methods.items()):
    ax.hist(
        known, bins=30, density=True, alpha=0.6,
        label="Known", color="steelblue"
    )
    ax.hist(
        unknown, bins=30, density=True, alpha=0.6,
        label="Unknown", color="darkorange"
    )
    ax.set_title(name)
    ax.set_xlabel("OOD score (higher = more unknown)")
    ax.set_ylabel("Density")
    ax.legend()

fig.suptitle("SmartVision: OOD Score Comparison")
fig.tight_layout()
fig.savefig(PLOT_DIR / "all_method_distributions.png", dpi=150)
plt.show()

print("Plot saved to results/plots/all_method_distributions.png")
