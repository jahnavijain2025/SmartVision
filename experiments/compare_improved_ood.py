
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
data = np.load(ROOT / "results" / "improved_ood_results.npz")

# Validation scores are used to choose thresholds.
val_msp = data["val_msp"]
val_energy = data["val_energy"]

# Test scores are used only for evaluation.
known_msp = data["known_msp"]
unknown_msp = data["unknown_msp"]
known_energy = data["known_energy"]
unknown_energy = data["unknown_energy"]

methods = {
    "MSP": {
        "val_known": -val_msp,
        "test_known": -known_msp,
        "test_unknown": -unknown_msp,
    },
    "Energy": {
        "val_known": val_energy,
        "test_known": known_energy,
        "test_unknown": unknown_energy,
    },
}

out_dir = ROOT / "results" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)

# Plot ROC curves from the test scores for final evaluation.
fig, ax = plt.subplots(figsize=(8, 6))

for name, values in methods.items():
    test_known = values["test_known"]
    test_unknown = values["test_unknown"]

    y_true = np.concatenate([
        np.zeros(len(test_known), dtype=int),
        np.ones(len(test_unknown), dtype=int),
    ])
    scores = np.concatenate([test_known, test_unknown])

    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)
    ax.plot(fpr, tpr, label=f"{name} (AUROC = {auc:.3f})")

ax.plot([0, 1], [0, 1], "--", label="Chance")
ax.set_xlabel("Known-image rejection rate")
ax.set_ylabel("Unknown-image detection recall")
ax.set_title("SmartVision: Improved Model OOD ROC")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()

plot_path = out_dir / "improved_ood_roc.png"
fig.savefig(plot_path, dpi=160)
plt.close(fig)

print("Saved ROC plot:", plot_path)
print("\n=== VALIDATION-SELECTED THRESHOLD EVALUATION ===")

summary = {}

for name, values in methods.items():
    val_known = values["val_known"]
    test_known = values["test_known"]
    test_unknown = values["test_unknown"]

    print(f"\n{name}")
    summary[name] = {}

    for target_rejection in [0.05, 0.10, 0.20, 0.30]:
        # Higher anomaly score means more likely to be unknown.
        threshold = float(
            np.quantile(val_known, 1 - target_rejection)
        )

        known_reject = float(np.mean(test_known > threshold))
        unknown_recall = float(np.mean(test_unknown > threshold))

        summary[name][str(target_rejection)] = {
            "threshold": threshold,
            "known_rejection_rate": known_reject,
            "unknown_detection_recall": unknown_recall,
        }

        print(
            f"Target known rejection={target_rejection:.0%} | "
            f"threshold={threshold:.4f} | "
            f"test known rejection={known_reject:.3f} | "
            f"test unknown recall={unknown_recall:.3f}"
        )

np.savez(
    ROOT / "results" / "improved_ood_threshold_comparison.npz",
    msp_thresholds=np.array([
        summary["MSP"][str(x)]["threshold"]
        for x in [0.05, 0.10, 0.20, 0.30]
    ]),
    energy_thresholds=np.array([
        summary["Energy"][str(x)]["threshold"]
        for x in [0.05, 0.10, 0.20, 0.30]
    ]),
)

print("\nSaved validation-selected threshold comparison.")
print("Note: test metrics are evaluation only; do not tune thresholds on them.")
