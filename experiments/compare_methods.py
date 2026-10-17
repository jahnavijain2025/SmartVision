
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

softmax = np.load(RESULTS / "evaluation_results.npz")
maha = np.load(RESULTS / "mahalanobis_results.npz")

# Load scores produced by the two detectors.
known_msp = softmax["known_msp"]
unknown_msp = softmax["unknown_msp"]

known_dist = maha["known_distances"]
unknown_dist = maha["unknown_distances"]

# Higher scores should mean "more likely to be unknown".
softmax_scores = -np.concatenate([known_msp, unknown_msp])
maha_scores = np.concatenate([known_dist, unknown_dist])

labels = np.concatenate([
    np.zeros(len(known_msp)),
    np.ones(len(unknown_msp)),
])

softmax_auc = roc_auc_score(labels, softmax_scores)
maha_auc = roc_auc_score(labels, maha_scores)

print("=== OOD METHOD COMPARISON ===")
print(f"Softmax AUROC:    {softmax_auc:.4f}")
print(f"Mahalanobis AUROC:{maha_auc:.4f}")

# Compare ROC curves across all possible thresholds.
def roc_points(scores, labels):
    thresholds = np.concatenate([
        [np.inf],
        np.sort(np.unique(scores))[::-1],
        [-np.inf],
    ])
    tpr, fpr = [], []

    for threshold in thresholds:
        predicted_unknown = scores >= threshold
        tpr.append(np.mean(predicted_unknown[labels == 1]))
        fpr.append(np.mean(predicted_unknown[labels == 0]))

    return np.array(fpr), np.array(tpr)

soft_fpr, soft_tpr = roc_points(softmax_scores, labels)
maha_fpr, maha_tpr = roc_points(maha_scores, labels)

out = RESULTS / "plots"
out.mkdir(exist_ok=True)

plt.figure(figsize=(8, 6))
plt.plot(soft_fpr, soft_tpr, label=f"Softmax MSP (AUROC={softmax_auc:.3f})")
plt.plot(maha_fpr, maha_tpr, label=f"Mahalanobis (AUROC={maha_auc:.3f})")
plt.plot([0, 1], [0, 1], "--", label="Chance")
plt.xlabel("False positive rate (known images rejected)")
plt.ylabel("True positive rate (unknown images rejected)")
plt.title("OOD Detection: ROC Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(out / "ood_roc_comparison.png", dpi=150)
plt.show()

print("ROC plot saved to results/plots/ood_roc_comparison.png")
