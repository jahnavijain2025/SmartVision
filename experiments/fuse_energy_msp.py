
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

data = np.load("results/resnet18_msp_energy_comparison.npz")

# Load saved scores
val_energy = np.asarray(data["val_energy"]).ravel()
val_msp = np.asarray(data["val_msp"]).ravel()

known_energy = np.asarray(data["known_energy"]).ravel()
unknown_energy = np.asarray(data["unknown_energy"]).ravel()

known_msp = np.asarray(data["known_msp"]).ravel()
unknown_msp = np.asarray(data["unknown_msp"]).ravel()

# Standardize scores using KNOWN VALIDATION statistics only.
# Higher values should indicate more likely unknown inputs.
def standardize(scores, mean, std):
    return (scores - mean) / max(float(std), 1e-12)

energy_mean = val_energy.mean()
energy_std = val_energy.std()
msp_mean = val_msp.mean()
msp_std = val_msp.std()

ve = standardize(val_energy, energy_mean, energy_std)
vm = standardize(val_msp, msp_mean, msp_std)

ke = standardize(known_energy, energy_mean, energy_std)
ue = standardize(unknown_energy, energy_mean, energy_std)

km = standardize(known_msp, msp_mean, msp_std)
um = standardize(unknown_msp, msp_mean, msp_std)

# For each weight, choose the threshold on known validation scores
# to target approximately 5% known rejection.
weights = [0.0, 0.25, 0.5, 0.75, 1.0]
candidates = []

for w in weights:
    val_score = w * ve + (1 - w) * vm
    threshold = np.quantile(val_score, 0.95, method="higher")

    known_score = w * ke + (1 - w) * km
    unknown_score = w * ue + (1 - w) * um

    known_rejection = np.mean(known_score >= threshold)
    unknown_recall = np.mean(unknown_score >= threshold)

    candidates.append({
        "weight": w,
        "threshold": threshold,
        "known_rejection": known_rejection,
        "unknown_recall": unknown_recall,
    })

# Select the weight using validation-known rejection only is not enough
# to identify which detector best detects unknowns. Without a separate
# validation outlier set, use Energy alone as the pre-registered baseline.
# Report all test results as exploratory rather than selecting the best
# weight from the test set.
print("Exploratory test comparison (do not choose a winner from test results)")
print("w = Energy weight; (1-w) = MSP weight")
print("-" * 72)

for result in candidates:
    w = result["weight"]
    fused_known = w * ke + (1 - w) * km
    fused_unknown = w * ue + (1 - w) * um

    y_true = np.concatenate([
        np.zeros(len(fused_known)),
        np.ones(len(fused_unknown))
    ])
    scores = np.concatenate([fused_known, fused_unknown])

    auroc = roc_auc_score(y_true, scores)
    aupr = average_precision_score(y_true, scores)

    print(
        f"Energy weight={w:.2f}, MSP weight={1-w:.2f} | "
        f"known rejection={result['known_rejection']:.4f} | "
        f"unknown recall={result['unknown_recall']:.4f} | "
        f"AUROC={auroc:.4f} | AUPR={aupr:.4f}"
    )

print("\nReminder: thresholds use known validation scores.")
print("No model was retrained or checkpoint changed.")
print("Without validation outliers, do not select a fusion weight")
print("based on unknown test recall or test AUROC.")
