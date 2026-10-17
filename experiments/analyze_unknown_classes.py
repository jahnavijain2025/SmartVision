
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
data = np.load(ROOT / "results" / "improved_ood_results.npz")

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
KNOWN = ["airplane", "automobile", "bird", "cat", "deer"]
UNKNOWN = ["dog", "frog", "horse", "ship", "truck"]

known_energy = data["known_energy"]
unknown_energy = data["unknown_energy"]
unknown_labels = data["unknown_original_labels"]

known_msp = data["known_msp"]
unknown_msp = data["unknown_msp"]

energy_threshold = float(data["energy_threshold"])
msp_threshold = float(data["msp_threshold"])

print("\n=== UNKNOWN CLASS ANALYSIS ===")
print("Higher energy means more likely unknown.")
print("Lower MSP means less confident in the predicted known class.")

for class_name in UNKNOWN:
    original_id = CLASSES.index(class_name)
    mask = unknown_labels == original_id

    e_scores = unknown_energy[mask]
    m_scores = unknown_msp[mask]

    energy_recall = np.mean(e_scores > energy_threshold)
    msp_recall = np.mean(m_scores < msp_threshold)

    # AUROC for this unknown class versus all known test images.
    y = np.concatenate([
        np.zeros(len(known_energy), dtype=int),
        np.ones(len(e_scores), dtype=int),
    ])
    energy_scores = np.concatenate([known_energy, e_scores])
    msp_scores = np.concatenate([-known_msp, -m_scores])

    energy_auc = roc_auc_score(y, energy_scores)
    msp_auc = roc_auc_score(y, msp_scores)

    print(f"\n{class_name.upper()}")
    print(f"  Images: {len(e_scores)}")
    print(f"  Energy detection recall: {energy_recall:.4f}")
    print(f"  Energy AUROC: {energy_auc:.4f}")
    print(f"  MSP detection recall: {msp_recall:.4f}")
    print(f"  MSP AUROC: {msp_auc:.4f}")

print("\nAnalysis complete.")
