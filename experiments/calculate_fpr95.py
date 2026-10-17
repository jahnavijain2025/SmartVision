
import numpy as np

data = np.load("results/resnet18_msp_energy_comparison.npz")

def calculate_fpr95(known_scores, unknown_scores, method):
    known_scores = np.asarray(known_scores).ravel()
    unknown_scores = np.asarray(unknown_scores).ravel()

    # Search thresholds at observed score values.
    # Unknown if score >= threshold.
    candidates = np.unique(np.concatenate([
        known_scores, unknown_scores
    ]))
    candidates = np.concatenate([
        [np.nextafter(candidates[-1], np.inf)],
        candidates[::-1]
    ])

    results = []
    for threshold in candidates:
        tpr = np.mean(unknown_scores >= threshold)
        fpr = np.mean(known_scores >= threshold)
        if tpr >= 0.95:
            results.append((fpr, tpr, threshold))

    # Lowest FPR among thresholds meeting >=95% TPR
    fpr, tpr, threshold = min(results, key=lambda x: (x[0], -x[1]))

    print(f"\n{method}")
    print(f"FPR@95TPR: {fpr:.4f} ({100*fpr:.2f}%)")
    print(f"Achieved TPR: {tpr:.4f} ({100*tpr:.2f}%)")
    print(f"Threshold: {threshold:.8f}")

calculate_fpr95(
    data["known_energy"], data["unknown_energy"], "Energy"
)
calculate_fpr95(
    data["known_msp"], data["unknown_msp"], "MSP uncertainty"
)
