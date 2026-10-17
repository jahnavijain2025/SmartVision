
import numpy as np

data = np.load("results/resnet18_msp_energy_comparison.npz")

val = np.asarray(data["val_msp"]).ravel()
known = np.asarray(data["known_msp"]).ravel()
unknown = np.asarray(data["unknown_msp"]).ravel()

# Select a threshold from known validation scores to target 5% rejection.
threshold = np.quantile(val, 0.95, method="higher")

known_rejection = np.mean(known >= threshold)
unknown_recall = np.mean(unknown >= threshold)

print(f"Validation-derived MSP threshold: {threshold:.8f}")
print(f"Known test rejection: {known_rejection:.4f} "
      f"({100 * known_rejection:.2f}%)")
print(f"Unknown test recall: {unknown_recall:.4f} "
      f"({100 * unknown_recall:.2f}%)")
