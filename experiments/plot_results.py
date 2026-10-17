
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "evaluation_results.npz"
PLOTS = ROOT / "results" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

if not RESULTS.exists():
    raise FileNotFoundError(
        "Run python src/evaluate.py first."
    )

data = np.load(RESULTS)
known = data["known_msp"]
unknown = data["unknown_msp"]
threshold = float(data["threshold"])

# Plot 1: Compare confidence distributions.
plt.figure(figsize=(9, 5))
plt.hist(
    known, bins=30, alpha=0.65,
    label="Known test images", density=True
)
plt.hist(
    unknown, bins=30, alpha=0.65,
    label="Unknown test images", density=True
)
plt.axvline(
    threshold, linestyle="--",
    label=f"Rejection threshold = {threshold:.3f}"
)
plt.xlabel("Maximum softmax probability")
plt.ylabel("Density")
plt.title("Confidence: Known vs Unknown Images")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS / "confidence_distribution.png", dpi=150)
plt.show()

# Plot 2: Rejection rate at different thresholds.
thresholds = np.linspace(0.05, 0.95, 50)
known_rejection = [
    np.mean(known < t) for t in thresholds
]
unknown_detection = [
    np.mean(unknown < t) for t in thresholds
]

plt.figure(figsize=(9, 5))
plt.plot(thresholds, known_rejection, label="Known rejection rate")
plt.plot(thresholds, unknown_detection, label="Unknown detection recall")
plt.xlabel("Confidence threshold")
plt.ylabel("Rate")
plt.title("Threshold Trade-off")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS / "threshold_tradeoff.png", dpi=150)
plt.show()

print("Plots saved in:", PLOTS)
