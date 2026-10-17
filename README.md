# SmartVision — Open-Set Image Classification

SmartVision is an image classification system designed to classify familiar categories and identify potentially unfamiliar inputs using out-of-distribution (OOD) detection.

## Problem

Traditional classifiers assign an input to one of their known classes even when the image belongs to a category outside their training set. SmartVision combines known-class classification with Energy-based OOD detection to help identify such inputs.

## Dataset

We use CIFAR-10 with five known classes:

- Airplane
- Automobile
- Bird
- Cat
- Deer

The remaining classes—dog, frog, horse, ship, and truck—are excluded from training and used as unknown categories during evaluation.

## Methodology

- **Classifier:** Pretrained ResNet-18, fine-tuned on the known classes.
- **Training objective:** LogitNorm fine-tuning.
- **OOD detection:** Energy score with a threshold calibrated on known-class validation data.
- **Confidence calibration:** Expected Calibration Error (ECE), multiclass Brier score, reliability diagram, and validation-fitted temperature scaling.
- **Demo:** Streamlit application.

## Results

| Metric | Result |
|---|---:|
| Known-class test accuracy | 95.82% |
| Energy OOD AUROC | 0.7925 |
| Energy OOD AUPR | 0.7664 |
| ECE after temperature scaling | 0.0050 |
| Brier score after temperature scaling | 0.0629 |

Temperature scaling improved confidence calibration without changing known-class accuracy. It was not established to improve OOD detection.

## Limitations

The OOD detector does not identify every unfamiliar input. At approximately 5.2% known-class rejection, unknown-class recall was approximately 25.44% in our evaluation. Some unfamiliar images may be confidently accepted as known classes.

Temperature scaling was evaluated on known-class test data; good calibration on known classes does not guarantee reliable unknown detection.

## Project Structure

- `app.py` — Streamlit demo
- `src/` — Main training, evaluation and calibration scripts
- `experiments/` — Additional experimental methods and analysis
- `models/` — Trained model checkpoints
- `results/` — Evaluation outputs and visualizations
- `data/` — Dataset split metadata

## Setup

Use Python 3.11 and install the required dependencies in a virtual environment. Install PyTorch with a build compatible with your system, then install the remaining dependencies from `requirements.txt`.

Download CIFAR-10 using the dataset preparation workflow and verify the expected split file exists at `data/processed/dataset_splits.npz`.

## Run the demo

From the project root:

```bash
streamlit run app.py
```

## Reproducibility

Training and evaluation scripts are located in `src/`. Additional experiments are preserved in `experiments/`. Reported metrics should be interpreted in the context of their corresponding model checkpoints and evaluation scripts.