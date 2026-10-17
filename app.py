
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KNOWN_CLASSES = ["airplane", "automobile", "bird", "cat", "deer"]

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# --------------------------------------------------
# Load model and matching threshold
# --------------------------------------------------

@st.cache_resource
def load_model(model_choice):
    if model_choice == "LogitNorm ResNet-18":
        checkpoint_path = (
            ROOT / "models" / "known_classifier_logitnorm.pth"
        )
        results_path = ROOT / "results" / "logitnorm_ood_results.npz"
    else:
        checkpoint_path = (
            ROOT / "models" / "known_classifier_resnet18.pth"
        )
        results_path = ROOT / "results" / "resnet18_ood_results.npz"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    if not results_path.exists():
        raise FileNotFoundError(
            f"Threshold results not found: {results_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(KNOWN_CLASSES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    with np.load(results_path) as results:
        threshold = float(results["threshold"])

    return model, threshold


# --------------------------------------------------
# Page
# --------------------------------------------------

st.set_page_config(
    page_title="SmartVision",
    page_icon="🔎",
    layout="centered",
)

st.title("🔎 SmartVision")
st.subheader("Open-Set Image Recognition")

st.write(
    "SmartVision classifies images into five known categories "
    "and can reject inputs that appear unfamiliar."
)

st.info(
    "Known categories: Airplane, Automobile, Bird, Cat and Deer."
)

model_choice = st.selectbox(
    "Choose model",
    ["LogitNorm ResNet-18", "Original ResNet-18"],
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
)


# --------------------------------------------------
# Inference
# --------------------------------------------------

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded image",
            use_container_width=True,
        )

        model, threshold = load_model(model_choice)

        tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            energy = -torch.logsumexp(logits, dim=1).item()

        predicted_idx = int(probabilities.argmax().item())
        predicted_class = KNOWN_CLASSES[predicted_idx]
        confidence = float(probabilities[predicted_idx].item())

        # Higher energy indicates a more unfamiliar input.
        rejected = energy > threshold

        st.divider()

        # --------------------------------------------------
        # Display exactly ONE final decision
        # --------------------------------------------------

        if rejected:
            st.error("⚠️ UNKNOWN / OUT-OF-DISTRIBUTION INPUT")

            st.header("Final prediction: Unknown")

            st.write(
                "The input has been rejected by the unknown-detection "
                "system. No known category is returned as the final "
                "prediction because the result is considered unreliable."
            )

            st.metric("Energy score", f"{energy:.3f}")
            st.caption(f"Rejection threshold: {threshold:.3f}")

            # Deliberately hide the predicted class, confidence,
            # and class-probability chart for rejected inputs.

        else:
            st.success("✅ KNOWN-CATEGORY PREDICTION")

            st.header(f"Final prediction: {predicted_class.title()}")

            st.metric("Classifier confidence", f"{confidence:.1%}")
            st.metric("Energy score", f"{energy:.3f}")

            st.caption(f"Rejection threshold: {threshold:.3f}")

            st.write("Class probabilities")

            chart_data = {
                name.title(): float(probabilities[i].item())
                for i, name in enumerate(KNOWN_CLASSES)
            }

            st.bar_chart(chart_data)

        st.divider()

        st.caption(
            "Unknown detection is a statistical decision, not a guarantee. "
            "Some unfamiliar images may be accepted as known, and some "
            "familiar images may be rejected. The threshold is calibrated "
            "using known validation data."
        )

    except (FileNotFoundError, KeyError) as exc:
        st.error(f"Unable to load the selected model or threshold: {exc}")

    except Exception as exc:
        st.error(f"Unable to process this image: {exc}")
