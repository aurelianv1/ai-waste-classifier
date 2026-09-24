from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v2


# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "best_waste_classifier.pth"


# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

DISPOSAL_GUIDE = {
    "cardboard": "♻️ Recycle in the paper/cardboard recycling bin.",
    "glass": "♻️ Recycle in the glass recycling bin.",
    "metal": "♻️ Recycle in the metal recycling bin.",
    "paper": "♻️ Recycle in the paper recycling bin.",
    "plastic": "♻️ Recycle in the plastic recycling bin.",
    "trash": "🗑️ Dispose of it in the general waste bin."
}


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


@st.cache_resource
def load_model():
    model = mobilenet_v2(weights=None)

    input_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        input_features,
        len(CLASSES)
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model = model.to(DEVICE)
    model.eval()

    return model


def predict(image, model):
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0]

    predicted_index = torch.argmax(probabilities).item()

    predicted_class = CLASSES[predicted_index]
    confidence = probabilities[predicted_index].item()

    return predicted_class, confidence, probabilities


# Load model
model = load_model()


# Page
st.set_page_config(
    page_title="AI Waste Classifier",
    page_icon="♻️",
    layout="centered"
)


st.title("♻️ AI Waste Classifier")

st.write(
    "Upload an image of a waste item and the CNN will "
    "classify it into one of six categories."
)


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded image",
        width=400
    )

    predicted_class, confidence, probabilities = predict(
        image,
        model
    )

    st.subheader("Prediction")

    if confidence >= 0.70:
        st.success(
            f"{predicted_class.upper()} — "
            f"{confidence * 100:.2f}% confidence"
        )

    elif confidence >= 0.50:
        st.warning(
            f"{predicted_class.upper()} — "
            f"{confidence * 100:.2f}% confidence\n\n"
            "The model is not highly confident about this prediction."
        )

    else:
        st.error(
            f"{predicted_class.upper()} — "
            f"{confidence * 100:.2f}% confidence\n\n"
            "Low-confidence prediction. Try another image."
        )


    # Only show a disposal recommendation
    # when confidence is at least 50%
    if confidence >= 0.50:
        st.subheader("Disposal recommendation")

        st.info(
            DISPOSAL_GUIDE[predicted_class]
        )

    st.subheader("Class probabilities")

    for class_name, probability in zip(
        CLASSES,
        probabilities
    ):
        percentage = probability.item() * 100

        st.write(
            f"**{class_name.capitalize()}** — "
            f"{percentage:.2f}%"
        )

        st.progress(float(probability.item()))