import sys

import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v2


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


model = mobilenet_v2(weights=None)

input_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(input_features, len(CLASSES))

model.load_state_dict(
    torch.load(
        "models/best_waste_classifier.pth",
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()


def predict(image_path):
    image = Image.open(image_path).convert("RGB")

    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(DEVICE)

    with torch.no_grad():
        output = model(image)

        probabilities = torch.softmax(output, dim=1)

    confidence, predicted_class = torch.max(
        probabilities,
        dim=1
    )

    predicted_label = CLASSES[predicted_class.item()]
    confidence = confidence.item() * 100

    return predicted_label, confidence, probabilities[0]


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python src/predict.py <image>")
        sys.exit(1)

    image_path = sys.argv[1]

    label, confidence, probabilities = predict(image_path)

    print()
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2f}%")

    print("\nProbabilities:")

    for class_name, probability in zip(CLASSES, probabilities):
        print(
            f"{class_name:10s}: "
            f"{probability.item() * 100:.2f}%"
        )