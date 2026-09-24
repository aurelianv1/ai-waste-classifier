import torch

from PIL import Image
from torchvision import transforms

from src.model import create_model


CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]


def test_trained_model_prediction():
    model = create_model(
        num_classes=len(CLASSES)
    )

    model.load_state_dict(
        torch.load(
            "models/best_waste_classifier.pth",
            map_location="cpu"
        )
    )

    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = Image.open(
        "examples/bottle.jpg"
    ).convert("RGB")

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

    confidence, predicted_index = torch.max(
        probabilities,
        dim=1
    )

    predicted_class = CLASSES[
        predicted_index.item()
    ]

    confidence = confidence.item()

    assert predicted_class in CLASSES
    assert 0.0 <= confidence <= 1.0
    assert probabilities.shape == (1, 6)
    assert torch.isclose(
        probabilities.sum(),
        torch.tensor(1.0),
        atol=1e-5
    )