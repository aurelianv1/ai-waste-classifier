import torch

from src.model import create_model


def test_model_has_correct_number_of_classes():
    model = create_model(
        num_classes=6
    )

    assert model.classifier[1].out_features == 6


def test_model_accepts_image():
    model = create_model(
        num_classes=6
    )

    model.eval()

    image = torch.randn(
        1,
        3,
        224,
        224
    )

    with torch.no_grad():
        output = model(image)

    assert output.shape == (1, 6)