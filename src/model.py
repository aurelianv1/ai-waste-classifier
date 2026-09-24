import torch.nn as nn

from torchvision.models import (
    mobilenet_v2,
    MobileNet_V2_Weights
)


def create_model(num_classes, pretrained=False, freeze_features=False):
    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None

    model = mobilenet_v2(weights=weights)

    if freeze_features:
        for parameter in model.features.parameters():
            parameter.requires_grad = False

    input_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        input_features,
        num_classes
    )

    return model