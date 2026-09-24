import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from dataset import test_loader, test_dataset
from model import create_model


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_CLASSES = len(test_dataset.classes)


# Create model architecture
model = create_model(
    num_classes=NUM_CLASSES
)


# Load best checkpoint
model.load_state_dict(
    torch.load(
        "models/best_waste_classifier.pth",
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()


all_predictions = []
all_labels = []

correct = 0
total = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        _, predictions = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


accuracy = 100 * correct / total

print(
    f"\nTest Accuracy: "
    f"{accuracy:.2f}%\n"
)


print("Classification Report:\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=test_dataset.classes,
        digits=4,
        zero_division=0
    )
)


cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("Confusion Matrix:\n")
print(cm)


# Create confusion matrix visualization
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=test_dataset.classes
)

fig, ax = plt.subplots(
    figsize=(8, 8)
)

display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d"
)

plt.title(
    "Waste Classification - Confusion Matrix"
)

plt.xticks(rotation=45)
plt.tight_layout()


plt.savefig(
    "results/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "\nConfusion matrix saved to "
    "results/confusion_matrix.png"
)