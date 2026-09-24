import torch
import torch.optim as optim

from dataset import train_loader, val_loader, train_dataset
from model import create_model


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_CLASSES = len(train_dataset.classes)
EPOCHS = 10
LEARNING_RATE = 0.001


# Create pretrained MobileNetV2 model
model = create_model(
    num_classes=NUM_CLASSES,
    pretrained=True,
    freeze_features=True
)

model = model.to(DEVICE)


# Loss function
criterion = torch.nn.CrossEntropyLoss()


# Optimizer
optimizer = optim.Adam(
    model.classifier.parameters(),
    lr=LEARNING_RATE
)


print(f"Device: {DEVICE}")
print(f"Classes: {train_dataset.classes}")
print(f"Number of classes: {NUM_CLASSES}")


best_val_accuracy = 0.0


for epoch in range(EPOCHS):

    # ---------- TRAINING ----------
    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        train_total += labels.size(0)

        train_correct += (
            predicted == labels
        ).sum().item()


    train_accuracy = (
        100 * train_correct / train_total
    )


    # ---------- VALIDATION ----------
    model.eval()

    val_correct = 0
    val_total = 0
    val_loss = 0.0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


    val_accuracy = (
        100 * val_correct / val_total
    )


    # Save best checkpoint
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "models/best_waste_classifier.pth"
        )

        print(
            f"New best model saved: "
            f"{val_accuracy:.2f}%"
        )


    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: "
        f"{train_loss / len(train_loader):.4f} "
        f"Train Accuracy: "
        f"{train_accuracy:.2f}% "
        f"Val Loss: "
        f"{val_loss / len(val_loader):.4f} "
        f"Val Accuracy: "
        f"{val_accuracy:.2f}%"
    )


print(
    f"\nTraining complete. "
    f"Best validation accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    "Best model saved to "
    "models/best_waste_classifier.pth"
)