import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from dataset import train_loader, val_loader, train_dataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = len(train_dataset.classes)
EPOCHS = 10
LEARNING_RATE = 0.001


# 1. Încărcăm MobileNetV2 pre-antrenat
weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights)


# 2. Înghețăm partea care extrage features
for parameter in model.features.parameters():
    parameter.requires_grad = False


# 3. Înlocuim classifier-ul original
input_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    input_features,
    NUM_CLASSES
)

model = model.to(DEVICE)


# 4. Loss function
criterion = nn.CrossEntropyLoss()


# 5. Optimizer
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

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()


    train_accuracy = 100 * train_correct / train_total


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

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()


    val_accuracy = 100 * val_correct / val_total

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "models/best_waste_classifier.pth"
        )

        print(f"New best model saved: {val_accuracy:.2f}%")


    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss / len(train_loader):.4f} "
        f"Train Accuracy: {train_accuracy:.2f}% "
        f"Val Loss: {val_loss / len(val_loader):.4f} "
        f"Val Accuracy: {val_accuracy:.2f}%"
    )


# 6. Salvăm modelul
# torch.save(
#     model.state_dict(),
#     "models/waste_classifier.pth"
# )

print("Model saved to models/waste_classifier.pth")