# Import Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# ✅ STEP 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# ✅ STEP 2: Update Dataset Path
root_dir = "/content/drive/MyDrive/IR_Spectro/data/"

# ✅ STEP 3: Update CNN Model (Supports More Functional Groups)
class IR_CNN(nn.Module):
    def __init__(self, num_classes):
        super(IR_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)  # Adjust dynamically

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ✅ STEP 4: Load Data (Automatically Detects Four Classes)
def load_data(root_dir):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.3,), (0.7,))
    ])

    # ✅ Debugging: Check if dataset exists
    if not os.path.exists(root_dir + "train/"):
        raise FileNotFoundError(f"❌ Dataset not found in {root_dir}. Check your Google Drive path!")

    # ✅ Load training data
    train_dataset = datasets.ImageFolder(root=root_dir + "train/", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    return train_loader, train_dataset

# ✅ STEP 5: Train Model
def train_model(model, train_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    num_epochs = 50  # Increase epochs for better training

    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}')

    torch.save(model.state_dict(), "/content/drive/MyDrive/ir_model.pth")  # ✅ Save model

# ✅ STEP 6: Evaluate Model
def evaluate_model(model, test_loader, class_names):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.append(labels.item())
            y_pred.append(predicted.item())

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

# ✅ STEP 7: Run Training & Evaluation
train_loader, train_dataset = load_data(root_dir)
num_classes = len(train_dataset.classes)

# Initialize model (Dynamically adapts to 4 classes)
model = IR_CNN(num_classes)
train_model(model, train_loader)

# ✅ Load Test Data
test_dataset = datasets.ImageFolder(root=root_dir + "test/", transform=transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.3,), (0.7,))
]))
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Evaluate Model
evaluate_model(model, test_loader, train_dataset.classes)