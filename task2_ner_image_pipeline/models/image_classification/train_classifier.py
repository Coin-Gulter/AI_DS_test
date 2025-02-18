import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import random

DATASET_DIR = "./dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")

# Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=transform)
val_dataset = datasets.ImageFolder(root=VAL_DIR, transform=transform)

# Define fraction to use (e.g., 20% of data)
FRACTION = 0.2  

# Get indices and randomly select a subset
train_indices = random.sample(range(len(train_dataset)), int(len(train_dataset) * FRACTION))
val_indices = random.sample(range(len(val_dataset)), int(len(val_dataset) * FRACTION))

# Create subset datasets
train_dataset = Subset(train_dataset, train_indices)
val_dataset = Subset(val_dataset, val_indices)

# Update DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

# Define the Model (ResNet-18)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, len(train_dataset.dataset.classes))  # Adjust final layer
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop with Validation
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        print(f"Train Loss = {running_loss/len(train_loader):.4f}, Train Acc = {train_acc:.2f}%")
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        print("Validating...")
        with torch.no_grad():
            for images, labels in tqdm(val_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * correct / total
        
        print(f"Epoch {epoch+1}: Train Loss = {running_loss/len(train_loader):.4f}, Train Acc = {train_acc:.2f}%, Val Loss = {val_loss/len(val_loader):.4f}, Val Acc = {val_acc:.2f}%")

if __name__ == "__main__":
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=1)

    MODEL_PATH = "./models/image_classification/image_classifier.pth"
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved at {MODEL_PATH}")
