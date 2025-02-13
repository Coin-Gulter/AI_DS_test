import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from mnist_classifier import mnist_classifier_interface

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32*7*7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32*7*7)
        x = self.relu(self.fc1(x))
        return self.fc2(x)
    
class CNNClassifier(mnist_classifier_interface.MnistClassifierInterface):
    def __init__(self, lr=0.01):
        self.model = CNN()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, train_loader, test_loader, epochs=10):
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            for images, labels in tqdm(train_loader):
                self.optimizer.zero_grad()
                output = self.model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()

        return self.test(test_loader)
    
    def test(self, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            print("Testing")
            for images, labels in tqdm(test_loader):
                output = self.model(images).argmax(dim=1)
                total += labels.size(0)
                correct += (output == labels).sum().item()
        return correct / total
    
    def predict(self, input):
        self.model.eval()
        with torch.no_grad():
            output = self.model(input).argmax(dim=1)
        return output