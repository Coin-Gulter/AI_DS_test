### **📌 Task 1: MNIST Image Classification with OOP**
This task implements **three different classifiers** for **MNIST digit classification** using **Object-Oriented Programming (OOP) principles** in Python.

### **🚀 Features**
✅ Implements three models:
- **Random Forest (RF)** (using `sklearn.ensemble.RandomForestClassifier`)
- **Feed-Forward Neural Network (FNN)** (using `PyTorch`)
- **Convolutional Neural Network (CNN)** (using `PyTorch`)

✅ Supports **training and inference** for each model.

✅ Includes **Jupyter Notebook** for **demonstration and edge cases**.

---

## **📁 Task Structure**
```
task1_image_classification/
│── mnist_classifier/
│   │── cnn_classifier.py              # Convolutional Neural Network
│   │── mnist_classifier_interface.py  # Abstract interface
│   │── mnist_classifier.py            # Unified classifier class
│   │── neural_network_classifier.py   # Feed-Forward Neural Network
│   │── random_forest_classifier.py    # Random Forest model
│── demo.ipynb                           # Jupyter Notebook for demonstration
│── inference.py                        # Inference script
│── README.md                            # Project documentation
│── requirements.txt                     # Dependencies
│── train.py                            # Training script
```

---

## **🔧 Installation & Setup**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/Coin-Gulter/AI_DS_test.git
cd task1_image_classification
```

### **2️⃣ Create Virtual Environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
.venv\Scripts\activate     # On Windows
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## **📊 Dataset**
This project uses the **MNIST dataset** (28×28 grayscale images of handwritten digits 0-9).

The datasets for train and testing are automatically downloaded in the code using:
```python
from torchvision import datasets
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
```

---

## **🛠 Training the Models**
To train a specific model, run:

```bash
python train.py --model cnn --epochs 5 --model_file ./model.pth # Options: cnn, rf, nn
```

Example:
```bash
python train.py --model rf --model_file ./model.pkl # Train Random Forest
```

Use model file with extension model.pth for the models 'cnn' and 'nn' and file with extension model.pkl with model 'rf'


### **Training Script (`train.py`)**
- Loads the **MNIST dataset**.
- Trains the specified **classifier (`cnn`, `rf`, `nn`)**.
- Saves the trained model.

---

## **🖼️ Running Inference**
To test a trained model:

```bash
python inference.py --model cnn --image test_image.png, --model_file ./model.pth
```
Use model file with extension model.pth for the models 'cnn' and 'nn' and file with extension model.pkl with model 'rf'

## **📝 Code Overview**
### **1️⃣ Interface: `MnistClassifierInterface`**
```python
from abc import ABC, abstractmethod

class MnistClassifierInterface(ABC):
    @abstractmethod
    def train(self, train_loader, test_loader):
        pass

    @abstractmethod
    def predict(self, X):
        pass
```
This ensures **all models** implement **train()** and **predict()**.

---

### **2️⃣ Random Forest Classifier**
```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from mnist_classifier.mnist_classifier_interface import MnistClassifierInterface

class RandomForestClassifierModel(MnistClassifierInterface):
    def __init__(self, n_estimators=100):
        self.model = RandomForestClassifier(n_estimators=n_estimators)

    def train(self, train_loader, test_loader):
        X_train, y_train = self._extract_data(train_loader)
        self.model.fit(X_train, y_train)

    def predict(self, X):
        X = X.view(X.shape[0], -1).numpy()
        return self.model.predict(X)
```
**Key Points**:
- Uses **sklearn's RandomForestClassifier**.
- **Flattens** images before training.

---

### **3️⃣ Feed-Forward Neural Network**
```python
import torch
import torch.nn as nn
import torch.optim as optim
from mnist_classifier.mnist_classifier_interface import MnistClassifierInterface

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

```
**Key Points**:
- **Fully connected network (FNN)**.
- Uses **ReLU activation** and **CrossEntropyLoss**.

---

### **4️⃣ Convolutional Neural Network (CNN)**
```python
import torch
import torch.nn as nn
import torch.optim as optim
from mnist_classifier.mnist_classifier_interface import MnistClassifierInterface

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
```
**Key Points**:
- **Two convolutional layers** + **MaxPooling**.
- **Fully connected layers** for classification.

---

### **5️⃣ Unified Classifier (`MnistClassifier`)**
```python
from mnist_classifier.random_forest_classifier import RandomForestClassifierModel
from mnist_classifier.neural_network_classifier import NeuralNetworkClassifier
from mnist_classifier.cnn_classifier import CNNClassifier

class MnistClassifier:
    def __init__(self, algorithm):
        if algorithm == "rf":
            self.model = RandomForestClassifierModel()
        elif algorithm == "nn":
            self.model = NeuralNetworkClassifier()
        elif algorithm == "cnn":
            self.model = CNNClassifier()
        else:
            raise ValueError("Invalid algorithm. Choose from 'rf', 'nn', 'cnn'.")

    def train(self, train_loader, test_loader):
        self.model.train(train_loader, test_loader)

    def predict(self, X):
        return self.model.predict(X)
```

---

## **📈 Performance Comparison**
| Model | Accuracy |
|--------|----------|
| **Random Forest** | ~94% |
| **Feed-Forward NN** | ~98% |
| **CNN** | **~99.2%** ✅ |
---