import torch
from torchvision import transforms
from PIL import Image
from mnist_classifier.mnist_classifier import MnistClassifier

# Load model
classifier = MnistClassifier(algorithm="cnn")

# Load image
transform = transforms.ToTensor()
image = Image.open("test_digit.png").convert("L")  # Convert to grayscale
image = transform(image).unsqueeze(0)  # Add batch dimension

# Predict
prediction = classifier.predict(image)
print("Predicted Label:", prediction.item())
