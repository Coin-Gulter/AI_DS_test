from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from mnist_classifier.mnist_classifier import MnistClassifier


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    
    classifier = MnistClassifier(algorithm='cnn')
    print(f"test result: {classifier.train(train_loader, test_loader, epochs=5)}")
    print(classifier.predict(test_loader.dataset[0][0].unsqueeze(0)))