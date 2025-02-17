import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from mnist_classifier.mnist_classifier import MnistClassifier


def main(model_type: str, epochs: int, model_file: str):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    
    classifier = MnistClassifier(algorithm=model_type)
    print(f"test result: {classifier.train(train_loader, test_loader, epochs=epochs)}")
    print(f"test prediction: {classifier.predict(test_loader.dataset[0][0].unsqueeze(0))}")
    classifier.save_weights(model_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a MNIST classifier (cnn, rf, nn).")
    parser.add_argument("--model", type=str, choices=["cnn", "rf", "nn"], required=True, help="Choose a model: cnn, rf, nn")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--model_file", type=str, default="./model.pth", help="Path to the model weights with extension .pth for 'cnn' or 'nn' and with .pkl for 'rf'.")

    args = parser.parse_args()

    main(model_type=args.model, epochs=args.epochs, model_file=args.model_file)