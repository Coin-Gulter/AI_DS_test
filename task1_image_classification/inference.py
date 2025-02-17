import argparse
from torchvision import transforms
from PIL import Image
from mnist_classifier.mnist_classifier import MnistClassifier


def main(model_type, image_path, model_file):
    classifier = MnistClassifier(algorithm=model_type)
    classifier.load_weights(path=model_file)

    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure single channel
        transforms.Resize((28, 28)),  # Resize to match MNIST input size
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize like training data
    ])

    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image = transform(image).unsqueeze(0)  # Add batch dimension

    prediction = classifier.predict(image)
    print(f"Predicted Label: {prediction.item()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on an image using a trained MNIST model.")
    parser.add_argument("--model", type=str, choices=["cnn", "rf", "nn"], required=True, help="Choose a model: cnn, rf, nn")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image (must be 28x28 grayscale).")
    parser.add_argument("--model_file", type=str, default="./model.pth", help="Path to the model weights with extension .pth for 'cnn' or 'nn' and with .pkl for 'rf'.")

    args = parser.parse_args()

    main(model_type=args.model, image_path=args.image, model_file=args.model_file)
