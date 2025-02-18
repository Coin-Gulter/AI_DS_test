import torch
import torch.nn as nn
from transformers import AutoModelForTokenClassification, AutoTokenizer
from torchvision import models, transforms
from PIL import Image
import argparse

# Load NER Model & Tokenizer
NER_MODEL_PATH = "./models/ner"
ner_tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_PATH)
ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_PATH)
ner_model.eval()

# Load Image Classification Model
CLASSIFIER_MODEL_PATH = "./models/image_classification/image_classifier.pth"
image_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
image_model.fc = nn.Linear(image_model.fc.in_features, 10)  # 10 animal classes
image_model.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH))
image_model.eval()

# Define Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

ANIMAL_CLASSES = ["butterfly", "cat", "cow", "dog", "elephant", "horse", "chicken", "sheep", "spider", "squirrel"]

# Function to Extract Animals from Text
def extract_animal(text):
    inputs = ner_tokenizer(text, return_tensors="pt")
    outputs = ner_model(**inputs).logits
    predictions = torch.argmax(outputs, dim=-1)[0].tolist()
    tokens = ner_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # Debug: Print token and label mapping
    print("Tokenized Text and Labels:")
    for token, label in zip(tokens, predictions):
        print(f"{token}: {label}")
    
    # Extract only tokens labeled as B-Animal (assuming label 1 means B-Animal)
    extracted_animals = [tokens[i] for i, label in enumerate(predictions) if label == 1]
    
    return set(extracted_animals)

# Function to Classify Image
def predict_animal(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = image_model(image)
    predicted_class = torch.argmax(output, dim=1).item()
    return ANIMAL_CLASSES[predicted_class]

# Main Verification Function
def verify_animal(text, image_path):
    extracted_animals = extract_animal(text)
    predicted_animal = predict_animal(image_path)
    print(f"Extracted from Text: {extracted_animals}")
    print(f"Predicted from Image: {predicted_animal}")
    return predicted_animal in extracted_animals


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Input text describing the image")
    parser.add_argument("--image", type=str, help="Path to the image file")
    args = parser.parse_args()
    
    result = verify_animal(args.text, args.image)
    print(f"Verification Result: {result}")
