import os
import zipfile
import random
import shutil
import torch
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from dotenv import load_dotenv

load_dotenv()

import kaggle



# -----------------------------
# 1. Download the Dataset
# -----------------------------
KAGGLE_DATASET = "alessiocorrado99/animals10"
DATASET_PATH = "./dataset"

os.makedirs(DATASET_PATH, exist_ok=True)

# Ensure dataset is structured correctly
DATA_DIR = os.path.join(DATASET_PATH, "raw-img")
TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VAL_DIR = os.path.join(DATASET_PATH, "val")
TEST_DIR = os.path.join(DATASET_PATH, "test")

# Download dataset
print("Downloading dataset...")
kaggle.api.dataset_download_files(KAGGLE_DATASET, path=DATASET_PATH, unzip=True)
print("Download complete!")

# -----------------------------
# 2. Translate the Folders
# -----------------------------
translate = {"cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel", "ragno": "spider"}
print("Renaming dataset folders...")
# Define class labels (from dataset info)
ANIMAL_CLASSES = list(translate.values())

animal_list_fr = os.listdir(DATA_DIR)

for folder_fr in animal_list_fr:
    folder_from = os.path.join(DATA_DIR, folder_fr)
    folder_to = os.path.join(DATA_DIR, translate[folder_fr])
    os.rename(folder_from, folder_to)
print("Renaming complete!")


for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    os.makedirs(folder, exist_ok=True)
    for cls in ANIMAL_CLASSES:
        os.makedirs(os.path.join(folder, cls), exist_ok=True)

# -----------------------------
# 3. Split into Train/Val/Test
# -----------------------------
TRAIN_RATIO, VAL_RATIO = 0.8, 0.1  # 80% Train, 10% Val, 10% Test
print("Splitting dataset...")

for cls in ANIMAL_CLASSES:
    cls_path = os.path.join(DATA_DIR, cls)
    images = os.listdir(cls_path)
    random.shuffle(images)
    
    train_split = int(len(images) * TRAIN_RATIO)
    val_split = int(len(images) * VAL_RATIO)
    
    train_imgs = images[:train_split]
    val_imgs = images[train_split:train_split + val_split]
    test_imgs = images[train_split + val_split:]
    
    for img in train_imgs:
        shutil.move(os.path.join(cls_path, img), os.path.join(TRAIN_DIR, cls, img))
    for img in val_imgs:
        shutil.move(os.path.join(cls_path, img), os.path.join(VAL_DIR, cls, img))
    for img in test_imgs:
        shutil.move(os.path.join(cls_path, img), os.path.join(TEST_DIR, cls, img))

shutil.rmtree(DATA_DIR)
print("Dataset successfully split into Train/Val/Test!")

# -----------------------------
# 4. Define Transformations
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# -----------------------------
# 5. Load Processed Dataset
# -----------------------------
print("Loading preprocessed dataset...")
train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=transform)
val_dataset = datasets.ImageFolder(root=VAL_DIR, transform=transform)
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=transform)

# Create DataLoaders
def get_dataloader(dataset, batch_size=32):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

train_loader = get_dataloader(train_dataset)
val_loader = get_dataloader(val_dataset)
test_loader = get_dataloader(test_dataset)

print("Dataset preprocessing complete! 🚀")
