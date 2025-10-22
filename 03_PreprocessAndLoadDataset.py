# 1. Imports
import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import pandas as pd

# 2. Dataset Filtering Function
def filter_imagefolder_dataset(imagefolder_dataset):
    valid_samples = []
    for path, class_idx in tqdm(imagefolder_dataset.samples, desc="Filtering dataset"):
        try:
            with Image.open(path) as img:
                img.load()  # Isso força a leitura completa da imagem
            valid_samples.append((path, class_idx))
        except (UnidentifiedImageError, OSError):
            print(f"Imagem corrompida ignorada: {path}")
            continue
    imagefolder_dataset.samples = valid_samples
    return imagefolder_dataset

# 3. Configuration
DATASET_PATH = 'data/raw'
PROCESSED_PATH = 'data/processed'
OUTPUT_PATH = 'output'
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
NUM_WORKERS = 4

os.makedirs(PROCESSED_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 4. Transforms
transform_train = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

transform_val_test = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# 5. Load and filter datasets
train_folder = datasets.ImageFolder(root=os.path.join(DATASET_PATH, 'train'), transform=transform_train)
train_folder = filter_imagefolder_dataset(train_folder)

test_folder = datasets.ImageFolder(root=os.path.join(DATASET_PATH, 'test'), transform=transform_val_test)
test_folder = filter_imagefolder_dataset(test_folder)

# 6. Print class info
print("Classes:", train_folder.classes)
print("Class-to-index mapping:", train_folder.class_to_idx)

# 7. Split train/val
val_size = int(0.2 * len(train_folder))
train_size = len(train_folder) - val_size
train_dataset, val_dataset = random_split(train_folder, [train_size, val_size])
test_dataset = test_folder

# 8. Final summary
total = len(train_dataset) + len(val_dataset) + len(test_dataset)

print("\nFinal dataset summary:")
print(f" - Train:      {len(train_dataset)} images ({100 * len(train_dataset)/total:.2f}%)")
print(f" - Validation: {len(val_dataset)} images ({100 * len(val_dataset)/total:.2f}%)")
print(f" - Test:       {len(test_dataset)} images ({100 * len(test_dataset)/total:.2f}%)")
print(f" - TOTAL:      {total} images")

# 9. Save processed datasets
torch.save(train_dataset, os.path.join(PROCESSED_PATH, 'train_dataset.pt'))
torch.save(val_dataset, os.path.join(PROCESSED_PATH, 'val_dataset.pt'))
torch.save(test_dataset, os.path.join(PROCESSED_PATH, 'test_dataset.pt'))

print("\nProcessed datasets saved to:", PROCESSED_PATH)

# 10. Save summary CSV
summary_data = {
    "Split": ["Train", "Validation", "Test", "Total"],
    "Images": [len(train_dataset), len(val_dataset), len(test_dataset), total],
    "Percentage": [
        round(100 * len(train_dataset)/total, 2),
        round(100 * len(val_dataset)/total, 2),
        round(100 * len(test_dataset)/total, 2),
        100.0
    ]
}
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(os.path.join(OUTPUT_PATH, "02_dataset_summary.csv"), index=False)

print("CSV summary saved to:", os.path.join(OUTPUT_PATH, "02_dataset_summary.csv"))
