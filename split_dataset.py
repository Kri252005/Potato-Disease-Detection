import os
import shutil
import random

# Paths
base_dir = "dataset_original"   # where you extracted Kaggle dataset
output_dir = "dataset"          # where train/val folders will be created

# Classes in dataset
classes = ["Potato___Early_blight", "Potato___Healthy", "Potato___Late_blight"]

# Create output directories
for split in ["train", "val"]:
    for cls in ["Early_Blight", "Healthy", "Other"]:
        os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

# Split ratio
train_ratio = 0.7

for cls in classes:
    # Map Kaggle class names to your custom labels
    if cls == "Potato___Early_blight":
        label = "Early_Blight"
    elif cls == "Potato___Healthy":
        label = "Healthy"
    else:  # Potato___Late_blight
        label = "Other"

    src_dir = os.path.join(base_dir, cls)
    images = os.listdir(src_dir)
    random.shuffle(images)

    split_idx = int(len(images) * train_ratio)
    train_files = images[:split_idx]
    val_files = images[split_idx:]

    for f in train_files:
        shutil.copy(os.path.join(src_dir, f), os.path.join(output_dir, "train", label, f))
    for f in val_files:
        shutil.copy(os.path.join(src_dir, f), os.path.join(output_dir, "val", label, f))

print("✅ Dataset split completed: 70% train, 30% val")
