#!/usr/bin/env python3
"""
Download and prepare the real fish dataset from Kaggle.
This replaces the synthetic sample data with actual fish images.
"""

import kagglehub
import os
import shutil
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split
import random

def download_dataset():
    """Download the fish dataset from Kaggle."""
    print("🐟 Downloading real fish dataset from Kaggle...")
    
    # Download latest version
    path = kagglehub.dataset_download("markdaniellampa/fish-dataset")
    print(f"✅ Dataset downloaded to: {path}")
    
    return path

def explore_dataset(dataset_path):
    """Explore the structure of the downloaded dataset."""
    print("\n📁 Exploring dataset structure...")
    
    dataset_path = Path(dataset_path)
    
    # List all files and directories
    for root, dirs, files in os.walk(dataset_path):
        level = root.replace(str(dataset_path), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files in each directory
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files) - 5} more files")
    
    return dataset_path

def organize_for_training(dataset_path):
    """Organize the dataset for our training pipeline."""
    print("\n🔧 Organizing dataset for training...")
    
    dataset_path = Path(dataset_path)
    output_path = Path("data/raw/fish_images")
    
    # Create output directories
    for split in ["train", "val", "test"]:
        (output_path / split).mkdir(parents=True, exist_ok=True)
    
    # Find all fish species directories
    species_dirs = []
    image_files = []
    
    # Look for fish species in the dataset
    for item in dataset_path.rglob("*"):
        if item.is_dir() and any(img_ext in str(item) for img_ext in ['.jpg', '.jpeg', '.png']):
            continue
        if item.is_dir():
            # Check if directory contains images
            image_count = len(list(item.glob("*.jpg")) + list(item.glob("*.jpeg")) + list(item.glob("*.png")))
            if image_count > 0:
                species_dirs.append(item)
                print(f"   Found species: {item.name} ({image_count} images)")
    
    # If no species directories found, look for images directly
    if not species_dirs:
        print("   No species directories found, looking for images directly...")
        all_images = list(dataset_path.rglob("*.jpg")) + list(dataset_path.rglob("*.jpeg")) + list(dataset_path.rglob("*.png"))
        print(f"   Found {len(all_images)} total images")
        
        # Group images by parent directory name as species
        species_groups = {}
        for img_path in all_images:
            species_name = img_path.parent.name
            if species_name not in species_groups:
                species_groups[species_name] = []
            species_groups[species_name].append(img_path)
        
        # Create species directories
        for species_name, images in species_groups.items():
            if len(images) >= 10:  # Only use species with at least 10 images
                print(f"   Processing {species_name}: {len(images)} images")
                
                # Split images into train/val/test
                train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
                val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)
                
                # Copy images to appropriate directories
                for split, img_list in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
                    split_dir = output_path / split / species_name
                    split_dir.mkdir(parents=True, exist_ok=True)
                    
                    for i, img_path in enumerate(img_list):
                        dest_path = split_dir / f"{species_name}_{split}_{i:03d}{img_path.suffix}"
                        shutil.copy2(img_path, dest_path)
                
                print(f"      Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")
    
    else:
        # Process species directories
        for species_dir in species_dirs:
            species_name = species_dir.name
            images = list(species_dir.glob("*.jpg")) + list(species_dir.glob("*.jpeg")) + list(species_dir.glob("*.png"))
            
            if len(images) >= 10:  # Only use species with at least 10 images
                print(f"   Processing {species_name}: {len(images)} images")
                
                # Split images into train/val/test (70/20/10)
                train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
                val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.33, random_state=42)
                
                # Copy images to appropriate directories
                for split, img_list in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
                    split_dir = output_path / split / species_name
                    split_dir.mkdir(parents=True, exist_ok=True)
                    
                    for i, img_path in enumerate(img_list):
                        dest_path = split_dir / f"{species_name}_{split}_{i:03d}{img_path.suffix}"
                        shutil.copy2(img_path, dest_path)
                
                print(f"      Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")
    
    return output_path

def count_dataset_stats(dataset_path):
    """Count and display dataset statistics."""
    print("\n📊 DATASET STATISTICS")
    print("=" * 40)
    
    total_images = 0
    species_count = 0
    
    for split in ["train", "val", "test"]:
        split_path = dataset_path / split
        if split_path.exists():
            split_images = len(list(split_path.rglob("*.jpg")) + list(split_path.rglob("*.jpeg")) + list(split_path.rglob("*.png")))
            total_images += split_images
            print(f"   {split.upper()}: {split_images:,} images")
            
            if split == "train":
                species_dirs = [d for d in split_path.iterdir() if d.is_dir()]
                species_count = len(species_dirs)
                print(f"   SPECIES: {species_count} classes")
                
                for species_dir in species_dirs:
                    species_images = len(list(species_dir.glob("*")))
                    print(f"      • {species_dir.name}: {species_images} images")
    
    print(f"   TOTAL: {total_images:,} images")
    
    return species_count, total_images

def update_config(species_count, species_names):
    """Update the configuration file with actual species found."""
    print("\n⚙️ Updating configuration...")
    
    config_path = Path("configs/real_fish_config.yaml")
    
    config = {
        'project_name': 'real-fish-classifier',
        'project': {
            'name': 'real-fish-classifier',
            'description': 'Fish classification using real Kaggle fish dataset'
        },
        'model': {
            'architecture': 'resnet50',
            'pretrained': True,
            'num_classes': species_count,
            'dropout': 0.3
        },
        'training': {
            'epochs': 20,  # More epochs for real data
            'learning_rate': 0.001,
            'batch_size': 32,  # Larger batch size
            'optimizer': 'adam',
            'scheduler': 'cosine',
            'early_stopping_patience': 7
        },
        'data': {
            'image_size': [224, 224],
            'train_split': 0.7,
            'val_split': 0.2,
            'test_split': 0.1,
            'augmentation': {
                'horizontal_flip': 0.5,
                'rotation': 15,
                'brightness': 0.2,
                'contrast': 0.2,
                'saturation': 0.2
            }
        },
        'species': {
            'classes': species_names
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"✅ Configuration saved to: {config_path}")
    return config_path

def main():
    """Main function to download and prepare the real dataset."""
    print("🚀 REAL FISH DATASET SETUP")
    print("=" * 50)
    
    # Download dataset
    dataset_path = download_dataset()
    
    # Explore structure
    dataset_path = explore_dataset(dataset_path)
    
    # Organize for training
    organized_path = organize_for_training(dataset_path)
    
    # Count statistics
    species_count, total_images = count_dataset_stats(organized_path)
    
    # Get species names
    train_path = organized_path / "train"
    species_names = [d.name for d in train_path.iterdir() if d.is_dir()] if train_path.exists() else []
    
    # Update config
    config_path = update_config(species_count, species_names)
    
    print("\n🎯 READY TO TRAIN WITH REAL DATA!")
    print("=" * 40)
    print(f"✅ Dataset: {total_images:,} real fish images")
    print(f"✅ Species: {species_count} different fish classes")
    print(f"✅ Config: {config_path}")
    
    print("\n🚀 Start training with:")
    print(f"   python scripts/train.py --data_path {organized_path} --config {config_path} --experiment_name real_fish_classifier")

if __name__ == "__main__":
    main() 