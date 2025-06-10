#!/usr/bin/env python3
"""
Download sample fish dataset for testing the aquaculture classifier.
This script downloads a small sample dataset to get you started.
"""

import os
import requests
import zipfile
from pathlib import Path
import shutil

def create_directory_structure():
    """Create the required directory structure for fish images."""
    base_path = Path("data/raw/fish_images")
    
    # Create directories for train/val/test splits
    for split in ["train", "val", "test"]:
        for species in ["Atlantic_Salmon", "Rainbow_Trout", "Cod", "Sea_Bass", "Sea_Bream"]:
            (base_path / split / species).mkdir(parents=True, exist_ok=True)
    
    print("✅ Created directory structure")
    return base_path

def download_sample_images():
    """Download sample fish images from public sources."""
    
    # Sample URLs for different fish species (these would be actual image URLs)
    sample_urls = {
        "Atlantic_Salmon": [
            "https://example.com/salmon1.jpg",  # Replace with actual URLs
            "https://example.com/salmon2.jpg",
        ],
        "Rainbow_Trout": [
            "https://example.com/trout1.jpg",
            "https://example.com/trout2.jpg",
        ],
        # Add more species as needed
    }
    
    print("📂 Sample dataset structure created!")
    print("🔗 To get actual fish images, try these sources:")
    print("\n1. Roboflow Universe:")
    print("   https://universe.roboflow.com/innoweave/kaggle-fish-detection-o8ghb")
    print("\n2. Kaggle Datasets:")
    print("   https://www.kaggle.com/search?q=fish+classification")
    print("\n3. Manual download instructions:")
    print("   - Visit the above links")
    print("   - Download datasets in image classification format")
    print("   - Extract to data/raw/fish_images/ following the structure:")
    print("     data/raw/fish_images/train/Atlantic_Salmon/")
    print("     data/raw/fish_images/train/Rainbow_Trout/")
    print("     etc.")

def create_sample_config():
    """Create a sample config for smaller dataset."""
    sample_config = """
# Sample configuration for testing with small dataset
project:
  name: "sample-fish-classifier"
  description: "Sample aquaculture fish classification"

model:
  architecture: "resnet18"  # Smaller model for testing
  pretrained: true
  num_classes: 5  # Reduced for sample dataset
  dropout: 0.3

training:
  epochs: 5  # Few epochs for testing
  learning_rate: 0.001
  batch_size: 8  # Small batch for testing
  optimizer: "adam"
  scheduler: "cosine"

data:
  image_size: [224, 224]
  train_split: 0.7
  val_split: 0.2
  test_split: 0.1

species:
  classes:
    - "Atlantic_Salmon"
    - "Rainbow_Trout" 
    - "Cod"
    - "Sea_Bass"
    - "Sea_Bream"
"""
    
    with open("configs/sample_config.yaml", "w") as f:
        f.write(sample_config)
    
    print("✅ Created sample configuration: configs/sample_config.yaml")

if __name__ == "__main__":
    print("🐟 Setting up sample fish dataset...")
    
    # Create directory structure
    base_path = create_directory_structure()
    
    # Create sample config
    create_sample_config()
    
    # Show download instructions
    download_sample_images()
    
    print("\n🎯 Next Steps:")
    print("1. Download actual fish images from the provided sources")
    print("2. Organize them in the created directory structure")
    print("3. Run: python scripts/train.py --config configs/sample_config.yaml")
    
    print("\n📁 Directory structure created:")
    print(f"   {base_path}") 