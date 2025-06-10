"""
Fish Dataset Module

Handles loading, preprocessing, and augmentation of fish images for classification.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class FishDataset(Dataset):
    """
    Custom Dataset for fish classification with support for data augmentation
    and quality validation.
    """
    
    def __init__(
        self,
        data_path: str,
        species_mapping: Dict[str, int],
        transform: Optional[A.Compose] = None,
        image_size: Tuple[int, int] = (224, 224),
        mode: str = "train"
    ):
        """
        Initialize the Fish Dataset.
        
        Args:
            data_path: Path to the image data directory
            species_mapping: Dictionary mapping species names to class indices
            transform: Albumentations transform pipeline
            image_size: Target image size (height, width)
            mode: Dataset mode ("train", "val", "test")
        """
        self.data_path = data_path
        self.species_mapping = species_mapping
        self.transform = transform
        self.image_size = image_size
        self.mode = mode
        
        # Load image paths and labels
        self.samples = self._load_samples()
        
        logger.info(f"Loaded {len(self.samples)} samples for {mode} mode")
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load image paths and corresponding labels."""
        samples = []
        
        for species_name, class_idx in self.species_mapping.items():
            species_path = os.path.join(self.data_path, species_name)
            
            if not os.path.exists(species_path):
                logger.warning(f"Species directory not found: {species_path}")
                continue
            
            for img_file in os.listdir(species_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(species_path, img_file)
                    samples.append((img_path, class_idx))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image_tensor, label)
        """
        img_path, label = self.samples[idx]
        
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
            
            # Apply transforms
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']
            else:
                # Basic preprocessing if no transform specified
                transform = A.Compose([
                    A.Resize(*self.image_size),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])
                transformed = transform(image=image)
                image = transformed['image']
            
            return image, label
            
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {str(e)}")
            # Return a blank image in case of error
            blank_image = torch.zeros(3, *self.image_size)
            return blank_image, label


class FishDataModule:
    """
    Data module for managing fish datasets with train/val/test splits.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data module.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.species_mapping = {
            species: idx for idx, species in enumerate(config['species']['classes'])
        }
        self.num_classes = len(self.species_mapping)
        
        # Setup transforms
        self.train_transform = self._get_train_transforms()
        self.val_transform = self._get_val_transforms()
        
    def _get_train_transforms(self) -> A.Compose:
        """Get training data augmentation transforms."""
        aug_config = self.config['data']['augmentation']
        
        return A.Compose([
            A.Resize(*self.config['data']['image_size']),
            A.HorizontalFlip(p=aug_config.get('horizontal_flip', 0.5)),
            A.VerticalFlip(p=aug_config.get('vertical_flip', 0.2)),
            A.Rotate(limit=aug_config.get('rotation', 15), p=0.5),
            A.ColorJitter(
                brightness=aug_config.get('brightness', 0.2),
                contrast=aug_config.get('contrast', 0.2),
                saturation=aug_config.get('saturation', 0.2),
                hue=aug_config.get('hue', 0.1),
                p=0.5
            ),
            A.GaussianBlur(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def _get_val_transforms(self) -> A.Compose:
        """Get validation/test transforms (no augmentation)."""
        return A.Compose([
            A.Resize(*self.config['data']['image_size']),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def get_dataloaders(self, data_path: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, validation, and test dataloaders.
        
        Args:
            data_path: Path to the dataset
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Create datasets
        train_dataset = FishDataset(
            data_path=os.path.join(data_path, 'train'),
            species_mapping=self.species_mapping,
            transform=self.train_transform,
            image_size=self.config['data']['image_size'],
            mode='train'
        )
        
        val_dataset = FishDataset(
            data_path=os.path.join(data_path, 'val'),
            species_mapping=self.species_mapping,
            transform=self.val_transform,
            image_size=self.config['data']['image_size'],
            mode='val'
        )
        
        test_dataset = FishDataset(
            data_path=os.path.join(data_path, 'test'),
            species_mapping=self.species_mapping,
            transform=self.val_transform,
            image_size=self.config['data']['image_size'],
            mode='test'
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def get_class_weights(self, data_path: str) -> torch.Tensor:
        """
        Calculate class weights for imbalanced dataset handling.
        
        Args:
            data_path: Path to the dataset
            
        Returns:
            Tensor of class weights
        """
        # Count samples per class
        class_counts = torch.zeros(self.num_classes)
        
        for species_name, class_idx in self.species_mapping.items():
            species_path = os.path.join(data_path, 'train', species_name)
            if os.path.exists(species_path):
                count = len([f for f in os.listdir(species_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                class_counts[class_idx] = count
        
        # Calculate weights (inverse frequency)
        total_samples = class_counts.sum()
        class_weights = total_samples / (self.num_classes * class_counts)
        
        # Handle zero counts
        class_weights[class_counts == 0] = 0
        
        return class_weights 