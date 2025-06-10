"""
Visualization utilities for fish classification project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_training_history(history: Dict[str, List[float]], 
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (15, 5)) -> None:
    """
    Plot training history including loss and accuracy curves.
    
    Args:
        history: Dictionary containing training metrics
        save_path: Optional path to save the plot
        figsize: Figure size tuple
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    axes[1].plot(history['train_acc'], label='Training Accuracy', linewidth=2)
    axes[1].plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         class_names: List[str],
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (10, 8),
                         normalize: bool = True) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Optional path to save the plot
        figsize: Figure size tuple
        normalize: Whether to normalize the confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_class_distribution(labels: np.ndarray,
                           class_names: List[str],
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (12, 6)) -> None:
    """
    Plot class distribution in the dataset.
    
    Args:
        labels: Array of labels
        class_names: List of class names
        save_path: Optional path to save the plot
        figsize: Figure size tuple
    """
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=figsize)
    bars = plt.bar(range(len(unique)), counts, color=sns.color_palette("husl", len(unique)))
    plt.title('Class Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Fish Species', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(range(len(unique)), [class_names[i] for i in unique], rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Class distribution plot saved to {save_path}")
    
    plt.show()


def visualize_predictions(images: torch.Tensor,
                         true_labels: torch.Tensor,
                         pred_labels: torch.Tensor,
                         class_names: List[str],
                         num_samples: int = 16,
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (15, 15)) -> None:
    """
    Visualize model predictions on sample images.
    
    Args:
        images: Batch of images
        true_labels: True labels
        pred_labels: Predicted labels
        class_names: List of class names
        num_samples: Number of samples to visualize
        save_path: Optional path to save the plot
        figsize: Figure size tuple
    """
    # Ensure we don't exceed available samples
    num_samples = min(num_samples, len(images))
    
    # Calculate grid size
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    axes = axes.flatten() if grid_size > 1 else [axes]
    
    for i in range(num_samples):
        # Convert tensor to numpy and denormalize
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        true_class = class_names[true_labels[i]]
        pred_class = class_names[pred_labels[i]]
        
        # Set color based on correctness
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        
        axes[i].imshow(img)
        axes[i].set_title(f'True: {true_class}\nPred: {pred_class}', 
                         fontsize=10, color=color, fontweight='bold')
        axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Model Predictions (Green=Correct, Red=Incorrect)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Predictions visualization saved to {save_path}")
    
    plt.show()


def plot_feature_maps(model: torch.nn.Module,
                     image: torch.Tensor,
                     layer_name: str,
                     save_path: Optional[str] = None,
                     figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Visualize feature maps from a specific layer.
    
    Args:
        model: PyTorch model
        image: Input image tensor (1, C, H, W)
        layer_name: Name of the layer to visualize
        save_path: Optional path to save the plot
        figsize: Figure size tuple
    """
    # Hook to capture feature maps
    features = {}
    
    def hook_fn(module, input, output):
        features[layer_name] = output
    
    # Register hook
    for name, module in model.named_modules():
        if name == layer_name:
            module.register_forward_hook(hook_fn)
            break
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(image)
    
    if layer_name not in features:
        logger.error(f"Layer {layer_name} not found in model")
        return
    
    # Get feature maps
    feature_maps = features[layer_name].squeeze(0)  # Remove batch dimension
    num_features = min(16, feature_maps.shape[0])  # Limit to 16 feature maps
    
    # Plot feature maps
    grid_size = int(np.ceil(np.sqrt(num_features)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    axes = axes.flatten() if grid_size > 1 else [axes]
    
    for i in range(num_features):
        feature_map = feature_maps[i].cpu().numpy()
        axes[i].imshow(feature_map, cmap='viridis')
        axes[i].set_title(f'Feature {i}', fontsize=10)
        axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(num_features, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Feature Maps from {layer_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature maps visualization saved to {save_path}")
    
    plt.show()


def plot_learning_curves(train_sizes: np.ndarray,
                        train_scores: np.ndarray,
                        val_scores: np.ndarray,
                        save_path: Optional[str] = None,
                        figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Plot learning curves to analyze model performance vs training set size.
    
    Args:
        train_sizes: Array of training sizes
        train_scores: Training scores for each size
        val_scores: Validation scores for each size
        save_path: Optional path to save the plot
        figsize: Figure size tuple
    """
    plt.figure(figsize=figsize)
    
    plt.plot(train_sizes, train_scores, 'o-', label='Training Score', linewidth=2)
    plt.plot(train_sizes, val_scores, 'o-', label='Validation Score', linewidth=2)
    
    plt.title('Learning Curves', fontsize=16, fontweight='bold')
    plt.xlabel('Training Set Size', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Learning curves saved to {save_path}")
    
    plt.show()


def create_classification_report_plot(y_true: np.ndarray,
                                    y_pred: np.ndarray,
                                    class_names: List[str],
                                    save_path: Optional[str] = None,
                                    figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Create a visual classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Optional path to save the plot
        figsize: Figure size tuple
    """
    # Get classification report as dict
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(report).iloc[:-1, :].T  # Exclude 'support' and summary stats
    
    # Create heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(df.iloc[:, :-1], annot=True, cmap='Blues', fmt='.3f')
    plt.title('Classification Report', fontsize=16, fontweight='bold')
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Classes', fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Classification report plot saved to {save_path}")
    
    plt.show() 