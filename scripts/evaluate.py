#!/usr/bin/env python3
"""
Model Evaluation Script for Aquaculture Fish Classification

This script evaluates trained models on test datasets and generates
comprehensive performance reports with visualizations.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.dataset import FishDataModule
from models.fish_classifier import load_pretrained_model
from utils.config import setup_config, setup_logging
from utils.device import setup_device
from utils.visualization import (
    plot_confusion_matrix, 
    plot_class_distribution,
    visualize_predictions,
    create_classification_report_plot
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Fish Classification Model")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint"
    )
    
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the test dataset directory"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Output directory for evaluation results"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save individual predictions to file"
    )
    
    parser.add_argument(
        "--visualize_samples",
        type=int,
        default=16,
        help="Number of sample predictions to visualize"
    )
    
    return parser.parse_args()


def evaluate_model(model, data_loader, device, class_names):
    """
    Evaluate model on dataset.
    
    Args:
        model: Trained PyTorch model
        data_loader: Test data loader
        device: Device to run evaluation on
        class_names: List of class names
        
    Returns:
        Dictionary containing evaluation results
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    all_images = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            images, targets = images.to(device), targets.to(device)
            
            # Get model outputs
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # Store some images for visualization
            if batch_idx == 0:
                all_images = images.cpu()
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = np.mean(all_predictions == all_targets)
    
    # Per-class metrics
    report = classification_report(
        all_targets, all_predictions, 
        target_names=class_names, 
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    results = {
        'accuracy': accuracy,
        'predictions': all_predictions,
        'targets': all_targets,
        'probabilities': all_probabilities,
        'classification_report': report,
        'confusion_matrix': cm,
        'sample_images': all_images,
        'sample_targets': all_targets[:len(all_images)],
        'sample_predictions': all_predictions[:len(all_images)]
    }
    
    return results


def generate_evaluation_report(results, class_names, output_dir):
    """
    Generate comprehensive evaluation report.
    
    Args:
        results: Evaluation results dictionary
        class_names: List of class names
        output_dir: Output directory path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Overall Accuracy: {results['accuracy']:.4f}")
    logger.info(f"Number of samples: {len(results['targets'])}")
    logger.info(f"Number of classes: {len(class_names)}")
    
    # Per-class performance
    logger.info("\nPer-Class Performance:")
    logger.info("-" * 40)
    for class_name in class_names:
        if class_name in results['classification_report']:
            metrics = results['classification_report'][class_name]
            logger.info(f"{class_name:>15}: Precision={metrics['precision']:.3f}, "
                       f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    
    # 1. Confusion Matrix
    plot_confusion_matrix(
        results['targets'],
        results['predictions'],
        class_names,
        save_path=output_dir / "confusion_matrix.png",
        normalize=True
    )
    
    # 2. Classification Report Heatmap
    create_classification_report_plot(
        results['targets'],
        results['predictions'],
        class_names,
        save_path=output_dir / "classification_report.png"
    )
    
    # 3. Class Distribution
    plot_class_distribution(
        results['targets'],
        class_names,
        save_path=output_dir / "class_distribution.png"
    )
    
    # 4. Sample Predictions
    if len(results['sample_images']) > 0:
        visualize_predictions(
            results['sample_images'],
            torch.tensor(results['sample_targets']),
            torch.tensor(results['sample_predictions']),
            class_names,
            num_samples=min(16, len(results['sample_images'])),
            save_path=output_dir / "sample_predictions.png"
        )
    
    # Save detailed results
    import json
    
    # Prepare JSON-serializable report
    json_report = {
        'accuracy': float(results['accuracy']),
        'num_samples': int(len(results['targets'])),
        'num_classes': int(len(class_names)),
        'class_names': class_names,
        'classification_report': results['classification_report'],
        'confusion_matrix': results['confusion_matrix'].tolist()
    }
    
    with open(output_dir / "evaluation_report.json", 'w') as f:
        json.dump(json_report, f, indent=2)
    
    # Save predictions if requested
    if args.save_predictions:
        predictions_data = {
            'targets': results['targets'].tolist(),
            'predictions': results['predictions'].tolist(),
            'probabilities': results['probabilities'].tolist(),
            'class_names': class_names
        }
        
        with open(output_dir / "predictions.json", 'w') as f:
            json.dump(predictions_data, f, indent=2)
    
    logger.info(f"Evaluation report saved to: {output_dir}")


def main():
    """Main evaluation function."""
    global args
    args = parse_args()
    
    # Setup logging
    log_dir = Path(args.output_dir) / "logs"
    setup_logging(log_dir)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting model evaluation...")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Data: {args.data_path}")
    logger.info(f"Output: {args.output_dir}")
    
    # Load configuration
    config = setup_config(args.config_path)
    
    # Override batch size if specified
    if args.batch_size:
        config.data.batch_size = args.batch_size
    
    # Setup device
    device = setup_device(config.hardware.device)
    
    # Load model
    logger.info("Loading model...")
    model = load_pretrained_model(args.model_path, config, str(device))
    
    # Setup data
    logger.info("Setting up data...")
    data_module = FishDataModule(config)
    
    # Create test data loader
    _, _, test_loader = data_module.get_dataloaders(args.data_path)
    
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Run evaluation
    logger.info("Running evaluation...")
    results = evaluate_model(
        model, test_loader, device, config.species.classes
    )
    
    # Generate report
    generate_evaluation_report(
        results, config.species.classes, args.output_dir
    )
    
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main() 