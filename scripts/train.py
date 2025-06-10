#!/usr/bin/env python3
"""
Main Training Script for Aquaculture Fish Classification

This script provides a command-line interface for training fish classification models
with comprehensive configuration management and logging.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import torch
import yaml
from omegaconf import OmegaConf
import warnings

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.dataset import FishDataModule
from models.fish_classifier import create_model
from training.trainer import FishTrainer
from utils.config import setup_config, setup_logging
from utils.device import setup_device
from utils.reproducibility import set_seed

warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Fish Classification Model")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the dataset directory"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments",
        help="Output directory for models and logs"
    )
    
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name for this experiment (overrides config)"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with reduced dataset size"
    )
    
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = setup_config(args.config)
    
    # Override config with command line arguments
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    
    if args.no_wandb:
        config.logging.use_wandb = False
    
    if args.debug:
        config.data.batch_size = 8
        config.training.epochs = 5
        config.logging.log_frequency = 1
        config.experiment_name = f"debug_{config.experiment_name}"
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup device
    device = setup_device(config.hardware.device)
    
    # Setup logging
    log_dir = Path(args.output_dir) / config.experiment_name / "logs"
    setup_logging(log_dir)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting experiment: {config.experiment_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Validate data path
    if not os.path.exists(args.data_path):
        raise ValueError(f"Data path does not exist: {args.data_path}")
    
    # Setup data module
    logger.info("Setting up data module...")
    data_module = FishDataModule(config)
    
    # Create data loaders
    train_loader, val_loader, test_loader = data_module.get_dataloaders(args.data_path)
    
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model: {config.model.architecture}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Setup trainer
    logger.info("Setting up trainer...")
    trainer = FishTrainer(model, config, device)
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and trainer.scheduler:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Create output directories
    model_dir = Path(args.output_dir) / config.experiment_name / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Start training
    logger.info("Starting training...")
    try:
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=str(model_dir)
        )
        
        # Save final model
        final_model_path = model_dir / "final_model.pth"
        trainer.save_checkpoint(str(final_model_path), trainer.epochs, trainer.best_val_acc)
        
        # Save training history
        history_path = model_dir / "training_history.yaml"
        with open(history_path, 'w') as f:
            yaml.dump(history, f)
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_metrics, test_preds, test_targets = trainer.validate_epoch(test_loader)
        
        logger.info("Test Results:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Save test results
        test_results = {
            'metrics': test_metrics,
            'config': OmegaConf.to_yaml(config)
        }
        
        results_path = model_dir / "test_results.yaml"
        with open(results_path, 'w') as f:
            yaml.dump(test_results, f)
        
        logger.info(f"Experiment completed successfully!")
        logger.info(f"Results saved to: {model_dir}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # Save current model state
        interrupted_path = model_dir / "interrupted_model.pth"
        trainer.save_checkpoint(str(interrupted_path), 0, 0.0)
        logger.info(f"Model saved to: {interrupted_path}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main() 