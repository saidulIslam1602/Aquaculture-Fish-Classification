"""
Training Module for Fish Classification

Comprehensive trainer with support for various optimizers, schedulers,
mixed precision training, and experiment tracking.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import wandb
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import os
from typing import Dict, Any, Optional, Tuple, List
import logging
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class FishTrainer:
    """
    Comprehensive trainer for fish classification models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: str = 'cuda'
    ):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model to train
            config: Training configuration
            device: Device to train on
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Training configuration
        self.epochs = config['training']['epochs']
        self.learning_rate = config['training']['learning_rate']
        self.weight_decay = config['training']['weight_decay']
        self.gradient_clipping = config['training'].get('gradient_clipping', None)
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup loss function
        self.criterion = self._setup_criterion()
        
        # Mixed precision training
        self.use_amp = config['hardware'].get('mixed_precision', False)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Model compilation (PyTorch 2.0)
        if config['hardware'].get('compile_model', False):
            self.model = torch.compile(self.model)
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config['training']['early_stopping']['patience'],
            min_delta=config['training']['early_stopping']['min_delta']
        )
        
        # Logging setup
        self.setup_logging()
        
        # Metrics tracking
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_model_state = None
        
        logger.info(f"Trainer initialized on {device}")
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer based on configuration."""
        optimizer_name = self.config['training']['optimizer'].lower()
        
        if optimizer_name == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_name == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_name == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        scheduler_name = self.config['training'].get('scheduler', None)
        
        if scheduler_name is None:
            return None
        elif scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs
            )
        elif scheduler_name == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif scheduler_name == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.1,
                patience=10
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    
    def _setup_criterion(self) -> nn.Module:
        """Setup loss function."""
        return nn.CrossEntropyLoss()
    
    def setup_logging(self):
        """Setup experiment tracking."""
        self.use_wandb = self.config['logging'].get('use_wandb', False)
        self.use_tensorboard = self.config['logging'].get('use_tensorboard', False)
        
        if self.use_wandb:
            wandb.init(
                project=self.config['project_name'],
                name=self.config['experiment_name'],
                config=self.config
            )
            wandb.watch(self.model)
        
        if self.use_tensorboard:
            log_dir = os.path.join(self.config['paths']['logs'], 'tensorboard')
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()
        
        train_loss = 0.0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(data)
                    loss = self.criterion(outputs, targets)
                
                self.scaler.scale(loss).backward()
                
                if self.gradient_clipping:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                loss.backward()
                
                if self.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
                
                self.optimizer.step()
            
            train_loss += loss.item()
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Log batch metrics
            if batch_idx % self.config['logging']['log_frequency'] == 0:
                self.log_batch_metrics(batch_idx, loss.item(), len(train_loader))
        
        # Calculate epoch metrics
        epoch_loss = train_loss / len(train_loader)
        epoch_acc = accuracy_score(all_targets, all_preds)
        epoch_f1 = f1_score(all_targets, all_preds, average='weighted')
        
        metrics = {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'f1_score': epoch_f1
        }
        
        self.train_metrics.update(metrics)
        
        return metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        self.val_metrics.reset()
        
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in tqdm(val_loader, desc="Validation"):
                data, targets = data.to(self.device), targets.to(self.device)
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(data)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(data)
                    loss = self.criterion(outputs, targets)
                
                val_loss += loss.item()
                
                # Get predictions
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate epoch metrics
        epoch_loss = val_loss / len(val_loader)
        epoch_acc = accuracy_score(all_targets, all_preds)
        epoch_f1 = f1_score(all_targets, all_preds, average='weighted')
        epoch_precision = precision_score(all_targets, all_preds, average='weighted')
        epoch_recall = recall_score(all_targets, all_preds, average='weighted')
        
        metrics = {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'f1_score': epoch_f1,
            'precision': epoch_precision,
            'recall': epoch_recall
        }
        
        self.val_metrics.update(metrics)
        
        return metrics, all_preds, all_targets
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_dir: str
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            save_dir: Directory to save models
            
        Returns:
            Training history
        """
        os.makedirs(save_dir, exist_ok=True)
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        logger.info("Starting training...")
        
        for epoch in range(self.epochs):
            start_time = time.time()
            
            # Training phase
            train_metrics = self.train_epoch(train_loader)
            
            # Validation phase
            val_metrics, val_preds, val_targets = self.validate_epoch(val_loader)
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['accuracy'])
                else:
                    self.scheduler.step()
            
            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            
            # Save best model
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.best_model_state = self.model.state_dict().copy()
                self.save_checkpoint(
                    os.path.join(save_dir, 'best_model.pth'),
                    epoch,
                    val_metrics['accuracy']
                )
            
            # Log epoch metrics
            epoch_time = time.time() - start_time
            self.log_epoch_metrics(epoch, train_metrics, val_metrics, epoch_time)
            
            # Early stopping check
            if self.early_stopping(val_metrics['accuracy']):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
            
            # Save periodic checkpoint
            if epoch % self.config['logging']['save_model_frequency'] == 0:
                self.save_checkpoint(
                    os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'),
                    epoch,
                    val_metrics['accuracy']
                )
        
        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        logger.info(f"Training completed. Best validation accuracy: {self.best_val_acc:.4f}")
        
        return history
    
    def save_checkpoint(self, filepath: str, epoch: int, val_acc: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath}")
    
    def log_batch_metrics(self, batch_idx: int, loss: float, total_batches: int):
        """Log batch-level metrics."""
        if self.use_wandb:
            wandb.log({
                'batch_loss': loss,
                'batch': batch_idx
            })
    
    def log_epoch_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch_time: float
    ):
        """Log epoch-level metrics."""
        logger.info(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )
        
        if self.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'epoch_time': epoch_time
            })
        
        if self.use_tensorboard:
            self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/Val', val_metrics['loss'], epoch)
            self.writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Accuracy/Val', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 15, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
    
    def __call__(self, val_score: float) -> bool:
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = val_score
            self.counter = 0
        return False


class MetricsTracker:
    """Utility for tracking metrics."""
    
    def __init__(self):
        self.metrics = {}
    
    def update(self, metrics: Dict[str, float]):
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def reset(self):
        self.metrics = {}
    
    def get_avg(self, key: str) -> float:
        if key in self.metrics and self.metrics[key]:
            return np.mean(self.metrics[key])
        return 0.0 