"""
Fish Classification Models

Implements various CNN architectures for fish species classification
with transfer learning capabilities.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import timm
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class FishClassifier(nn.Module):
    """
    Main fish classifier with support for multiple backbone architectures.
    """
    
    def __init__(
        self,
        architecture: str = "resnet50",
        num_classes: int = 10,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False
    ):
        """
        Initialize the fish classifier.
        
        Args:
            architecture: Backbone architecture name
            num_classes: Number of fish species classes
            pretrained: Whether to use pretrained weights
            dropout: Dropout probability
            freeze_backbone: Whether to freeze backbone parameters
        """
        super(FishClassifier, self).__init__()
        
        self.architecture = architecture
        self.num_classes = num_classes
        self.dropout = dropout
        
        # Create backbone
        self.backbone = self._create_backbone(architecture, pretrained)
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
        
        # Get feature dimension
        self.feature_dim = self._get_feature_dim()
        
        # Create classifier head
        self.classifier = self._create_classifier_head()
        
        logger.info(f"Created {architecture} classifier with {num_classes} classes")
    
    def _create_backbone(self, architecture: str, pretrained: bool) -> nn.Module:
        """Create the backbone network."""
        if architecture.startswith('resnet'):
            if architecture == 'resnet18':
                model = models.resnet18(pretrained=pretrained)
            elif architecture == 'resnet34':
                model = models.resnet34(pretrained=pretrained)
            elif architecture == 'resnet50':
                model = models.resnet50(pretrained=pretrained)
            elif architecture == 'resnet101':
                model = models.resnet101(pretrained=pretrained)
            else:
                raise ValueError(f"Unsupported ResNet architecture: {architecture}")
            
            # Remove the final classification layer
            return nn.Sequential(*list(model.children())[:-1])
        
        elif architecture.startswith('efficientnet'):
            # Use timm for EfficientNet
            model = timm.create_model(architecture, pretrained=pretrained)
            return nn.Sequential(*list(model.children())[:-1])
        
        elif architecture == 'vit_base_patch16_224':
            # Vision Transformer
            model = timm.create_model(architecture, pretrained=pretrained)
            return nn.Sequential(*list(model.children())[:-1])
        
        elif architecture.startswith('convnext'):
            # ConvNeXt models
            model = timm.create_model(architecture, pretrained=pretrained)
            return nn.Sequential(*list(model.children())[:-1])
        
        else:
            # Try to use timm for other architectures
            try:
                model = timm.create_model(architecture, pretrained=pretrained)
                return nn.Sequential(*list(model.children())[:-1])
            except Exception as e:
                raise ValueError(f"Unsupported architecture: {architecture}. Error: {str(e)}")
    
    def _get_feature_dim(self) -> int:
        """Get the feature dimension of the backbone."""
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            return features.view(features.size(0), -1).size(1)
    
    def _create_classifier_head(self) -> nn.Module:
        """Create the classification head."""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(self.dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(256, self.num_classes)
        )
    
    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone parameters frozen")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone parameters unfrozen")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        features = self.backbone(x)
        return self.classifier(features)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the backbone.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature tensor
        """
        with torch.no_grad():
            features = self.backbone(x)
            return features.view(features.size(0), -1)


class EnsembleFishClassifier(nn.Module):
    """
    Ensemble classifier combining multiple models for improved accuracy.
    """
    
    def __init__(
        self,
        models: list,
        weights: Optional[list] = None,
        voting: str = "soft"
    ):
        """
        Initialize ensemble classifier.
        
        Args:
            models: List of trained models
            weights: Optional weights for each model
            voting: Voting strategy ("soft" or "hard")
        """
        super(EnsembleFishClassifier, self).__init__()
        
        self.models = nn.ModuleList(models)
        self.weights = weights if weights else [1.0] * len(models)
        self.voting = voting
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        logger.info(f"Created ensemble with {len(models)} models using {voting} voting")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble.
        
        Args:
            x: Input tensor
            
        Returns:
            Ensemble predictions
        """
        outputs = []
        
        for model, weight in zip(self.models, self.weights):
            model.eval()
            with torch.no_grad():
                output = model(x)
                
                if self.voting == "soft":
                    output = torch.softmax(output, dim=1)
                
                outputs.append(output * weight)
        
        if self.voting == "soft":
            return torch.stack(outputs).sum(dim=0)
        else:
            # Hard voting
            predictions = [torch.argmax(output, dim=1) for output in outputs]
            # Simple majority vote (could be improved)
            ensemble_pred = torch.mode(torch.stack(predictions), dim=0)[0]
            # Convert back to one-hot for consistency
            num_classes = outputs[0].size(1)
            one_hot = torch.zeros_like(outputs[0])
            one_hot.scatter_(1, ensemble_pred.unsqueeze(1), 1)
            return one_hot


def create_model(config: Dict[str, Any]) -> FishClassifier:
    """
    Factory function to create a fish classifier model.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Configured model instance
    """
    model_config = config['model']
    
    model = FishClassifier(
        architecture=model_config['architecture'],
        num_classes=model_config['num_classes'],
        pretrained=model_config['pretrained'],
        dropout=model_config['dropout'],
        freeze_backbone=model_config.get('freeze_backbone', False)
    )
    
    return model


def load_pretrained_model(model_path: str, config: Dict[str, Any], device: str = 'cpu') -> FishClassifier:
    """
    Load a pretrained model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
        config: Model configuration
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    model = create_model(config)
    
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded pretrained model from {model_path}")
    
    return model 