"""Model architectures for fish classification."""

from .fish_classifier import FishClassifier, EnsembleFishClassifier, create_model, load_pretrained_model

__all__ = ['FishClassifier', 'EnsembleFishClassifier', 'create_model', 'load_pretrained_model'] 