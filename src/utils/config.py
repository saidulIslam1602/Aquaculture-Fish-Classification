"""
Configuration utilities for fish classification project.
"""

import os
import logging
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from typing import Union


def setup_config(config_path: Union[str, Path]) -> DictConfig:
    """
    Load and setup configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration object
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = OmegaConf.load(config_path)
    
    # Validate required fields
    required_fields = [
        'project_name',
        'data',
        'model',
        'training',
        'species.classes'
    ]
    
    for field in required_fields:
        if not OmegaConf.select(config, field):
            raise ValueError(f"Required configuration field missing: {field}")
    
    # Update num_classes based on species list
    config.model.num_classes = len(config.species.classes)
    
    return config


def setup_logging(log_dir: Union[str, Path], log_level: str = "INFO") -> None:
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory to save log files
        log_level: Logging level
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging configuration
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup complete. Log files saved to: {log_dir}")


def save_config(config: DictConfig, save_path: Union[str, Path]) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration to save
        save_path: Path to save configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        OmegaConf.save(config, f)


def merge_configs(base_config: DictConfig, override_config: DictConfig) -> DictConfig:
    """
    Merge two configurations with override taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    return OmegaConf.merge(base_config, override_config) 