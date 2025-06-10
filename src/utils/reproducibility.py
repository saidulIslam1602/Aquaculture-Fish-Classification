"""
Reproducibility utilities for fish classification project.
"""

import random
import numpy as np
import torch
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    Set random seeds for reproducible results.
    
    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms (may impact performance)
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Environment variables for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if deterministic:
        # Use deterministic algorithms
        torch.use_deterministic_algorithms(True)
        
        # Set cuDNN to deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set additional environment variables
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        logger.info(f"Seed set to {seed} with deterministic algorithms enabled")
        logger.warning("Deterministic mode may significantly impact performance")
    else:
        # Allow cuDNN to optimize for performance
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        
        logger.info(f"Seed set to {seed} with non-deterministic algorithms for better performance")


def get_random_states() -> dict:
    """
    Get current random states for all random number generators.
    
    Returns:
        Dictionary containing random states
    """
    states = {
        'python_random': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    }
    
    return states


def set_random_states(states: dict) -> None:
    """
    Set random states for all random number generators.
    
    Args:
        states: Dictionary containing random states
    """
    random.setstate(states['python_random'])
    np.random.set_state(states['numpy'])
    torch.set_rng_state(states['torch'])
    
    if torch.cuda.is_available() and states['torch_cuda'] is not None:
        torch.cuda.set_rng_state(states['torch_cuda'])


def create_reproducible_dataloader_worker_init_fn(seed: int) -> callable:
    """
    Create a worker initialization function for DataLoader to ensure reproducibility.
    
    Args:
        seed: Base seed value
        
    Returns:
        Worker initialization function
    """
    def worker_init_fn(worker_id: int) -> None:
        """Initialize random seeds for DataLoader workers."""
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    
    return worker_init_fn


def check_reproducibility(model: torch.nn.Module, 
                         data: torch.Tensor, 
                         num_runs: int = 5) -> bool:
    """
    Check if model outputs are reproducible across multiple runs.
    
    Args:
        model: PyTorch model to test
        data: Input data tensor
        num_runs: Number of test runs
        
    Returns:
        True if outputs are identical across runs
    """
    model.eval()
    outputs = []
    
    for i in range(num_runs):
        with torch.no_grad():
            output = model(data)
            outputs.append(output.clone())
    
    # Check if all outputs are identical
    reference = outputs[0]
    for i, output in enumerate(outputs[1:], 1):
        if not torch.allclose(reference, output, atol=1e-6):
            logger.warning(f"Output mismatch detected at run {i+1}")
            return False
    
    logger.info(f"Model outputs are reproducible across {num_runs} runs")
    return True


def save_random_states(filepath: str) -> None:
    """
    Save current random states to file.
    
    Args:
        filepath: Path to save random states
    """
    states = get_random_states()
    torch.save(states, filepath)
    logger.info(f"Random states saved to {filepath}")


def load_random_states(filepath: str) -> None:
    """
    Load random states from file.
    
    Args:
        filepath: Path to load random states from
    """
    states = torch.load(filepath)
    set_random_states(states)
    logger.info(f"Random states loaded from {filepath}")


class ReproducibilityContext:
    """Context manager for temporarily setting reproducibility settings."""
    
    def __init__(self, seed: int = 42, deterministic: bool = True):
        """
        Initialize reproducibility context.
        
        Args:
            seed: Random seed
            deterministic: Whether to use deterministic algorithms
        """
        self.seed = seed
        self.deterministic = deterministic
        self.original_states = None
        self.original_deterministic = None
        self.original_benchmark = None
    
    def __enter__(self):
        # Save original states
        self.original_states = get_random_states()
        self.original_deterministic = torch.backends.cudnn.deterministic
        self.original_benchmark = torch.backends.cudnn.benchmark
        
        # Set reproducibility
        set_seed(self.seed, self.deterministic)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original states
        set_random_states(self.original_states)
        torch.backends.cudnn.deterministic = self.original_deterministic
        torch.backends.cudnn.benchmark = self.original_benchmark 