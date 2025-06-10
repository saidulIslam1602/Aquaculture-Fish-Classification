"""
Device utilities for fish classification project.
"""

import torch
import logging
from typing import Union

logger = logging.getLogger(__name__)


def setup_device(device: Union[str, torch.device] = "auto") -> torch.device:
    """
    Setup the computation device (CPU, CUDA, MPS).
    
    Args:
        device: Device specification ("auto", "cpu", "cuda", "mps", or device index)
        
    Returns:
        PyTorch device object
    """
    if isinstance(device, torch.device):
        return device
    
    if device == "auto":
        if torch.cuda.is_available():
            device_obj = torch.device("cuda")
            logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_obj = torch.device("mps")
            logger.info("MPS (Apple Silicon) available. Using MPS.")
        else:
            device_obj = torch.device("cpu")
            logger.info("Using CPU.")
    
    elif device == "cpu":
        device_obj = torch.device("cpu")
        logger.info("Using CPU (forced).")
    
    elif device.startswith("cuda"):
        if torch.cuda.is_available():
            device_obj = torch.device(device)
            gpu_id = device_obj.index if device_obj.index is not None else 0
            logger.info(f"Using CUDA device {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        else:
            logger.warning("CUDA not available, falling back to CPU.")
            device_obj = torch.device("cpu")
    
    elif device == "mps":
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_obj = torch.device("mps")
            logger.info("Using MPS (Apple Silicon).")
        else:
            logger.warning("MPS not available, falling back to CPU.")
            device_obj = torch.device("cpu")
    
    else:
        raise ValueError(f"Unsupported device specification: {device}")
    
    return device_obj


def get_device_info() -> dict:
    """
    Get detailed information about available devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        "cpu_count": torch.get_num_threads()
    }
    
    if torch.cuda.is_available():
        info.update({
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_count": torch.cuda.device_count(),
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "gpu_memory": [
                torch.cuda.get_device_properties(i).total_memory / 1e9 
                for i in range(torch.cuda.device_count())
            ]
        })
    
    return info


def print_device_info():
    """Print detailed device information."""
    info = get_device_info()
    
    print("=" * 50)
    print("DEVICE INFORMATION")
    print("=" * 50)
    print(f"PyTorch Version: {info['pytorch_version']}")
    print(f"CPU Threads: {info['cpu_count']}")
    
    if info['cuda_available']:
        print(f"CUDA Available: Yes")
        print(f"CUDA Version: {info['cuda_version']}")
        print(f"cuDNN Version: {info['cudnn_version']}")
        print(f"GPU Count: {info['gpu_count']}")
        for i, (name, memory) in enumerate(zip(info['gpu_names'], info['gpu_memory'])):
            print(f"  GPU {i}: {name} ({memory:.1f} GB)")
    else:
        print(f"CUDA Available: No")
    
    if info['mps_available']:
        print(f"MPS Available: Yes")
    else:
        print(f"MPS Available: No")
    
    print("=" * 50)


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU memory cache cleared.")


def get_memory_usage(device: torch.device) -> dict:
    """
    Get memory usage information for the specified device.
    
    Args:
        device: PyTorch device
        
    Returns:
        Dictionary with memory usage information
    """
    if device.type == "cuda":
        memory_allocated = torch.cuda.memory_allocated(device) / 1e9
        memory_cached = torch.cuda.memory_reserved(device) / 1e9
        memory_total = torch.cuda.get_device_properties(device).total_memory / 1e9
        
        return {
            "allocated_gb": memory_allocated,
            "cached_gb": memory_cached,
            "total_gb": memory_total,
            "free_gb": memory_total - memory_cached,
            "utilization": memory_cached / memory_total * 100
        }
    else:
        return {"message": "Memory monitoring only available for CUDA devices"} 