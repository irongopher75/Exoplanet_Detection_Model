"""
Random seed management for reproducibility.

This module provides utilities to set random seeds across all random number
generators used in the pipeline (Python random, NumPy, PyTorch).
"""

import random
import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    """
    Set random seeds for reproducibility across numpy, torch, and python.
    
    This function ensures that all random operations are deterministic
    when the same seed is used, enabling reproducible experiments.
    
    Parameters
    ----------
    seed : int
        Random seed value. Should be a positive integer.
    
    Notes
    -----
    - Sets Python's random module seed
    - Sets NumPy's random seed
    - Sets PyTorch's random seed (CPU and CUDA)
    - Enables deterministic CUDA operations (may impact performance)
    - Disables CUDA benchmarking for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Set CUDA seeds if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Enable deterministic behavior (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
