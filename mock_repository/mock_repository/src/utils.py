"""
Utility functions
"""

import logging
import random
import numpy as np
import torch


def setup_logger(log_file, log_level='INFO'):
    """
    Setup logger for training.
    
    Args:
        log_file: Path to log file
        log_level: Logging level
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level))
    
    handler = logging.FileHandler(log_file)
    logger.addHandler(handler)
    
    return logger


def set_random_seed(seed=42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(use_cuda=True):
    """
    Get PyTorch device (CUDA or CPU).
    
    Args:
        use_cuda: Whether to use CUDA if available
    
    Returns:
        torch.device object
    """
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    return device
