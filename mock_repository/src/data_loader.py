"""
Data loading and preprocessing utilities for image datasets
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os


class ImageDataset(Dataset):
    """Custom Dataset for loading images"""
    pass


def load_image_dataset(data_dir, batch_size=32, shuffle=True):
    """
    Load images from directory and create DataLoader.
    
    Args:
        data_dir: Path to image directory
        batch_size: Number of images per batch
        shuffle: Whether to shuffle data
    
    Returns:
        DataLoader object
    """
    dataset = ImageDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def preprocess_images(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Preprocess images with normalization and resizing.
    
    Args:
        images: Batch of images
        mean: Mean values for normalization
        std: Standard deviation for normalization
    
    Returns:
        Preprocessed images
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform(images)


def augment_images(images, augmentation_config):
    """
    Apply data augmentation to images.
    
    Args:
        images: Batch of images
        augmentation_config: Dictionary with augmentation parameters
    
    Returns:
        Augmented images
    """
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2)
    ])
    return transform(images)
