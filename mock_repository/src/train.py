"""
Training script with optimization loop
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch_num):
    """
    Train model for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        optimizer: Optimizer instance
        criterion: Loss function
        device: Device to train on
        epoch_num: Current epoch number
    
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate_model(model, val_loader, criterion, device):
    """
    Validate model on validation set.
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
    
    Returns:
        Validation loss and accuracy
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    return val_loss / len(val_loader), correct / len(val_loader.dataset)


def save_checkpoint(model, optimizer, epoch, loss, accuracy, save_path):
    """
    Save model checkpoint to disk.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer instance
        epoch: Current epoch
        loss: Current loss
        accuracy: Current accuracy
        save_path: Path to save checkpoint
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }, save_path)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Load model checkpoint from disk.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model to load weights into
        optimizer: Optional optimizer to load state into
    
    Returns:
        Epoch number and metrics
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
