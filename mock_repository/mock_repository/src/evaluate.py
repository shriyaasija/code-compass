"""
Evaluation utilities
"""

import numpy as np
import matplotlib.pyplot as plt


def calculate_accuracy(predictions, targets):
    """
    Calculate classification accuracy.
    
    Args:
        predictions: Model predictions
        targets: Ground truth labels
    
    Returns:
        Accuracy score
    """
    correct = (predictions == targets).sum()
    total = len(targets)
    return correct / total


def compute_confusion_matrix(predictions, targets, num_classes):
    """
    Compute confusion matrix for predictions.
    
    Args:
        predictions: Model predictions
        targets: Ground truth labels
        num_classes: Number of classes
    
    Returns:
        Confusion matrix
    """
    confusion_matrix = np.zeros((num_classes, num_classes))
    
    for pred, target in zip(predictions, targets):
        confusion_matrix[target, pred] += 1
    
    return confusion_matrix


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    """
    Plot training and validation curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        save_path: Path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot accuracies
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.savefig(save_path)
