
# Now create mock code files that correspond to the JSON tree

import os

# Create directory structure
os.makedirs('mock_repository/src', exist_ok=True)

# Create data_loader.py
data_loader_code = '''"""
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
'''

with open('mock_repository/src/data_loader.py', 'w') as f:
    f.write(data_loader_code)

# Create model.py
model_code = '''"""
Neural network architecture definitions using PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageClassifier(nn.Module):
    """
    Convolutional neural network for image classification
    """
    
    def __init__(self, num_classes, input_channels=3, dropout_rate=0.5):
        """
        Initialize the image classifier model.
        
        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels (3 for RGB)
            dropout_rate: Dropout probability
        """
        super(ImageClassifier, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
        
        Returns:
            Output logits
        """
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
    def get_feature_maps(self, x, layer_name):
        """
        Extract feature maps from a specific layer.
        
        Args:
            x: Input tensor
            layer_name: Name of layer to extract from
        
        Returns:
            Feature maps from specified layer
        """
        features = {}
        # Implementation for feature extraction
        return features


def initialize_weights(model, init_type='xavier'):
    """
    Initialize model weights with specified method.
    
    Args:
        model: PyTorch model
        init_type: Type of initialization ('xavier' or 'kaiming')
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if init_type == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            elif init_type == 'kaiming':
                nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
'''

with open('mock_repository/src/model.py', 'w') as f:
    f.write(model_code)

# Create train.py
train_code = '''"""
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
'''

with open('mock_repository/src/train.py', 'w') as f:
    f.write(train_code)

# Create evaluate.py
evaluate_code = '''"""
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
'''

with open('mock_repository/src/evaluate.py', 'w') as f:
    f.write(evaluate_code)

# Create utils.py
utils_code = '''"""
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
'''

with open('mock_repository/src/utils.py', 'w') as f:
    f.write(utils_code)

print("✅ Created mock repository with all code files:")
print("   mock_repository/")
print("   └── src/")
print("       ├── data_loader.py")
print("       ├── model.py")
print("       ├── train.py")
print("       ├── evaluate.py")
print("       └── utils.py")

import ast

def extract_functions(file_path):
    with open(file_path, 'r') as f:
        source = f.read()
    node = ast.parse(source)
    results = []
    for n in ast.walk(node):
        if isinstance(n, ast.FunctionDef):
            results.append({
                "title": n.name,
                "type": "function",
                "start_line": n.lineno,
                "end_line": n.end_lineno if hasattr(n, "end_lineno") else n.lineno,
                "summary": ast.get_docstring(n) or ""
            })
        if isinstance(n, ast.ClassDef):
            results.append({
                "title": n.name,
                "type": "class",
                "start_line": n.lineno,
                "end_line": n.end_lineno if hasattr(n, "end_lineno") else n.lineno,
                "summary": ast.get_docstring(n) or "",
                "nodes": []  # You can fill this via a recursive call for methods if desired
            })
    return results

def build_tree(repo_path):
    tree = {
        "title": "repo-root",
        "type": "repository",
        "path": repo_path,
        "nodes": []
    }
    for dirpath, dirnames, filenames in os.walk(repo_path):
        relpath = os.path.relpath(dirpath, repo_path)
        node = {
            "title": relpath,
            "type": "folder" if relpath != "." else "repository",
            "path": dirpath,
            "nodes": []
        }
        for fname in filenames:
            if fname.endswith(".py"):
                fpath = os.path.join(dirpath, fname)
                child = {
                    "title": fname,
                    "type": "file_py",
                    "path": fpath,
                    "nodes": extract_functions(fpath)
                }
                node["nodes"].append(child)
        # We'll add folders recursively if needed
        if relpath == ".":
            tree["nodes"].extend(node["nodes"])
        else:
            # For simplicity only one-level nesting (expand as needed)
            tree["nodes"].append(node)
    return tree

tree_json = build_tree("./mockrepository")
with open("mock_pageindex_tree.json", "w") as f:
    import json
    json.dump(tree_json, f, indent=2)
print("Wrote tree JSON")
