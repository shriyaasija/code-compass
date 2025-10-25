"""
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
