
"""
Generate mock_pageindex_tree.json with REAL embeddings
Run this to create proper test data
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer

print("🔄 Generating mock PageIndex tree with real embeddings...")
print("="*70)

# Initialize model
print("\n1. Loading SentenceTransformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model loaded")

def generate_mock_tree():
    """Generate realistic mock data with REAL embeddings"""

    print("\n2. Generating embeddings for all nodes...")

    # Helper to create embedding from text
    def embed(text):
        return model.encode(text).tolist()

    mock_tree = {
        "node_id": "repo_root",
        "node_type": "repository",
        "repository_name": "ml_image_classifier",
        "repository_summary": "A PyTorch-based image classification project with data preprocessing, model training, and evaluation components",
        "embedding": embed("A PyTorch-based image classification project with data preprocessing, model training, and evaluation components"),
        "children": [
            {
                "node_id": "file_1",
                "node_type": "file",
                "file_name": "data_loader.py",
                "file_path": "src/data_loader.py",
                "summary": "Data loading and preprocessing utilities for image datasets",
                "embedding": embed("Data loading and preprocessing utilities for image datasets"),
                "children": [
                    {
                        "node_id": "func_1_1",
                        "node_type": "function",
                        "name": "load_image_dataset",
                        "signature": "def load_image_dataset(data_dir, batch_size=32, shuffle=True)",
                        "start_line": 15,
                        "end_line": 35,
                        "summary": "Loads images from directory and creates PyTorch DataLoader with specified batch size and shuffling",
                        "embedding": embed("Loads images from directory and creates PyTorch DataLoader with specified batch size and shuffling"),
                        "docstring": "Load images from directory and create DataLoader.\n\nArgs:\n    data_dir: Path to image directory\n    batch_size: Number of images per batch\n    shuffle: Whether to shuffle data\n\nReturns:\n    DataLoader object"
                    },
                    {
                        "node_id": "func_1_2",
                        "node_type": "function",
                        "name": "preprocess_images",
                        "signature": "def preprocess_images(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])",
                        "start_line": 37,
                        "end_line": 50,
                        "summary": "Applies normalization and resizing transformations to images using ImageNet statistics",
                        "embedding": embed("Applies normalization and resizing transformations to images using ImageNet statistics"),
                        "docstring": "Preprocess images with normalization and resizing."
                    },
                    {
                        "node_id": "func_1_3",
                        "node_type": "function",
                        "name": "augment_images",
                        "signature": "def augment_images(images, augmentation_config)",
                        "start_line": 52,
                        "end_line": 70,
                        "summary": "Applies data augmentation techniques including random rotation, flipping, cropping, and color jittering",
                        "embedding": embed("Applies data augmentation techniques including random rotation, flipping, cropping, and color jittering"),
                        "docstring": "Apply data augmentation to images."
                    }
                ]
            },
            {
                "node_id": "file_2",
                "node_type": "file",
                "file_name": "model.py",
                "file_path": "src/model.py",
                "summary": "Neural network architecture definitions using PyTorch including CNN layers and forward pass logic",
                "embedding": embed("Neural network architecture definitions using PyTorch including CNN layers and forward pass logic"),
                "children": [
                    {
                        "node_id": "class_2_1",
                        "node_type": "class",
                        "name": "ImageClassifier",
                        "signature": "class ImageClassifier(nn.Module)",
                        "start_line": 20,
                        "end_line": 95,
                        "summary": "Convolutional neural network for image classification with residual connections and batch normalization",
                        "embedding": embed("Convolutional neural network for image classification with residual connections and batch normalization"),
                        "children": [
                            {
                                "node_id": "method_2_1_1",
                                "node_type": "method",
                                "name": "__init__",
                                "signature": "def __init__(self, num_classes, input_channels=3, dropout_rate=0.5)",
                                "start_line": 25,
                                "end_line": 50,
                                "summary": "Initializes model architecture with convolutional layers, pooling, batch normalization, dropout, and fully connected layers",
                                "embedding": embed("Initializes model architecture with convolutional layers, pooling, batch normalization, dropout, and fully connected layers"),
                                "docstring": "Initialize the image classifier model."
                            },
                            {
                                "node_id": "method_2_1_2",
                                "node_type": "method",
                                "name": "forward",
                                "signature": "def forward(self, x)",
                                "start_line": 52,
                                "end_line": 70,
                                "summary": "Defines forward propagation through convolutional layers, pooling, and fully connected layers",
                                "embedding": embed("Defines forward propagation through convolutional layers, pooling, and fully connected layers"),
                                "docstring": "Forward pass through the network."
                            },
                            {
                                "node_id": "method_2_1_3",
                                "node_type": "method",
                                "name": "get_feature_maps",
                                "signature": "def get_feature_maps(self, x, layer_name)",
                                "start_line": 72,
                                "end_line": 85,
                                "summary": "Extracts intermediate feature maps from specified layer for visualization and analysis",
                                "embedding": embed("Extracts intermediate feature maps from specified layer for visualization and analysis"),
                                "docstring": "Extract feature maps from a specific layer."
                            }
                        ]
                    },
                    {
                        "node_id": "func_2_2",
                        "node_type": "function",
                        "name": "initialize_weights",
                        "signature": "def initialize_weights(model, init_type='xavier')",
                        "start_line": 97,
                        "end_line": 115,
                        "summary": "Initializes model weights using Xavier or Kaiming initialization methods for better convergence",
                        "embedding": embed("Initializes model weights using Xavier or Kaiming initialization methods for better convergence"),
                        "docstring": "Initialize model weights with specified method."
                    }
                ]
            },
            {
                "node_id": "file_3",
                "node_type": "file",
                "file_name": "train.py",
                "file_path": "src/train.py",
                "summary": "Training script with optimization loop, loss computation, gradient descent, and checkpoint saving",
                "embedding": embed("Training script with optimization loop, loss computation, gradient descent, and checkpoint saving"),
                "children": [
                    {
                        "node_id": "func_3_1",
                        "node_type": "function",
                        "name": "train_one_epoch",
                        "signature": "def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch_num)",
                        "start_line": 18,
                        "end_line": 50,
                        "summary": "Trains model for one epoch by iterating through batches, computing loss, backpropagation, and updating weights",
                        "embedding": embed("Trains model for one epoch by iterating through batches, computing loss, backpropagation, and updating weights"),
                        "docstring": "Train model for one epoch."
                    },
                    {
                        "node_id": "func_3_2",
                        "node_type": "function",
                        "name": "validate_model",
                        "signature": "def validate_model(model, val_loader, criterion, device)",
                        "start_line": 52,
                        "end_line": 75,
                        "summary": "Evaluates model on validation dataset without gradient computation and returns loss and accuracy metrics",
                        "embedding": embed("Evaluates model on validation dataset without gradient computation and returns loss and accuracy metrics"),
                        "docstring": "Validate model on validation set."
                    },
                    {
                        "node_id": "func_3_3",
                        "node_type": "function",
                        "name": "save_checkpoint",
                        "signature": "def save_checkpoint(model, optimizer, epoch, loss, accuracy, save_path)",
                        "start_line": 77,
                        "end_line": 92,
                        "summary": "Saves model checkpoint including state dict, optimizer state, epoch number, loss, and accuracy metrics",
                        "embedding": embed("Saves model checkpoint including state dict, optimizer state, epoch number, loss, and accuracy metrics"),
                        "docstring": "Save model checkpoint to disk."
                    },
                    {
                        "node_id": "func_3_4",
                        "node_type": "function",
                        "name": "load_checkpoint",
                        "signature": "def load_checkpoint(checkpoint_path, model, optimizer=None)",
                        "start_line": 94,
                        "end_line": 110,
                        "summary": "Loads model checkpoint from disk and restores model weights and optimizer state for continued training",
                        "embedding": embed("Loads model checkpoint from disk and restores model weights and optimizer state for continued training"),
                        "docstring": "Load model checkpoint from disk."
                    }
                ]
            },
            {
                "node_id": "file_4",
                "node_type": "file",
                "file_name": "evaluate.py",
                "file_path": "src/evaluate.py",
                "summary": "Evaluation utilities for computing accuracy, precision, recall, F1 score, and confusion matrix",
                "embedding": embed("Evaluation utilities for computing accuracy, precision, recall, F1 score, and confusion matrix"),
                "children": [
                    {
                        "node_id": "func_4_1",
                        "node_type": "function",
                        "name": "calculate_accuracy",
                        "signature": "def calculate_accuracy(predictions, targets)",
                        "start_line": 10,
                        "end_line": 20,
                        "summary": "Computes classification accuracy by comparing predicted labels with ground truth targets",
                        "embedding": embed("Computes classification accuracy by comparing predicted labels with ground truth targets"),
                        "docstring": "Calculate classification accuracy."
                    },
                    {
                        "node_id": "func_4_2",
                        "node_type": "function",
                        "name": "compute_confusion_matrix",
                        "signature": "def compute_confusion_matrix(predictions, targets, num_classes)",
                        "start_line": 22,
                        "end_line": 40,
                        "summary": "Generates confusion matrix showing true vs predicted labels for multi-class classification",
                        "embedding": embed("Generates confusion matrix showing true vs predicted labels for multi-class classification"),
                        "docstring": "Compute confusion matrix for predictions."
                    },
                    {
                        "node_id": "func_4_3",
                        "node_type": "function",
                        "name": "plot_training_curves",
                        "signature": "def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path)",
                        "start_line": 42,
                        "end_line": 65,
                        "summary": "Creates matplotlib visualizations of training and validation loss and accuracy curves over epochs",
                        "embedding": embed("Creates matplotlib visualizations of training and validation loss and accuracy curves over epochs"),
                        "docstring": "Plot training and validation curves."
                    }
                ]
            },
            {
                "node_id": "file_5",
                "node_type": "file",
                "file_name": "utils.py",
                "file_path": "src/utils.py",
                "summary": "Utility functions for logging, configuration management, device selection, and seed setting",
                "embedding": embed("Utility functions for logging, configuration management, device selection, and seed setting"),
                "children": [
                    {
                        "node_id": "func_5_1",
                        "node_type": "function",
                        "name": "setup_logger",
                        "signature": "def setup_logger(log_file, log_level='INFO')",
                        "start_line": 8,
                        "end_line": 22,
                        "summary": "Configures Python logging with file and console handlers for training progress monitoring",
                        "embedding": embed("Configures Python logging with file and console handlers for training progress monitoring"),
                        "docstring": "Setup logger for training."
                    },
                    {
                        "node_id": "func_5_2",
                        "node_type": "function",
                        "name": "set_random_seed",
                        "signature": "def set_random_seed(seed=42)",
                        "start_line": 24,
                        "end_line": 35,
                        "summary": "Sets random seeds for Python, NumPy, and PyTorch to ensure reproducible results",
                        "embedding": embed("Sets random seeds for Python, NumPy, and PyTorch to ensure reproducible results"),
                        "docstring": "Set random seed for reproducibility."
                    },
                    {
                        "node_id": "func_5_3",
                        "node_type": "function",
                        "name": "get_device",
                        "signature": "def get_device(use_cuda=True)",
                        "start_line": 37,
                        "end_line": 48,
                        "summary": "Detects and returns CUDA GPU device if available, otherwise returns CPU device for computation",
                        "embedding": embed("Detects and returns CUDA GPU device if available, otherwise returns CPU device for computation"),
                        "docstring": "Get PyTorch device (CUDA or CPU)."
                    }
                ]
            }
        ]
    }

    return mock_tree

# Generate the tree
tree = generate_mock_tree()

# Save to file
print("\n3. Saving to mock_pageindex_tree.json...")
with open('mock_pageindex_tree.json', 'w') as f:
    json.dump(tree, f, indent=2)

print("✅ Saved!")

# Verify
print("\n4. Verifying...")
with open('mock_pageindex_tree.json', 'r') as f:
    loaded = json.load(f)

def count_nodes(node):
    count = 1
    if 'children' in node:
        for child in node['children']:
            count += count_nodes(child)
    return count

total = count_nodes(loaded)
print(f"Total nodes: {total}")
print(f"Root has embedding: {'embedding' in loaded}")
print(f"Embedding dimension: {len(loaded['embedding'])}")

print("\n" + "="*70)
print("✅ DONE! mock_pageindex_tree.json now has REAL embeddings!")
print("="*70)
print("\nNow restart your API and try searching again.")
print("The query 'How do I train the model?' should now match:")
print("  - train_one_epoch")
print("  - validate_model")
print("  - save_checkpoint")
