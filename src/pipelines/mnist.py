# ==============================================================================
# Title:       MNIST Data Preparation Pipeline
# File:        /pipelines/mnist.py
# Author:      KATO Kengo
# Description: Pipeline for importing, preparing and exporting MNIST
# ==============================================================================

from torchvision.datasets import MNIST

from src.pipelines.batch.batching import batch_dataset
from src.pipelines.load.loader import load_dataset, split_dataset
from src.pipelines.transform.transformer import transform_dataset


def run_mnist_pipeline():
    """Load and prepare MNIST, returning training, validation and test loaders."""
    train_data, test_data = load_dataset(MNIST)
    train_data, test_data = transform_dataset(train_data, test_data)
    train_subset, validation_subset, test_data = split_dataset(train_data, test_data)
    return batch_dataset(train_subset, validation_subset, test_data)
