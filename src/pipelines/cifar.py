# ==============================================================================
# Title:       CIFAR-10 Data Preparation Pipeline
# File:        /pipelines/cifar.py
# Author:      KATO Kengo
# Description: Pipeline for importing, preparing and exporting CIFAR-10
# ==============================================================================

from torchvision.datasets import CIFAR10

from src.pipelines.batch.batching import batch_dataset
from src.pipelines.load.loader import load_dataset, split_dataset
from src.pipelines.transform.transformer import transform_dataset


def run_cifar_pipeline():
    """Load and prepare CIFAR-10, returning training, validation and test loaders."""
    train_data, test_data = load_dataset(CIFAR10)
    train_data, test_data = transform_dataset(train_data, test_data)
    train_subset, validation_subset, test_data = split_dataset(train_data, test_data)
    return batch_dataset(train_subset, validation_subset, test_data)
