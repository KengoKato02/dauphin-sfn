# ==============================================================================
# Title:       Dataset Loading
# File:        /pipelines/load/loader.py
# Author:      KATO Kengo
# Description: Functions for importing datasets and creating validation splits
# ==============================================================================

import torch
from torch.utils.data import random_split

from src.utils.decorators import handle_exceptions
from src.utils.logger import divider_logger, generic_logger


@handle_exceptions
def load_dataset(dataset_class):
    """Load the official training and test portions and inspect one example.

    Args:
        dataset_class: The torchvision MNIST or CIFAR10 dataset class.

    Returns:
        train_data, test_data: Datasets containing original images and labels.
    """
    train_data = dataset_class(
        root="data",
        train=True,
        download=True,
    )

    test_data = dataset_class(
        root="data",
        train=False,
        download=True,
    )

    image, label = train_data[0]

    divider_logger()
    print("TRAIN & TEST")
    generic_logger({
        "training_examples": len(train_data),
        "test_examples": len(test_data),
        "image_type": type(image),
        "image_size": image.size,
        "label": label,
        "label_type": type(label),
    })

    return train_data, test_data


@handle_exceptions
def split_dataset(train_data, test_data):
    """Hold out 10% of training examples for validation, keeping test data separate.

    Returns:
        train_subset, validation_subset, test_data: Non-overlapping training
        and validation subsets, followed by the unchanged official test set.
    """
    validation_size = len(train_data) // 10
    training_size = len(train_data) - validation_size
    split_generator = torch.Generator().manual_seed(42)

    train_subset, validation_subset = random_split(
        train_data,
        lengths=[training_size, validation_size],
        generator=split_generator,
    )

    divider_logger()
    print("SPLIT")
    generic_logger({
        "training": len(train_subset),
        "validation": len(validation_subset),
        "test": len(test_data),
    })

    return train_subset, validation_subset, test_data
