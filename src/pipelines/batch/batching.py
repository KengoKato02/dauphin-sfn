# ==============================================================================
# Title:       Dataset Batching
# File:        /pipelines/batch/batching.py
# Author:      KATO Kengo
# Description: Functions for supplying training, validation and test batches
# ==============================================================================

from src.utils.logger import divider_logger, generic_logger
from torch.utils.data import DataLoader
from src.utils.decorators import handle_exceptions

import torch


@handle_exceptions
def batch_dataset(train_subset, validation_subset, test_data):
    """Create DataLoaders and inspect one training batch.

    Training order is shuffled using seed 42. Validation and test order is
    fixed. Batches contain up to 32 examples, including the final smaller batch.
    Loading uses the current process with num_workers=0.

    Returns:
        train_loader, validation_loader, test_loader: Loaders for each split.
    """
    batch_size = 32

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        generator=torch.Generator().manual_seed(42),
    )

    validation_loader = DataLoader(
        validation_subset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    batch_iterator = iter(train_loader)
    images, labels = next(batch_iterator)

    divider_logger()
    print("BATCHING")
    generic_logger({
        "image_shape": images.shape,
        "image_dtype": images.dtype,
        "label_shape": labels.shape,
        "label_dtype": labels.dtype,
        "labels": labels,
    })
    print("Training batches:", len(train_loader))

    return train_loader, validation_loader, test_loader
