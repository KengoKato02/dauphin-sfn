# ==============================================================================
# Title:       MNIST Data Preparation Pipeline
# File:        /src/pipelines/mnist.py
# Author:      KATO Kengo
# Description: Pipeline for importing, preparing and exporting MNIST
# ==============================================================================

from src.utils.decorators import handle_exceptions
from torchvision.datasets import MNIST
from src.pipelines.save.saver import save_dataset
from src.pipelines.load.loader import load_dataset, split_dataset
from src.pipelines.batch.batching import batch_dataset
from src.pipelines.transform.transformer import transform_dataset


@handle_exceptions
def run_mnist_pipeline():
    """Prepare and save MNIST, returning training, validation and test loaders.

    Follow 3 stage flow: 
        - Load 
        - Transform
        - Batch
    """
    # 1) Load
    train_data, test_data = load_dataset(MNIST)
    # 2) Transform
    train_data, test_data = transform_dataset(train_data, test_data)
    train_subset, validation_subset, test_data = split_dataset(train_data, test_data)
    save_dataset("mnist", train_subset, validation_subset, test_data)  # Save
    # 3) Batch
    return batch_dataset(train_subset, validation_subset, test_data)
