# ==============================================================================
# Title:       Prepared Data Loading
# File:        /src/utils/loaders.py
# Author:      KATO Kengo
# Description: Load and select prepared training tensors for model training
# ==============================================================================

from torch.utils.data import TensorDataset
from src.utils.decorators import handle_exceptions
from src.pipelines.load.loader import DATA_DIR

import torch


@handle_exceptions
def load_training_data(dataset_name, limit=512, training_data=None):
    """Select training tensors from a saved export or supplied dataset mapping.

    Return CPU float64 images and int64 labels. Validation and test examples
    stay unused. Set limit=None to select the full prepared training split.
    """
    if dataset_name not in ("mnist", "cifar10"):
        raise ValueError("dataset_name must be mnist or cifar10.")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive or None.")

    if training_data is None:
        path = DATA_DIR / "processed" / dataset_name / "train.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        images, labels = saved["images"], saved["labels"]
    else:
        images, labels = training_data[dataset_name].tensors

    images = images[:limit].to(device="cpu", dtype=torch.float64)
    labels = labels[:limit].to(device="cpu", dtype=torch.int64)
    return TensorDataset(images, labels)
