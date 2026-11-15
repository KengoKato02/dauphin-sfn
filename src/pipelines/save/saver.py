# ==============================================================================
# Title:       Processed Dataset Storage
# File:        /src/pipelines/save/saver.py
# Author:      KATO Kengo
# Description: Save prepared image tensors, labels and split indices to disk
# ==============================================================================

from src.utils.logger import divider_logger, generic_logger
from torch.utils.data import Subset
from src.utils.decorators import handle_exceptions
from src.pipelines.load.loader import DATA_DIR

import torch
import torchvision


@handle_exceptions
def save_dataset(dataset_name, train_subset, validation_subset, test_data):
    """Save transformed examples for each split under the repository data folder.

    Each .pt file contains images of shape (N, channels, 10, 10), integer labels,
    original example indices, the transform description and package versions.
    Training and validation indices refer to the official training portion.
    Test indices refer to the official test portion.

    Args:
        dataset_name: Output directory name, such as mnist or cifar10.
        train_subset: Prepared training examples after the validation split.
        validation_subset: Prepared validation examples held out from training.
        test_data: Prepared official test examples.

    Returns:
        output_dir: Directory containing train.pt, validation.pt and test.pt.
    """
    output_dir = DATA_DIR / "processed" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = (
        ("train", train_subset),
        ("validation", validation_subset),
        ("test", test_data),
    )

    for split_name, dataset in splits:
        first_image, _ = dataset[0]
        images = torch.empty(
            (len(dataset), *first_image.shape),
            dtype=first_image.dtype,
        )
        labels = torch.empty(len(dataset), dtype=torch.int64)

        # Retrieve every example so its resize and tensor conversion are applied.
        for index in range(len(dataset)):
            image, label = dataset[index]
            images[index] = image
            labels[index] = label

        if isinstance(dataset, Subset):
            source_data = dataset.dataset
            indices = torch.tensor(dataset.indices, dtype=torch.int64)
        else:
            source_data = dataset
            indices = torch.arange(len(dataset))

        torch.save({
            "images": images,
            "labels": labels,
            "indices": indices,
            "source_split": "test" if split_name == "test" else "train",
            "transform": repr(source_data.transform),
            "torch_version": str(torch.__version__),
            "torchvision_version": str(torchvision.__version__),
        }, output_dir / f"{split_name}.pt")

    divider_logger()
    print("PROCESSED DATA")
    generic_logger({"directory": output_dir})

    return output_dir
