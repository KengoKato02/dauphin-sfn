"""MNIST and CIFAR-10 loaders with reproducible, disjoint validation splits."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DataLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader
    input_shape: tuple[int, int, int]


def _limit(dataset: Dataset, limit: int | None, seed: int) -> Dataset:
    if limit is None:
        return dataset
    if limit < 1:
        raise ValueError("Subset limits must be positive, or None for the full split.")
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))
    return Subset(dataset, indices[:limit].tolist())


def load_data(
    name: str,
    *,
    batch_size: int = 64,
    data_dir: str | Path = PROJECT_ROOT / "data",
    download: bool = False,
    image_size: int | None = None,
    validation_fraction: float = 0.1,
    train_limit: int | None = None,
    eval_limit: int | None = None,
    seed: int = 42,
    num_workers: int = 0,
) -> DataLoaders:
    """Load normalized images, keeping the official test set out of training.

    ``image_size=None`` preserves native resolution. Use 10 for the paper's
    small-network setting; CIFAR-10 retains its three color channels.
    Downloading is opt-in. No dataset is loaded when this module is imported.
    """
    name = name.lower()
    if name not in {"mnist", "cifar10"}:
        raise ValueError("name must be 'mnist' or 'cifar10'.")
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie strictly between 0 and 1.")
    if image_size is not None and image_size < 1:
        raise ValueError("image_size must be positive, or None for native resolution.")
    for limit in (train_limit, eval_limit):
        if limit is not None and limit < 1:
            raise ValueError("Subset limits must be positive, or None for the full split.")

    channels, native_size = (1, 28) if name == "mnist" else (3, 32)
    operations = []
    if image_size is not None:
        operations.append(transforms.Resize((image_size, image_size), antialias=True))
    # A fixed mapping to [-1, 1] avoids estimating statistics from validation/test data.
    operations.extend(
        [transforms.ToTensor(), transforms.Normalize((0.5,) * channels, (0.5,) * channels)]
    )
    transform = transforms.Compose(operations)
    dataset_class = datasets.MNIST if name == "mnist" else datasets.CIFAR10
    training_data = dataset_class(
        root=str(data_dir), train=True, transform=transform, download=download
    )
    test_data = dataset_class(
        root=str(data_dir), train=False, transform=transform, download=download
    )

    validation_size = int(len(training_data) * validation_fraction)
    if not 0 < validation_size < len(training_data):
        raise ValueError("The validation split must leave at least one example in each split.")
    train_data, validation_data = random_split(
        training_data,
        [len(training_data) - validation_size, validation_size],
        generator=torch.Generator().manual_seed(seed),
    )
    splits = (
        _limit(train_data, train_limit, seed + 1),
        _limit(validation_data, eval_limit, seed + 2),
        _limit(test_data, eval_limit, seed + 3),
    )
    loaders = [
        DataLoader(
            split,
            batch_size=batch_size,
            shuffle=(index == 0),
            num_workers=num_workers,
            # Separate generators keep evaluation from changing training's shuffle order.
            generator=torch.Generator().manual_seed(seed + index),
        )
        for index, split in enumerate(splits)
    ]
    size = image_size if image_size is not None else native_size
    return DataLoaders(*loaders, input_shape=(channels, size, size))
