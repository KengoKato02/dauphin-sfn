import pytest
import torch
from torch.utils.data import Subset

from utils.load import load_data


def source_indices(dataset):
    indices = list(range(len(dataset)))
    while isinstance(dataset, Subset):
        indices = [dataset.indices[index] for index in indices]
        dataset = dataset.dataset
    return set(indices)


@pytest.mark.parametrize("name,channels", [("mnist", 1), ("cifar10", 3)])
def test_shapes_splits_and_normalization(offline_datasets, name, channels):
    loaders = load_data(name, image_size=10, batch_size=4, train_limit=12, eval_limit=3)
    inputs, labels = next(iter(loaders.train))
    assert inputs.shape == (4, channels, 10, 10)
    assert loaders.input_shape == (channels, 10, 10)
    assert labels.dtype == torch.int64
    assert inputs.min() >= -1 and inputs.max() <= 1
    assert len(loaders.train.dataset) == 12
    assert len(loaders.validation.dataset) == len(loaders.test.dataset) == 3
    assert not source_indices(loaders.train.dataset) & source_indices(loaders.validation.dataset)


def test_evaluation_does_not_change_training_shuffle(offline_datasets):
    first = load_data("mnist", batch_size=4, seed=7)
    torch.manual_seed(12345)
    second = load_data("mnist", batch_size=4, seed=7)
    next(iter(first.validation))
    next(iter(first.test))
    torch.testing.assert_close(next(iter(first.train))[0], next(iter(second.train))[0])
    assert source_indices(first.train.dataset) == source_indices(second.train.dataset)


def test_native_resolution_is_preserved(offline_datasets):
    loaders = load_data("cifar10")
    assert loaders.input_shape == (3, 32, 32)
    assert next(iter(loaders.test))[0].shape[1:] == (3, 32, 32)
