"""Small image datasets exercise the real transforms without network downloads."""

import pytest
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image

from utils import load


@pytest.fixture
def offline_datasets(monkeypatch):
    def dataset_class(channels, size):
        class Images(Dataset):
            def __init__(self, root, train, transform, download):
                self.train = train
                self.transform = transform

            def __len__(self):
                return 40 if self.train else 20

            def __getitem__(self, index):
                pixels = torch.full((channels, size, size), index * 5, dtype=torch.uint8)
                return self.transform(to_pil_image(pixels)), index % 10

        return Images

    monkeypatch.setattr(load.datasets, "MNIST", dataset_class(1, 28))
    monkeypatch.setattr(load.datasets, "CIFAR10", dataset_class(3, 32))
