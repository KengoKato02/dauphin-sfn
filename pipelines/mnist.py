# ==============================================================================
# Title:       MNIST Data Preparation Pipeline
# File:        /pipelines/mnist.py
# Author:      KATO Kengo
# Description: Pipeline for importing, preparing and exporting MNIST
# ==============================================================================

import torch

from utils.logger import divider_logger, generic_logger
from viz.mnist import plot_mnist_resize_comparison
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision.transforms import (
    InterpolationMode,
    ToTensor,
    Compose,
    Resize
)

# ========================== Train & Test ========================== #

"""Import both training and testing datasets 
Understand the meaning behind each parameters:
    - root="data": store or find the files in the project data directory.
    - train=True: open the official training portion.
    - train=False: open the official test portion.
    - download=True: download missing files. Existing downloads are reused.
"""

train_data = MNIST(
    root="data",
    train=True,
    download=True
)

test_data = MNIST(
    root="data",
    train=False,
    download=True
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
    "label_type": type(label)
})


# ========================== Tensor Conversion ========================== #

# Convert MNIST image to a native, readable Tensor (Matrix)
# Image -> Tensor. Arrange dimensions. Convert Pixel Values to binary
to_tensor = ToTensor()
image_tensor = to_tensor(image)

divider_logger()
print("TENSOR CONVERSION")
generic_logger({
    "tensor_type": type(image_tensor),
    "shape": image_tensor.shape,
    "dtype": image_tensor.dtype,
    "min": image_tensor.min().item(),
    "max": image_tensor.max().item(),
    "requires_grad": image_tensor.requires_grad
})


# ========================== Resizing ========================== #
"""TODO: Fill Comments for meaning
"""

resize = Resize(
    size=(10, 10),
    interpolation=InterpolationMode.BILINEAR,  # TODO: Explain
    antialias=True,
)

resized_image = resize(image)
resized_tensor = to_tensor(resized_image)

# Log
divider_logger()
print("RESIZING")
generic_logger({
    "resized_shape": resized_tensor.shape,
    "pixel_count": resized_tensor.numel(),
    "min": resized_tensor.min().item(),
    "max": resized_tensor.max().item()
})

# Plot
plot_mnist_resize_comparison(image_tensor, resized_tensor)


# ========================== Preparation Set ========================== #
"""TODO: Fill Comments for meaning
"""

image_transform = Compose([
    resize,
    to_tensor,
])

prepared_train_data = MNIST(
    root="data",
    train=True,
    download=False,
    transform=image_transform,
)

prepared_test_data = MNIST(
    root="data",
    train=False,
    download=False,
    transform=image_transform,
)

prepared_image, label = prepared_train_data[0]

# Log
divider_logger()
print("PREPARATION SET")
generic_logger({
    "shape": prepared_image.shape,
    "dtype": prepared_image.dtype,
    "min": prepared_image.min().item(),
    "max": prepared_image.max().item(),
    "label": label
})


# ========================== SPLIT ========================== #
"""TODO: Fill Comments for meaning
"""

# Make the random selection repeatable.
split_generator = torch.Generator().manual_seed(42)

train_subset, validation_subset = random_split(
    prepared_train_data,
    lengths=[54000, 6000],
    generator=split_generator,
)

# Log
divider_logger()
print("SPLIT")
generic_logger({
    "training": len(train_subset),
    "validation": len(validation_subset),
    "test": len(prepared_test_data)
})
