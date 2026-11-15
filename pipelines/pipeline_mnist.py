# ==============================================================================
# Title:       MNIST Data Preparation Pipeline
# File:        /pipelines/mnist.py
# Author:      KATO Kengo
# Description: Pipeline for importing, preparing and exporting MNIST
# ==============================================================================

from torchvision.transforms import ToTensor, Resize, InterpolationMode
from src.utils.logger import divider_logger, generic_logger
from torchvision.datasets import MNIST

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

print("Training examples:", len(train_data))
print("Test examples:", len(test_data))

print("Image type:", type(image))
print("Image size:", image.size)

print("Label:", label)
print("Label type:", type(label))

# ========================== Tensor Conversion ========================== #

# Convert MNIST image to a native, readable Tensor (Matrix)
# Image -> Tensor. Arrange dimensions. Convert Pixel Values to binary
to_tensor = ToTensor()
image_tensor = to_tensor(image)

divider_logger()
generic_logger({
    "tensor_type": type(image_tensor),
    "shape": image_tensor.shape,
    "dtype": image_tensor.dtype,
    "min": image_tensor.min().item(),
    "max": image_tensor.max().item(),
    "requires_grad": image_tensor.requires_grad
})

# ========================== Resizing ========================== #

resize = Resize(
    size=(10, 10),
    interpolation=InterpolationMode.BILINEAR,
    antialias=True,
)

resized_image = resize(image)
resized_tensor = to_tensor(resized_image)

divider_logger()
generic_logger({
    "resized_shape": resized_tensor.shape,
    "pixel_count": resized_tensor.numel(),
    "min": resized_tensor.min().item(),
    "max": resized_tensor.max().item()
})
