# ==============================================================================
# Title:       Dataset Image Transformation
# File:        /src/pipelines/transform/transformer.py
# Author:      KATO Kengo
# Description: Functions for converting, resizing and preparing dataset images
# ==============================================================================

from src.utils.logger import divider_logger, generic_logger
from src.utils.decorators import handle_exceptions
from torchvision.transforms import Resize, Compose, ToTensor, InterpolationMode


@handle_exceptions
def _image_operations():
    """Share the same resize and tensor conversion between the pipeline and demo."""
    resize = Resize(
        size=(10, 10),
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    return resize, ToTensor()


@handle_exceptions
def transform_dataset(train_data, test_data):
    """Attach automatic image preparation to the training and test datasets.

    Images are resized to 10 by 10 with bilinear interpolation and antialiasing.
    ToTensor converts pixels to floating-point values in [0, 1]. Grayscale
    images retain one channel; colour images retain three channels. B vs RGB

    Returns:
        train_data, test_data: The same datasets with their image transform set.
        Labels and downloaded files are unchanged.
    """
    image_transform = Compose(_image_operations())

    # Preparation runs when an image is retrieved, not when this function runs.
    train_data.transform = image_transform
    test_data.transform = image_transform

    return train_data, test_data


@handle_exceptions
def demonstrate_image_transform(image):
    """Inspect tensor conversion and resizing for one original PIL image.

    Call this separately from a demonstration utility. It prints the original
    and resized tensor properties without changing any dataset.

    Returns:
        image_tensor, resized_tensor: Tensors suitable for comparison or plotting.
    """
    resize, to_tensor = _image_operations()

    # ========================== Tensor Conversion ========================== #

    image_tensor = to_tensor(image)

    divider_logger()
    print("TENSOR CONVERSION")
    generic_logger({
        "tensor_type": type(image_tensor),
        "shape": image_tensor.shape,
        "dtype": image_tensor.dtype,
        "min": image_tensor.min().item(),
        "max": image_tensor.max().item(),
        "requires_grad": image_tensor.requires_grad,
    })

    # ============================== Resizing ============================== #
    # Both MNIST and CIFAR-10 are downsampled to 10 x 10 for this experiment.

    resized_image = resize(image)
    resized_tensor = to_tensor(resized_image)

    divider_logger()
    print("RESIZING")
    generic_logger({
        "resized_shape": resized_tensor.shape,
        "pixel_count": resized_tensor.numel(),
        "min": resized_tensor.min().item(),
        "max": resized_tensor.max().item(),
    })

    return image_tensor, resized_tensor
