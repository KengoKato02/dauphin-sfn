# ==============================================================================
# Title:       Dataset Image Transformation
# File:        /pipelines/transform/transformer.py
# Author:      KATO Kengo
# Description: Functions for converting, resizing and preparing dataset images
# ==============================================================================

from src.utils.decorators import handle_exceptions
from src.utils.logger import divider_logger, generic_logger
from torchvision.transforms import Compose, InterpolationMode, Resize, ToTensor


@handle_exceptions
def transform_dataset(train_data, test_data):
    """Inspect image conversion and resizing, then apply them to both datasets.

    Images are resized to 10 by 10 with bilinear interpolation and antialiasing.
    ToTensor converts pixels to floating-point values in [0, 1]. Grayscale
    images retain one channel; colour images retain three channels.

    Returns:
        train_data, test_data: The same datasets with their image transform set.
        Labels and downloaded files are unchanged.
    """
    image, label = train_data[0]

    # ========================== Tensor Conversion ========================== #

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
        "requires_grad": image_tensor.requires_grad,
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
    print("RESIZING")
    generic_logger({
        "resized_shape": resized_tensor.shape,
        "pixel_count": resized_tensor.numel(),
        "min": resized_tensor.min().item(),
        "max": resized_tensor.max().item(),
    })

    # ========================== Preparation Set ========================== #

    image_transform = Compose([
        resize,
        to_tensor,
    ])

    # Apply the same preparation whenever either dataset retrieves an image.
    train_data.transform = image_transform
    test_data.transform = image_transform

    prepared_image, label = train_data[0]

    divider_logger()
    print("PREPARATION SET")
    generic_logger({
        "shape": prepared_image.shape,
        "dtype": prepared_image.dtype,
        "min": prepared_image.min().item(),
        "max": prepared_image.max().item(),
        "label": label,
    })

    return train_data, test_data
