# ==============================================================================
# Title:       MNIST Plots
# File:        utils/visuals/mnist.py
# Author:      KATO Kengo
# Description: Functions leveraging Matplotlib for MNIST related visuals
# ==============================================================================

from src.utils.decorators import handle_exceptions

import matplotlib.pyplot as plt
import torch


@handle_exceptions
def plot_mnist_resize_comparison(
    original_tensor: torch.Tensor,
    resized_tensor: torch.Tensor,
) -> None:
    """Display an MNIST image before and after resizing, side by side.

    Args:
        original_tensor: Original CPU image with shape (1, height, width).
        resized_tensor: Resized CPU image with shape (1, new_height, new_width).

    Both tensors should contain grayscale pixel values in [0, 1] and require
    no gradients. The same brightness scale is used for both images.
    This function displays the images; it does not resize or change them.
    """
    fig, axes = plt.subplots(1, 2)

    # Select the single grayscale channel to give imshow a 2D image.
    axes[0].imshow(
        original_tensor[0],
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axes[0].set_title(
        f"Original: {original_tensor.shape[1]} × {original_tensor.shape[2]}"
    )

    # Show individual pixels without smoothing them for display.
    axes[1].imshow(
        resized_tensor[0],
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axes[1].set_title(
        f"Resized: {resized_tensor.shape[1]} × {resized_tensor.shape[2]}"
    )

    fig.tight_layout()
    plt.show()
