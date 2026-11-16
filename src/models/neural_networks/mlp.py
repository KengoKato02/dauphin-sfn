# ==============================================================================
# Title:       Single Hidden Layer MLP
# File:        /src/models/neural_networks/mlp.py
# Author:      KATO Kengo
# Description: Build a small classifier for resized MNIST and CIFAR-10 images
# ==============================================================================

from math import prod
from torch import nn
from src.utils.decorators import handle_exceptions


@handle_exceptions
def build_mlp(input_shape: tuple[int, int, int], hidden_units: int = 5) -> nn.Sequential:
    """Build a classifier with one tanh hidden layer and ten output scores.

    Args:
        input_shape: Image shape (channels, height, width), without the batch.
        hidden_units: Number of neurons in the single hidden layer.

    Images are flattened inside the model. The output contains logits, so use
    cross entropy directly without adding a separate softmax layer.
    """
    if len(input_shape) != 3 or any(size < 1 for size in input_shape):
        raise ValueError("input_shape must contain positive channels, height and width.")
    if hidden_units < 1:
        raise ValueError("hidden_units must be positive.")

    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(prod(input_shape), hidden_units),
        nn.Tanh(),  # Use Hyperbolic Tangent for activation function
        nn.Linear(hidden_units, 10),  # Output: 10 for 0 - 9
    )
