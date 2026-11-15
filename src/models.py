# ==============================================================================
# Title:       Neural Network Models Imports
# File:        models.py
# Author:      KATO Kengo
# Description: Dedicated file for model imports relevant to Dauphin et al.
# ==============================================================================

from torch import nn
from math import prod


def build_mlp(input_shape: tuple[int, int, int], hidden_units: int = 5) -> nn.Sequential:
    """Build for Multi-layer Perceptron

        Return logits for ten classes, using tanh for smooth second derivatives.
    """
    if hidden_units < 1:
        raise ValueError("hidden_units must be positive.")
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(prod(input_shape), hidden_units),
        nn.Tanh(),
        nn.Linear(hidden_units, 10),
    )
