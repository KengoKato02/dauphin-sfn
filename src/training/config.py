# ==============================================================================
# Title:       Training Configuration
# File:        /src/training/config.py
# Author:      KATO Kengo
# Description: Define independent training settings and published MNIST presets
# ==============================================================================

from math import isfinite
from typing import Literal
from dataclasses import dataclass
from src.utils.decorators import handle_exceptions


@dataclass(frozen=True)
class ExperimentConfig:
    """Settings for one independently initialized model and training run.

    learning_rate stays fixed within the run. Each width can use a different
    configuration. batch_size=None means ONE FULL selected training batch.
    train_limit=None selects the complete prepared training split.
    curvature_samples=0 disables dense Hessian diagnostics for wider MSGD models.
    """

    dataset: Literal["mnist", "cifar10"] = "mnist"
    method: Literal["msgd", "damped_newton"] = "msgd"
    hidden_units: int = 5
    learning_rate: float = 0.05
    momentum: float = 0.0
    batch_size: int | None = 32
    epochs: int = 3
    train_limit: int | None = 512
    curvature_samples: int = 0
    max_parameters: int = 2000
    seed: int = 42
    name: str | None = None

    @handle_exceptions
    def __post_init__(self):
        if self.dataset not in ("mnist", "cifar10"):
            raise ValueError("dataset must be mnist or cifar10.")
        if self.method not in ("msgd", "damped_newton"):
            raise ValueError("method must be msgd or damped_newton.")
        if self.hidden_units < 1 or self.epochs < 0:
            raise ValueError("hidden_units must be positive and epochs nonnegative.")
        if not isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if not 0 <= self.momentum < 1:
            raise ValueError("momentum must be between 0 (inclusive) and 1 (exclusive).")
        if self.method == "damped_newton" and self.momentum != 0:
            raise ValueError("momentum applies to MSGD, not damped Newton.")
        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError("batch_size must be positive or None for full batch.")
        if self.train_limit is not None and self.train_limit < 1:
            raise ValueError("train_limit must be positive or None.")
        if self.curvature_samples < 0 or self.max_parameters < 1:
            raise ValueError("curvature_samples must be nonnegative and max_parameters positive.")


@handle_exceptions
def mnist_preprint_configs(epochs=50, train_limit=512, seed=42,
                           curvature_samples=0, max_parameters=2000):
    """Return the three MNIST SGD configurations from Table 1 of arXiv:1405.4604v2.

    The reported quantities are hidden units, learning rate, momentum and batch
    size. Seed, sample limit and curvature settings are our experiment choices.
    These are earlier MNIST results, not verified CIFAR-10 settings.
    """
    # Take from Pascanu & Dauphin et al.
    settings = (
        (5, 0.074, 0.031, 10),
        (25, 0.040, 0.017, 10),
        (50, 0.015, 0.254, 1),
    )
    return [
        ExperimentConfig(
            dataset="mnist", method="msgd", hidden_units=units,
            learning_rate=rate, momentum=momentum, batch_size=batch_size,
            epochs=epochs, train_limit=train_limit, seed=seed,
            curvature_samples=curvature_samples, max_parameters=max_parameters,
        )
        for units, rate, momentum, batch_size in settings
    ]
