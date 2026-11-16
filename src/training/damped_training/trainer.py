# ==============================================================================
# Title:       Damped Newton Training
# File:        /src/training/damped_training/trainer.py
# Author:      KATO Kengo
# Description: Train an MLP using the exact dense damped Newton update
# ==============================================================================

from src.training.common import train_model
from src.utils.decorators import handle_exceptions
from src.algorithms.damped_newton import DampedNewton


@handle_exceptions
def train_damped_newton(model, dataset, config):
    """Supply Newton's batch update to the same shared epoch loop as MSGD.

    batch_size=None in config makes each update use the complete selected set.
    """
    optimizer = DampedNewton(model, lr=config.learning_rate, max_parameters=config.max_parameters)
    return train_model(model, dataset, optimizer.step, config)
