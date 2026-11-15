# ==============================================================================
# Title:       Batch Gradient Descent Algorithm
# File:        batch_gd.py
# Author:      KATO Kengo
# Description: Implementation of Classical Batch Gradient Descent
# ==============================================================================

from src.utils.decorators import handle_exceptions

import torch

# ============================ Hand-Written Examples ============================ #


@handle_exceptions
def _compute_gradient(
    loss: torch.Tensor,
    theta_curr: torch.Tensor
) -> torch.Tensor:
    # Type cast for Handling Tensor
    (gradient,) = torch.autograd.grad(loss, theta_curr)
    return gradient


@handle_exceptions
def gradient_descent_singleton(
    theta_curr: torch.Tensor,
    loss: torch.Tensor,
    learning_rate: float
) -> torch.Tensor:
    """Implement basic batch Gradient Descent.

    Formula: theta_t+1 = theta_t - learning_rate(grad(theta_t))

    Args: 
        theta_curr (dtype: torch.Tensor): Training parameters at current
        loss (dtype: torch.tensor): Loss value using MSE
        learning_rate (dtype: float): Set learning rate

    Returns
        theta_next (Vector): Updated training params vector based on
        negative direction fo descent
    """
    # Compute derivative(loss)/derivate(theta)
    derived_loss = _compute_gradient(loss, theta_curr)
    theta_next = theta_curr - learning_rate * derived_loss
    return theta_next
