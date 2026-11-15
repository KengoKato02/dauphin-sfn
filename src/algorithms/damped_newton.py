# ==============================================================================
# Title:       Damped Newton
# File:        /src/algorithms/damped_newton.py
# Author:      KATO Kengo
# Description: Exact Hessian calculations and a damped Newton comparison step
# ==============================================================================

from torch.func import functional_call
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.nn.functional import cross_entropy

import torch


def parameter_loss(model, vector, images, labels):
    """Evaluate cross entropy at a parameter vector without changing the model."""
    parameters = {}
    offset = 0
    for name, parameter in model.named_parameters():
        size = parameter.numel()
        parameters[name] = vector[offset:offset + size].view_as(parameter)
        offset += size
    logits = functional_call(model, parameters, (images,))
    return cross_entropy(logits, labels)


def loss_gradient_hessian(model, images, labels, max_parameters=2000):
    """Calculate exact derivatives of this batch's mean cross entropy.

    The size limit keeps this reference implementation on small models. It
    checks before allocating the dense parameter-by-parameter Hessian.
    """
    vector = parameters_to_vector(model.parameters()).detach().requires_grad_(True)
    if vector.numel() > max_parameters:
        raise ValueError(
            f"Exact Hessian needs {vector.numel()} parameters; limit is {max_parameters}. "
            "Use fewer hidden units, or disable curvature tracking for a wider MSGD run."
        )

    def objective(parameters):
        return parameter_loss(model, parameters, images, labels)

    loss = objective(vector)
    gradient, = torch.autograd.grad(loss, vector)
    hessian = torch.autograd.functional.hessian(objective, vector, vectorize=True)
    hessian = (hessian + hessian.T) / 2
    return loss.detach(), gradient.detach(), hessian.detach()


class DampedNewton:
    """Try positive-definite H + damping * I shifts and keep the best batch loss.

    The damping grid comes from Appendix F.1. Candidate selection and the
    optional rejected step are explicit choices for this small reference.
    """

    def __init__(self, model, lr=1.0, max_parameters=2000):
        self.model = model
        self.lr = lr
        self.max_parameters = max_parameters
        self.damping_values = (1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001)
        self.selected_damping = None
        if lr <= 0:
            raise ValueError("lr must be positive.")

    def step(self, images, labels):
        """Apply one Newton step using derivatives and candidates on the same batch."""
        loss, gradient, hessian = loss_gradient_hessian(
            self.model, images, labels, self.max_parameters
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
        vector = parameters_to_vector(self.model.parameters()).detach()
        projected_gradient = eigenvectors.T @ gradient
        best_vector = vector
        best_loss = loss.item()
        self.selected_damping = None

        with torch.no_grad():
            for damping in self.damping_values:
                if eigenvalues[0].item() + damping <= 0:
                    continue
                direction = eigenvectors @ (projected_gradient / (eigenvalues + damping))
                candidate = vector - self.lr * direction
                candidate_loss = parameter_loss(self.model, candidate, images, labels).item()
                if candidate_loss < best_loss:
                    best_loss = candidate_loss
                    best_vector = candidate
                    self.selected_damping = damping
            vector_to_parameters(best_vector, self.model.parameters())

        return best_loss
