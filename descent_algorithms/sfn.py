"""Exact, damped saddle-free Newton for small educational problems.

This implements Eq. (5) of Dauphin et al. (2014), with fixed positive damping.
It deliberately forms the full Hessian. Algorithm 1's Krylov approximation and
per-update damping selection are later extensions, not implemented here.
"""

from collections.abc import Callable, Iterable
from math import isfinite

import torch
from torch import Tensor
from torch.optim import Optimizer


def saddle_free_direction(gradient: Tensor, hessian: Tensor, damping: float) -> Tensor:
    """Compute -(abs(H) + damping I)^(-1) g in the Hessian eigenbasis.

    abs(H) means absolute *eigenvalues*, not elementwise absolute values.
    The spectral solve avoids explicitly forming a matrix inverse.
    """
    if not isfinite(damping) or damping <= 0:
        raise ValueError("damping must be finite and strictly positive.")
    if gradient.ndim != 1 or hessian.shape != (gradient.numel(), gradient.numel()):
        raise ValueError("Expected a flat gradient and a matching square Hessian.")
    if not torch.isfinite(gradient).all() or not torch.isfinite(hessian).all():
        raise ValueError("The gradient and Hessian must contain only finite values.")
    # Roundoff can introduce tiny asymmetries in an otherwise symmetric Hessian.
    eigenvalues, eigenvectors = torch.linalg.eigh((hessian + hessian.T) / 2)
    projected_gradient = eigenvectors.T @ gradient
    return -(eigenvectors @ (projected_gradient / (eigenvalues.abs() + damping)))


class SFN(Optimizer):
    """Dense SFN with a closure that returns a scalar loss without backward().

    Costs O(P^2) Hessian storage and O(P^3) eigendecomposition for P parameters.
    CPU float64 is recommended for studying the Hessian spectrum.
    """
    