# ==============================================================================
# Title:       Training Measurements
# File:        /src/training/metrics.py
# Author:      KATO Kengo
# Description: Measure training loss, classification error and optional negative curvature
# ==============================================================================

from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from src.utils.decorators import handle_exceptions
from src.algorithms.damped_newton import loss_gradient_hessian

import torch


@handle_exceptions
def measure_model(model, dataset, curvature_samples=128, max_parameters=2000):
    """Measure error on all selected training examples and curvature on a fixed probe.

    lambda_min is the smallest signed eigenvalue of the probe loss Hessian.
    Negative magnitude is zero when the Hessian has no negative eigenvalues.
    Set curvature_samples=0 to skip dense Hessians for larger MSGD models.
    """
    model.eval()
    total_loss = 0.0
    mistakes = 0
    with torch.no_grad():
        for images, labels in DataLoader(dataset, batch_size=256, shuffle=False):
            logits = model(images)
            total_loss += cross_entropy(logits, labels, reduction="sum").item()
            mistakes += (logits.argmax(dim=1) != labels).sum().item()

    minimum = None
    negative_magnitude = None
    if curvature_samples:
        images, labels = dataset.tensors
        _, _, hessian = loss_gradient_hessian(
            model, images[:curvature_samples], labels[:curvature_samples], max_parameters
        )
        minimum = torch.linalg.eigvalsh(hessian)[0].item()
        negative_magnitude = max(0.0, -minimum)

    return {
        "loss": total_loss / len(dataset),
        "train_error_pct": 100.0 * mistakes / len(dataset),
        "lambda_min": minimum,
        "negative_magnitude": negative_magnitude,
    }
