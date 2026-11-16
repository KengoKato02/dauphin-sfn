# ==============================================================================
# Title:       Negative Curvature by Epoch
# File:        /src/utils/visuals/mlp_plots/negative_curvature_by_epoch.py
# Author:      KATO Kengo
# Description: Show the magnitude of the most negative Hessian eigenvalue by epoch
# ==============================================================================

from src.utils.decorators import handle_exceptions
from src.utils.visuals.mlp_plots.common import plot_epoch_metric


@handle_exceptions
def plot_negative_curvature_by_epoch(results, ax=None):
    """Plot negative Hessian eigenvalue magnitude for MNIST or CIFAR-10.

    Use the recorded training probe measurements. Zero magnitudes are omitted
    on the log axis. Display a message when curvature tracking was disabled.
    """
    return plot_epoch_metric(
        results, "negative_magnitude", "|Most negative eigenvalue| (probe)", ax=ax,
        missing_message="Curvature tracking disabled\nEnable curvature_samples in a config",
    )
