# ==============================================================================
# Title:       Training Error by Epoch
# File:        /src/utils/visuals/mlp_plots/training_error_by_epoch.py
# Author:      KATO Kengo
# Description: Show MLP training classification error over successive epochs
# ==============================================================================

from src.utils.decorators import handle_exceptions
from src.utils.visuals.mlp_plots.common import plot_epoch_metric


@handle_exceptions
def plot_training_error_by_epoch(results, ax=None):
    """Plot every run's training error over epochs for MNIST or CIFAR-10.

    Use a logarithmic error axis. Pass ax to reuse an existing figure.
    """
    return plot_epoch_metric(results, "train_error_pct", "Train error (%)", ax=ax)
