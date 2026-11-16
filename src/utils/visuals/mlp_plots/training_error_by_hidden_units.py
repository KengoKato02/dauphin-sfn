# ==============================================================================
# Title:       Training Error by Hidden Units
# File:        /src/utils/visuals/mlp_plots/training_error_by_hidden_units.py
# Author:      KATO Kengo
# Description: Compare minimum observed training error across MLP hidden-layer sizes
# ==============================================================================

from src.utils.decorators import handle_exceptions
from src.utils.visuals.mlp_plots.common import group_runs, create_plot, format_plot


@handle_exceptions
def plot_training_error_by_hidden_units(results, ax=None):
    """Plot minimum training error per method and width for MNIST or CIFAR-10.

    Each point uses the lowest observed error across epochs and repeated runs
    at that width. Pass ax to draw in an existing figure; otherwise create one.
    """
    ax = create_plot(results, ax)
    for label, runs in group_runs(results):
        widths = sorted({run["hidden_units"] for run in runs})
        errors = [
            min(item["train_error_pct"] for run in runs
                if run["hidden_units"] == width for item in run["history"])
            for width in widths
        ]
        ax.plot(widths, errors, marker="o", label=label)
    ax.set_xticks(sorted({run["hidden_units"] for run in results}))
    return format_plot(ax, "Hidden units", "Minimum train error (%)")
