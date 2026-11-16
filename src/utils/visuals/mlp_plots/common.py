# ==============================================================================
# Title:       Shared MLP Plot Utilities
# File:        /src/utils/visuals/mlp_plots/common.py
# Author:      KATO Kengo
# Description: Share plot setup, run labels and epoch curves across MLP visuals
# ==============================================================================

from matplotlib.ticker import MaxNLocator
from src.utils.decorators import handle_exceptions

import matplotlib.pyplot as plt


@handle_exceptions
def create_plot(results, ax=None):
    """Create or reuse axes for results from one dataset: MNIST or CIFAR-10."""
    if not results:
        raise ValueError("Provide at least one experiment result to plot.")
    datasets = {result["dataset"] for result in results}
    if len(datasets) != 1:
        raise ValueError("Pass results for one dataset at a time.")
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    ax.set_title(next(iter(datasets)).upper())
    return ax


@handle_exceptions
def group_runs(results):
    """Group runs by method and sort each group by its hidden-layer width."""
    methods = dict.fromkeys(result["method"] for result in results)
    for method in methods:
        label = "MSGD" if method == "msgd" else "Damped Newton"
        runs = sorted(
            (run for run in results if run["method"] == method),
            key=lambda run: run["hidden_units"],
        )
        yield label, runs


@handle_exceptions
def format_plot(ax, xlabel, ylabel, log_scale=False, epoch_axis=False):
    """Apply shared labels, axis formatting and legend, then return the figure."""
    ax.set(xlabel=xlabel, ylabel=ylabel, yscale="log" if log_scale else "linear")
    if epoch_axis:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.25)
    if ax.lines:
        ax.legend(fontsize="small")
    ax.figure.tight_layout()
    return ax.figure


@handle_exceptions
def plot_epoch_metric(results, metric, ylabel, ax=None, missing_message=None):
    """Plot every run's epoch history, omitting zero values on the log axis."""
    ax = create_plot(results, ax)
    for label, runs in group_runs(results):
        for index, run in enumerate(runs, start=1):
            history = run["history"]
            if not any(item[metric] is not None for item in history):
                continue
            width = run["hidden_units"]
            curve_label = run.get("name") or f"{label}, h={width}"
            if sum(other["hidden_units"] == width for other in runs) > 1:
                curve_label += f", run {run.get('run_id', index)}"
            ax.plot(
                [item["epoch"] for item in history],
                [item[metric] or float("nan") for item in history],
                label=curve_label,
            )

    if not ax.lines and missing_message:
        ax.text(0.5, 0.5, missing_message, ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    return format_plot(ax, "Epoch", ylabel, log_scale=True, epoch_axis=True)
