# ==============================================================================
# Title:       Figure 3 Starter Experiment
# File:        /src/experiments/figure3.py
# Author:      KATO Kengo
# Description: Train small classifiers and measure error and negative curvature
# ==============================================================================

from torch.utils.data import DataLoader, TensorDataset
from matplotlib.ticker import MaxNLocator
from torch.nn.functional import cross_entropy
from src.pipelines.load.loader import DATA_DIR
from src.algorithms.damped_newton import DampedNewton, loss_gradient_hessian

import torch
import matplotlib.pyplot as plt


def load_training_data(dataset_name, limit=512):
    """Read prepared training tensors; validation and test examples stay unused."""
    if dataset_name not in ("mnist", "cifar10"):
        raise ValueError("dataset_name must be mnist or cifar10.")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive or None.")
    path = DATA_DIR / "processed" / dataset_name / "train.pt"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    images = saved["images"][:limit].to(torch.float64)
    labels = saved["labels"][:limit].to(torch.int64)
    return TensorDataset(images, labels)


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


def train_model(model, dataset, optimizer, epochs=3, batch_size=32,
                curvature_samples=128, seed=42, max_parameters=2000):
    """Train with plain SGD on minibatches, or the exact damped Newton reference.

    Record epoch zero and one measurement after each complete training pass.
    A fresh loader generator gives comparison runs the same minibatch order.
    """
    if epochs < 0 or batch_size < 1 or curvature_samples < 0:
        raise ValueError("epochs and curvature_samples must be nonnegative; batch_size positive.")
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False,
        generator=torch.Generator().manual_seed(seed), num_workers=0,
    )
    history = [{"epoch": 0, **measure_model(model, dataset, curvature_samples, max_parameters)}]
    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in loader:
            if isinstance(optimizer, DampedNewton):
                optimizer.step(images, labels)
            else:
                optimizer.zero_grad(set_to_none=True)
                loss = cross_entropy(model(images), labels)
                loss.backward()
                optimizer.step()

        metrics = measure_model(model, dataset, curvature_samples, max_parameters)
        if not torch.isfinite(torch.tensor(metrics["loss"])):
            raise FloatingPointError("Training loss became non-finite.")
        history.append({"epoch": epoch, **metrics})
        print(f"Epoch {epoch}: loss={metrics['loss']:.4f}, "
              f"train error={metrics['train_error_pct']:.2f}%", flush=True)
    return history


def plot_results(results):
    """Plot the three Figure 3 quantities for each dataset that was run.

    Epoch curves use the largest configured width. Minimum error uses every
    recorded epoch. Zeros are omitted on log axes, rather than replaced by a
    fabricated positive value. These are single-run curves, not tuned optima.
    """
    datasets = list(dict.fromkeys(result["dataset"] for result in results))
    figure, axes = plt.subplots(len(datasets), 3, figsize=(14, 4 * len(datasets)), squeeze=False)
    for row, dataset_name in enumerate(datasets):
        runs = [result for result in results if result["dataset"] == dataset_name]
        largest_width = max(result["hidden_units"] for result in runs)
        methods = list(dict.fromkeys(result["method"] for result in runs))
        for method in methods:
            label = "MSGD" if method == "msgd" else "Damped Newton"
            method_runs = sorted(
                (run for run in runs if run["method"] == method),
                key=lambda run: run["hidden_units"],
            )
            axes[row, 0].plot(
                [run["hidden_units"] for run in method_runs],
                [min(item["train_error_pct"] for item in run["history"]) for run in method_runs],
                marker="o", label=label,
            )
            curve = next(run for run in method_runs if run["hidden_units"] == largest_width)
            history = curve["history"]
            axes[row, 1].plot(
                [item["epoch"] for item in history],
                [item["train_error_pct"] or float("nan") for item in history],
                label=label,
            )
            axes[row, 2].plot(
                [item["epoch"] for item in history],
                [item["negative_magnitude"] or float("nan") for item in history],
                label=label,
            )

        axes[row, 0].set(xlabel="Hidden units", ylabel="Minimum train error (%)")
        axes[row, 0].set_xticks(sorted({run["hidden_units"] for run in runs}))
        axes[row, 1].set(xlabel="Epoch", ylabel="Train error (%)", yscale="log")
        axes[row, 2].set(xlabel="Epoch", ylabel="|Most negative eigenvalue| (probe)",
                         yscale="log")
        for column, axis in enumerate(axes[row]):
            unit_label = "unit" if largest_width == 1 else "units"
            suffix = "" if column == 0 else f" — {largest_width} hidden {unit_label}"
            axis.set_title(f"{dataset_name.upper()}{suffix}")
            if column:
                axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.grid(True, alpha=0.25)
            axis.legend()
    figure.tight_layout()
    return figure
