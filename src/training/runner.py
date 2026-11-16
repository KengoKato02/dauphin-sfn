# ==============================================================================
# Title:       Training Runner
# File:        /src/training/runner.py
# Author:      KATO Kengo
# Description: Run configured training jobs and collect their settings and history
# ==============================================================================

from dataclasses import asdict
from torch.optim import SGD
from src.utils.loaders import load_training_data
from src.training.config import ExperimentConfig
from src.training.trainer import train_model
from src.utils.decorators import handle_exceptions
from src.algorithms.damped_newton import DampedNewton
from src.models.neural_networks.mlp import build_mlp

import torch


@handle_exceptions
def _create_model(config, dataset, run_id):
    """Initialize a model from the run's seed and check the existing Hessian limit."""
    torch.manual_seed(config.seed)
    input_shape = tuple(dataset.tensors[0].shape[1:])
    model = build_mlp(input_shape, config.hidden_units).double()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    if (config.curvature_samples or config.method == "damped_newton") and (
        parameter_count > config.max_parameters
    ):
        raise ValueError(
            f"Run {run_id} needs a {parameter_count}-parameter Hessian; "
            f"limit is {config.max_parameters}. Disable curvature for a wider MSGD "
            "run, or raise max_parameters deliberately."
        )
    return model, parameter_count


@handle_exceptions
def _create_optimizer(config, model):
    """Create a fresh optimizer using this run's learning rate and momentum."""
    if config.method == "msgd":
        return SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    return DampedNewton(model, lr=config.learning_rate, max_parameters=config.max_parameters)


@handle_exceptions
def _run_experiment(config, dataset, run_id):
    """Build and train one model, returning its settings and measurements."""
    model, parameter_count = _create_model(config, dataset, run_id)
    optimizer = _create_optimizer(config, model)
    batch_size = len(dataset) if config.batch_size is None else config.batch_size

    print(f"Run {run_id}: {config.dataset} | {config.method} | "
          f"h={config.hidden_units} | lr={config.learning_rate} | "
          f"momentum={config.momentum} | batch={batch_size}", flush=True)
    history = train_model(
        model, dataset, optimizer, epochs=config.epochs, batch_size=batch_size,
        curvature_samples=config.curvature_samples, seed=config.seed,
        max_parameters=config.max_parameters,
    )
    return {
        "run_id": run_id,
        "name": config.name,
        "dataset": config.dataset,
        "method": config.method,
        "hidden_units": config.hidden_units,
        "config": asdict(config),
        "training_examples": len(dataset),
        "effective_batch_size": batch_size,
        "parameter_count": parameter_count,
        "dtype": "float64",
        "device": "cpu",
        "history": history,
    }


@handle_exceptions
def run_experiments(configs, training_data=None):
    """Validate configurations, reuse selected data and run each model separately.

    Optionally supply a mapping of dataset names to TensorDatasets. Each run
    gets a fresh model, optimizer and shuffled loader, as before.
    """
    configs = list(configs)
    if not configs:
        raise ValueError("Provide at least one experiment configuration.")
    if any(not isinstance(config, ExperimentConfig) for config in configs):
        raise TypeError("configs must contain ExperimentConfig objects.")

    cached_data = {}
    results = []
    for run_id, config in enumerate(configs, start=1):
        key = (config.dataset, config.train_limit)
        if key not in cached_data:
            cached_data[key] = load_training_data(config.dataset, config.train_limit, training_data)
        dataset = cached_data[key]
        if len(dataset) == 0:
            raise ValueError("The selected training dataset is empty.")
        results.append(_run_experiment(config, dataset, run_id))
    return results
