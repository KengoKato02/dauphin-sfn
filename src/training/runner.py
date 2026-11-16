# ==============================================================================
# Title:       Training Runner
# File:        /src/training/runner.py
# Author:      KATO Kengo
# Description: Run configured training jobs and collect their settings and history
# ==============================================================================

from dataclasses import asdict
from src.utils.decorators import handle_exceptions
from src.models.neural_networks.mlp import build_mlp
from src.training.msgd_training.trainer import train_msgd
from src.training.damped_training.trainer import train_damped_newton

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
def run_experiment(config, dataset, run_id):
    """Build one model, select its trainer and return settings and measurements."""
    model, parameter_count = _create_model(config, dataset, run_id)
    batch_size = len(dataset) if config.batch_size is None else config.batch_size

    print(f"Run {run_id}: {config.dataset} | {config.method} | "
          f"h={config.hidden_units} | lr={config.learning_rate} | "
          f"momentum={config.momentum} | batch={batch_size}", flush=True)
    if config.method == "msgd":
        history = train_msgd(model, dataset, config)
    else:
        history = train_damped_newton(model, dataset, config)
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
