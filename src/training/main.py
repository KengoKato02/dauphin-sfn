# ==============================================================================
# Title:       Training Pipelines
# File:        /src/training/main.py
# Author:      KATO Kengo
# Description: Load selected datasets and orchestrate configured MLP training runs
# ==============================================================================

from src.utils.loaders import load_training_data
from src.training.config import ExperimentConfig
from src.training.runner import run_experiment
from src.utils.decorators import handle_exceptions


@handle_exceptions
def run_experiments(configs, training_data=None):
    """Validate configurations, reuse selected data and dispatch each training run.

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
        results.append(run_experiment(config, dataset, run_id))
    return results
