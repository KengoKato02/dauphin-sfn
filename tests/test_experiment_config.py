"""Check configuration-driven runs without downloading either dataset."""

from dataclasses import replace
from torch.utils.data import TensorDataset
from src.training.config import ExperimentConfig, mnist_preprint_configs
from src.utils.decorators import handle_exceptions
from src.utils.visuals.mlp_plots.training_error_by_epoch import plot_training_error_by_epoch
from src.utils.visuals.mlp_plots.negative_curvature_by_epoch import plot_negative_curvature_by_epoch
from src.utils.visuals.mlp_plots.training_error_by_hidden_units import \
    plot_training_error_by_hidden_units

import torch
import pytest
import matplotlib.pyplot as plt
import src.training.runner as experiment


@handle_exceptions
def tiny_data(channels=1):
    images = torch.linspace(0, 1, 8 * channels * 4).reshape(8, channels, 2, 2)
    return TensorDataset(images, torch.tensor([0, 1] * 4))


@handle_exceptions
def test_published_presets():
    configs = mnist_preprint_configs(epochs=2, train_limit=8)
    assert [(c.hidden_units, c.learning_rate, c.momentum, c.batch_size) for c in configs] == [
        (5, 0.074, 0.031, 10), (25, 0.040, 0.017, 10), (50, 0.015, 0.254, 1),
    ]
    assert all(c.epochs == 2 and c.train_limit == 8 and c.curvature_samples == 0 for c in configs)


@handle_exceptions
def test_configs_reach_training_and_repeated_runs_are_independent(monkeypatch):
    loaded = []
    observed = []
    train = experiment.train_model

    @handle_exceptions
    def load(name, limit, training_data=None):
        loaded.append((name, limit))
        images, labels = tiny_data().tensors
        return TensorDataset(images.double(), labels)

    @handle_exceptions
    def record(model, data, optimizer, **settings):
        observed.append((model[1].out_features, optimizer.param_groups[0]["lr"],
                         optimizer.param_groups[0]["momentum"], settings))
        return train(model, data, optimizer, **settings)

    monkeypatch.setattr(experiment, "load_training_data", load)
    monkeypatch.setattr(experiment, "train_model", record)
    first = ExperimentConfig(hidden_units=2, learning_rate=0.1, momentum=0.2,
                             batch_size=3, epochs=2, train_limit=8)
    second = replace(first, hidden_units=4, learning_rate=0.01, momentum=0.5,
                     batch_size=4, epochs=1)
    results = experiment.run_experiments([first, second, first])

    assert loaded == [("mnist", 8)]
    assert [(width, rate, momentum, opts["batch_size"], opts["epochs"])
            for width, rate, momentum, opts in observed] == [
        (2, 0.1, 0.2, 3, 2), (4, 0.01, 0.5, 4, 1), (2, 0.1, 0.2, 3, 2),
    ]
    assert [len(result["history"]) for result in results] == [3, 2, 3]
    assert results[0]["history"] == results[2]["history"]
    assert results[0]["history"] is not results[2]["history"]
    assert results[0]["parameter_count"] != results[1]["parameter_count"]
    assert results[1]["config"]["learning_rate"] == 0.01


@handle_exceptions
def test_newton_full_batch_respects_selection_and_learning_rate(monkeypatch):
    steps = []
    step = experiment.DampedNewton.step

    @handle_exceptions
    def record(optimizer, images, labels):
        steps.append((len(images), optimizer.lr))
        return step(optimizer, images, labels)

    monkeypatch.setattr(experiment.DampedNewton, "step", record)
    config = ExperimentConfig(dataset="cifar10", method="damped_newton", hidden_units=1,
                              learning_rate=0.5, batch_size=None, epochs=2, train_limit=4)
    results = experiment.run_experiments([config], {"cifar10": tiny_data(channels=3)})
    assert steps == [(4, 0.5), (4, 0.5)]
    assert results[0]["training_examples"] == results[0]["effective_batch_size"] == 4
    assert results[0]["history"][-1]["loss"] < results[0]["history"][0]["loss"]


@handle_exceptions
def test_wide_model_hessian_limit_is_checked_before_training():
    config = ExperimentConfig(hidden_units=50, max_parameters=10, curvature_samples=2)
    with pytest.raises(ValueError, match="Disable curvature"):
        experiment.run_experiments([config], {"mnist": tiny_data()})


@handle_exceptions
def test_plot_keeps_mixed_widths_and_repeated_trials():
    configs = [
        ExperimentConfig(hidden_units=width, epochs=0, train_limit=4)
        for width in (1, 5, 5)
    ]
    configs.append(replace(configs[0], method="damped_newton", batch_size=None))
    results = experiment.run_experiments(configs, {"mnist": tiny_data()})
    figure, axes = plt.subplots(1, 3)
    plot_training_error_by_hidden_units(results, ax=axes[0])
    plot_training_error_by_epoch(results, ax=axes[1])
    plot_negative_curvature_by_epoch(results, ax=axes[2])
    assert len(axes[0].lines) == 2
    assert len(axes[1].lines) == 4
    assert len({line.get_label() for line in axes[1].lines}) == 4
    assert not axes[2].axison
    plt.close(figure)

    for plot in (plot_training_error_by_hidden_units, plot_training_error_by_epoch,
                 plot_negative_curvature_by_epoch):
        with pytest.raises(ValueError, match="at least one"):
            plot([])
        with pytest.raises(ValueError, match="one dataset"):
            plot([results[0], {**results[0], "dataset": "cifar10"}])
