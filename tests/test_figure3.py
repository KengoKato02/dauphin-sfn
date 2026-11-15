"""Offline checks for the small Figure 3 experiment."""

from torch import nn
from src.models import build_mlp
from torch.optim import SGD
from torch.utils.data import TensorDataset
from src.experiments.figure3 import train_model, plot_results
from src.algorithms.damped_newton import DampedNewton, loss_gradient_hessian

import torch
import pytest
import matplotlib.pyplot as plt


@pytest.mark.parametrize("channels", [1, 3])
def test_mlp_accepts_image_batches(channels):
    model = build_mlp((channels, 10, 10), hidden_units=5)
    assert model(torch.zeros(4, channels, 10, 10)).shape == (4, 10)
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        channels * 100 * 5 + 5 + 5 * 10 + 10
    )


def test_exact_derivatives_match_binary_softmax():
    model = nn.Linear(1, 2, bias=False).double()
    with torch.no_grad():
        model.weight.zero_()
    images = torch.ones(1, 1, dtype=torch.float64)
    labels = torch.zeros(1, dtype=torch.int64)

    loss, gradient, hessian = loss_gradient_hessian(model, images, labels)

    assert loss.item() == pytest.approx(torch.log(torch.tensor(2.0)).item(), rel=1e-7)
    torch.testing.assert_close(gradient, torch.tensor([-0.5, 0.5], dtype=torch.float64))
    torch.testing.assert_close(
        hessian, torch.tensor([[0.25, -0.25], [-0.25, 0.25]], dtype=torch.float64)
    )
    with pytest.raises(ValueError, match="limit"):
        loss_gradient_hessian(model, images, labels, max_parameters=1)


def test_newton_matches_shifted_linear_solve_and_reduces_loss():
    model = nn.Linear(1, 2, bias=False).double()
    with torch.no_grad():
        model.weight.zero_()
    images = torch.ones(1, 1, dtype=torch.float64)
    labels = torch.zeros(1, dtype=torch.int64)
    before, gradient, hessian = loss_gradient_hessian(model, images, labels)
    optimizer = DampedNewton(model)

    after = optimizer.step(images, labels)

    assert after < before.item()
    assert optimizer.selected_damping in optimizer.damping_values
    expected = -torch.linalg.solve(
        hessian + optimizer.selected_damping * torch.eye(2, dtype=torch.float64), gradient
    )
    torch.testing.assert_close(model.weight.flatten(), expected)


def test_newton_leaves_parameters_when_no_candidate_improves():
    model = nn.Linear(1, 2, bias=False).double()
    before = model.weight.detach().clone()
    optimizer = DampedNewton(model)
    optimizer.step(torch.zeros(1, 1, dtype=torch.float64), torch.zeros(1, dtype=torch.int64))
    assert optimizer.selected_damping is None
    torch.testing.assert_close(model.weight, before)


def test_msgd_reduces_loss_and_repeats_with_same_seed():
    images = torch.cat((torch.zeros(4, 1, 10, 10), torch.ones(4, 1, 10, 10))).double()
    labels = torch.tensor([0] * 4 + [1] * 4)
    dataset = TensorDataset(images, labels)
    histories = []
    for _ in range(2):
        torch.manual_seed(42)
        model = build_mlp((1, 10, 10), hidden_units=1).double()
        optimizer = SGD(model.parameters(), lr=0.2)
        histories.append(train_model(
            model, dataset, optimizer, epochs=2, batch_size=4, curvature_samples=2
        ))
    assert histories[0] == histories[1]
    assert histories[0][-1]["loss"] < histories[0][0]["loss"]
    assert [item["epoch"] for item in histories[0]] == [0, 1, 2]
    assert all(item["negative_magnitude"] >= 0 for item in histories[0])


def test_rgb_newton_training_and_plot():
    torch.manual_seed(42)
    dataset = TensorDataset(torch.rand(4, 3, 10, 10).double(), torch.tensor([0, 1, 2, 3]))
    model = build_mlp((3, 10, 10), hidden_units=1).double()
    history = train_model(
        model, dataset, DampedNewton(model), epochs=1, batch_size=4, curvature_samples=2
    )
    assert len(history) == 2
    figure = plot_results([
        {"dataset": name, "method": "damped_newton", "hidden_units": 1, "history": history}
        for name in ("mnist", "cifar10")
    ])
    assert len(figure.axes) == 6
    assert figure.axes[2].get_yscale() == "log"
    plt.close(figure)
