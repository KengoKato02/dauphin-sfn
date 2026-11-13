"""Check SFN against known curvature, including cases that fool Newton."""

import pytest
import torch

from descent_algorithms.sfn import SFN, saddle_free_direction


def test_spectral_absolute_value_preserves_descent_on_rotated_saddle():
    basis = torch.tensor([[1.0, -1.0], [1.0, 1.0]], dtype=torch.float64) / 2**0.5
    spectrum = torch.tensor([-2.0, 10.0], dtype=torch.float64)
    hessian = basis @ torch.diag(spectrum) @ basis.T
    gradient = torch.tensor([0.3, -0.7], dtype=torch.float64)
    direction = saddle_free_direction(gradient, hessian, damping=0.1)
    absolute_hessian = basis @ torch.diag(spectrum.abs()) @ basis.T
    torch.testing.assert_close((absolute_hessian + 0.1 * torch.eye(2)) @ direction, -gradient)
    assert gradient @ direction < 0


def test_sfn_moves_away_from_negative_curvature_and_toward_positive_curvature():
    point = torch.nn.Parameter(torch.tensor([0.8, 0.2], dtype=torch.float64))
    optimizer = SFN([point], lr=0.5, damping=0.1)
    before = point.detach().clone()
    optimizer.step(lambda: 5 * point[0] ** 2 - point[1] ** 2)
    expected = before + 0.5 * torch.tensor(
        [-10 * before[0] / 10.1, 2 * before[1] / 2.1], dtype=torch.float64
    )
    torch.testing.assert_close(point, expected)
    assert 0 < point[0] < before[0]
    assert point[1] > before[1]


def test_sfn_matches_independent_hessian_on_nonlinear_loss():
    def objective(point):
        return torch.sin(point[0] * point[1]) + point[0] ** 4

    point = torch.nn.Parameter(torch.tensor([0.4, -0.3], dtype=torch.float64))
    before = point.detach().clone().requires_grad_()
    gradient = torch.autograd.grad(objective(before), before)[0]
    hessian = torch.autograd.functional.hessian(objective, before)
    values, vectors = torch.linalg.eigh(hessian)
    expected = before - 0.1 * torch.linalg.solve(
        vectors @ torch.diag(values.abs()) @ vectors.T + 0.2 * torch.eye(2), gradient
    )
    SFN([point], lr=0.1, damping=0.2).step(lambda: objective(point))
    torch.testing.assert_close(point, expected)


def test_damping_handles_linear_loss_and_unused_parameters():
    point = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))
    unused = torch.nn.Parameter(torch.tensor([3.0], dtype=torch.float64))
    SFN([point, unused], lr=0.1, damping=0.5).step(lambda: 2 * point[0] + point[1])
    torch.testing.assert_close(point, torch.tensor([0.6, 1.8], dtype=torch.float64))
    torch.testing.assert_close(unused, torch.tensor([3.0], dtype=torch.float64))


def test_exact_stationary_saddle_has_zero_step():
    point = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    SFN([point]).step(lambda: point[0] ** 2 - point[1] ** 2)
    torch.testing.assert_close(point, torch.zeros_like(point))


def test_dense_parameter_limit_is_checked_before_a_step():
    with pytest.raises(ValueError, match="above the"):
        SFN([torch.nn.Parameter(torch.zeros(2001))])


@pytest.mark.parametrize("damping", [0, -0.1, float("inf"), float("nan")])
def test_invalid_damping_is_rejected(damping):
    with pytest.raises(ValueError, match="damping"):
        SFN([torch.nn.Parameter(torch.ones(2))], damping=damping)
