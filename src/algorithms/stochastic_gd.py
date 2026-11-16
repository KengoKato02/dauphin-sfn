"""Single-example stochastic gradient descent (batch size one)."""

from torch import Tensor
from torch.optim import SGD as TorchSGD
from collections.abc import Iterable
from src.utils.decorators import handle_exceptions


class SGD(TorchSGD):
    """Use PyTorch's plain SGD update; main.py supplies one example per step."""

    batch_size = 1

    @handle_exceptions
    def __init__(self, params: Iterable[Tensor], lr: float = 0.01) -> None:
        super().__init__(params, lr=lr)
