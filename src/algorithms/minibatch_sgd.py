"""Stochastic gradient descent on the mean loss of a minibatch."""

from torch import Tensor
from collections.abc import Iterable
from torch.optim import SGD as TorchSGD


class MiniBatchSGD(TorchSGD):
    """The same update as SGD, with multiple examples per step supplied by main.py."""

    def __init__(self, params: Iterable[Tensor], lr: float = 0.01, batch_size: int = 64) -> None:
        if batch_size < 2:
            raise ValueError(
                "MiniBatchSGD requires batch_size >= 2; use SGD for batch size one.")
        self.batch_size = batch_size
        super().__init__(params, lr=lr)
