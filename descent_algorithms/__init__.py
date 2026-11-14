"""Independent implementations exposed through the usual optimizer step API."""

from descent_algorithms.minibatch_sgd import MiniBatchSGD
from descent_algorithms.sfn import SFN
from descent_algorithms.stochastic_gd import SGD

__all__ = ["SGD", "MiniBatchSGD", "SFN"]
