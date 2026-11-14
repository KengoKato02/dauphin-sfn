# ==============================================================================
# Title:       Tensor Usage Reference Document
# File:        tensors.py
# Author:      KATO Kengo
# Description: File used for basic reference on Tensor usage from PyTorch
# ==============================================================================

import torch

scalar = torch.tensor(1.0)          # Scalar = 0-Dimensional Tensor
vector = torch.tensor([1, 0])       # Vector = 1-Dimensional Tensor
matrix = torch.tensor([1.0, 2.0],   # Matrix = 2-Dimensional Tensor
                      [3.0, 4.0])

zeros = torch.zeros(2, 3)           # 2 rows, 3 columns, all zeros
ones = torch.ones(2, 3)             # All ones

# Random values from a standard normal distribution
random = torch.randn(2, 3)
sequence = torch.arange(5)          # tensor([0, 1, 2, 3, 4])

# ============================ Shapes ============================ #

print(matrix.shape)                 # torch.Size([2, 2])
print(matrix.dtype)                 # torch.float32
print(matrix.device)                # cpu
print(matrix.ndim)                  # 2 dimensions
print(matrix.numel())               # 4 total elements

print(scalar.item())                # Python float: 2.0

# ============================ Value Selections ============================ #

matrix[0, 1]                        # First row, second column: tensor(2.)
matrix[0]                           # First row: tensor([1., 2.])
matrix[:, 1]                        # Second column: tensor([2., 4.])

matrix.reshape(4)                   # Shape (4,)
matrix.reshape(1, -1)               # Shape (1, 4)
matrix.flatten()                    # Shape (4,)
matrix.T                            # Transpose: swap rows and columns

vector.unsqueeze(0)                 # Shape (1, 3): add a dimension
vector.unsqueeze(1)                 # Shape (3, 1): add a dimension
