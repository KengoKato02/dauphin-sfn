# ==============================================================================
# Title:       Main Entry Point
# File:        main.py
# Author:      Kengo Reimers Kato
# Description: Main file for orchestration of functions
# ==============================================================================

from descent_algorithms.batch_gd import gradient_descent

import torch


# Use x=2, y=5, w=1, b=0 and learning rate 0.1.
# Calculate the prediction, loss, both derivatives, and one gradient-descent
# update by hand. Then represent w and b as tensors requiring gradients
# and check your calculation withPyTorch. Explain what changed after the update.

x = torch.tensor(2.0)
y = torch.tensor(5.0)

learning_rate = torch.tensor(0.1)

# Initial Training Parameters
theta_init = torch.tensor([1.0, 0.0],
                          requires_grad=True)
w = theta_init[0]
b = theta_init[1]

# Forward Pass Example
pred = w*x + b              # Prediction
loss = (pred - y) ** 2      # Loss function

# Use for single computation (Hand confirmation)
batch_gd = gradient_descent(theta_init, loss, learning_rate)
print(batch_gd.shape)
print(batch_gd)

# # Backwards Propogation
# loss.backward
