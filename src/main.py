# ==============================================================================
# Title:       Main Entry Point
# File:        main.py
# Author:      KATO Kengo
# Description: Main file for orchestration of functions
# ==============================================================================

from src.algorithms.batch_gd import gradient_descent_singleton

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

# # Use for single computation (Hand confirmation)
# batch_gd = gradient_descent_singleton(theta_init, loss, learning_rate)
# print(batch_gd.shape)
# print(batch_gd)

# PyTorch Backwards Propagation
loss.backward()

with torch.no_grad():
    theta_new = theta_init - learning_rate * theta_init.grad

theta_new.requires_grad_(True)

print("=========================")
print("new w:", theta_new[0].item())  # Approximately 2.2
print("new b:", theta_new[1].item())  # Approximately 0.6
print("theta new:", theta_new)

# Ensure to clear gradients
w.grad = None
b.grad = None 
