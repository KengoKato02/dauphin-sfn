# ==============================================================================
# Title:       Main Entry Point
# File:        main.py
# Author:      KATO Kengo
# Description: Personal playground for direct execution
# ==============================================================================

from torchvision.datasets import MNIST
from src.algorithms.batch_gd import gradient_descent_singleton
from src.utils.visuals.mnist import plot_mnist_resize_comparison
from src.pipelines.load.loader import load_dataset
from src.pipelines.transform.transformer import demonstrate_image_transform

import torch

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

######################
# Image Demonstration
######################
train_data, test_data = load_dataset(MNIST)
image, label = train_data[0]
print("LABEL: ", label)
original_tensor, resized_tensor = demonstrate_image_transform(image)
plot_mnist_resize_comparison(original_tensor, resized_tensor)
