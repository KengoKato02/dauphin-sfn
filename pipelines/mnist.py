# ==============================================================================
# Title:       MNIST Data Preparation Pipeline
# File:        /pipelines/mnist.py
# Author:      KATO Kengo
# Description: Pipeline for importing, preparing and exporting MNIST
# ==============================================================================

from torchvision.datasets import MNIST

"""Import both training and testing datasets 
Understand the meaning behind each parameters:
    - root="data": store or find the files in the project data directory.
    - train=True: open the official training portion.
    - train=False: open the official test portion.
    - download=True: download missing files. Existing downloads are reused.
"""

train_data = MNIST(
    root="data",
    train=True,
    download=True
)

test_data = MNIST(
    root="data",
    train=False,
    download=True
)

image, label = train_data[0]

print("Training examples:", len(train_data))
print("Test examples:", len(test_data))

print("Image type:", type(image))
print("Image size:", image.size)

print("Label:", label)
print("Label type:", type(label))
