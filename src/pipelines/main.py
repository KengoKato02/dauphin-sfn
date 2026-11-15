# ==============================================================================
# Title:       Data Preparation Pipelines
# File:        /pipelines/main.py
# Author:      KATO Kengo
# Description: Run MNIST and CIFAR-10 processing pipelines in parallel
# ==============================================================================

from concurrent.futures import ThreadPoolExecutor

from src.pipelines.cifar import run_cifar_pipeline
from src.pipelines.mnist import run_mnist_pipeline


def run_pipelines():
    """Run both pipelines in parallel and return their results as (mnist, cifar)."""
    with ThreadPoolExecutor(max_workers=2) as executor:
        mnist = executor.submit(run_mnist_pipeline)
        cifar = executor.submit(run_cifar_pipeline)
        return mnist.result(), cifar.result()
