# ==============================================================================
# Title:       Model Training
# File:        /src/training/trainer.py
# Author:      KATO Kengo
# Description: Train a model over batches and record measurements after each epoch
# ==============================================================================

from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy
from src.training.metrics import measure_model
from src.utils.decorators import handle_exceptions
from src.algorithms.damped_newton import DampedNewton

import torch


@handle_exceptions
def train_model(model, dataset, optimizer, epochs=3, batch_size=32,
                curvature_samples=128, seed=42, max_parameters=2000):
    """Train with plain SGD on minibatches, or the exact damped Newton reference.

    Record epoch zero and one measurement after each complete training pass.
    A fresh loader generator gives comparison runs the same minibatch order.
    """
    if epochs < 0 or batch_size < 1 or curvature_samples < 0:
        raise ValueError("epochs and curvature_samples must be nonnegative; batch_size positive.")
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False,
        generator=torch.Generator().manual_seed(seed), num_workers=0,
    )
    history = [{"epoch": 0, **measure_model(model, dataset, curvature_samples, max_parameters)}]
    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in loader:
            if isinstance(optimizer, DampedNewton):
                optimizer.step(images, labels)
            else:
                optimizer.zero_grad(set_to_none=True)
                loss = cross_entropy(model(images), labels)
                loss.backward()
                optimizer.step()

        metrics = measure_model(model, dataset, curvature_samples, max_parameters)
        if not torch.isfinite(torch.tensor(metrics["loss"])):
            raise FloatingPointError("Training loss became non-finite.")
        history.append({"epoch": epoch, **metrics})
        print(f"Epoch {epoch}: loss={metrics['loss']:.4f}, "
              f"train error={metrics['train_error_pct']:.2f}%", flush=True)
    return history
