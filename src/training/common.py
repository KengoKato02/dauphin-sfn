# ==============================================================================
# Title:       Shared Training Loop
# File:        /src/training/common.py
# Author:      KATO Kengo
# Description: Batch data, repeat epochs and record measurements for either optimizer
# ==============================================================================

from torch.utils.data import DataLoader
from src.training.metrics import measure_model
from src.utils.decorators import handle_exceptions

import torch


@handle_exceptions
def train_model(model, dataset, update_batch, config):
    """Run the shared epoch loop using the optimizer's batch-update function.

    update_batch receives images and labels and performs one parameter update.
    Record epoch zero and one measurement after each complete training pass.
    """
    batch_size = len(dataset) if config.batch_size is None else config.batch_size
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False,
        generator=torch.Generator().manual_seed(config.seed), num_workers=0,
    )
    history = [{"epoch": 0, **measure_model(
        model, dataset, config.curvature_samples, config.max_parameters
    )}]
    for epoch in range(1, config.epochs + 1):
        model.train()
        for images, labels in loader:
            update_batch(images, labels)

        metrics = measure_model(model, dataset, config.curvature_samples, config.max_parameters)
        if not torch.isfinite(torch.tensor(metrics["loss"])):
            raise FloatingPointError("Training loss became non-finite.")
        history.append({"epoch": epoch, **metrics})
        print(f"Epoch {epoch}: loss={metrics['loss']:.4f}, "
              f"train error={metrics['train_error_pct']:.2f}%", flush=True)
    return history
