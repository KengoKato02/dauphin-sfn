# ==============================================================================
# Title:       MSGD Training
# File:        /src/training/msgd_training/trainer.py
# Author:      KATO Kengo
# Description: Train an MLP using minibatch SGD and classical momentum
# ==============================================================================

from torch.optim import SGD
from src.training.common import train_model
from torch.nn.functional import cross_entropy
from src.utils.decorators import handle_exceptions


@handle_exceptions
def train_msgd(model, dataset, config):
    """Create SGD and supply its batch update to the shared training loop."""
    optimizer = SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)

    @handle_exceptions
    def update_batch(images, labels):
        """Clear gradients, differentiate the batch loss and update parameters."""
        optimizer.zero_grad(set_to_none=True)
        loss = cross_entropy(model(images), labels)
        loss.backward()
        optimizer.step()

    return train_model(model, dataset, update_batch, config)
