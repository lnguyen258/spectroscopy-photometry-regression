from torch import optim
from torch.optim import lr_scheduler


OPTIM_REGISTRY = {
    'adam': optim.Adam,
    'adamw': optim.AdamW,
    'sgd': optim.SGD,
    # Add more optimizers here as needed
}

SCHEDULER_REGISTRY = {
    'exponential_lr': lr_scheduler.ExponentialLR,
    'reduce_lr_on_plateau': lr_scheduler.ReduceLROnPlateau,
    # Add more schedulers here as needed
}
