# Coding by SUNN(01139138@hyundai-autoever.com)

from fvcore.common.registry import Registry

TRAINER_REGISTRY = Registry("TRAINER")
TRAINER_REGISTRY.__doc__ = "Registry for Trainer."


# trainers.
from .default import DefaultTrainer

__all__ = [
    'DefaultTrainer'
]
