# Coding by SUNN(01139138@hyundai-autoever.com)

from fvcore.common.registry import Registry

OPTIMIZER_REGISTRY = Registry("OPTIMIZER")
OPTIMIZER_REGISTRY.__doc__ = "Registry for Optimizer."


# optimizers.
from .functions import Adam

__all__ = [
    'Adam'
]
