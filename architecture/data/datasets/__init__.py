# Coding by BAEK(01153450@hyundai-autoever.com)

from fvcore.common.registry import Registry

DATASETS_REGISTRY = Registry("DATASETS")
DATASETS_REGISTRY.__doc__ = "Registry for Data-Sets."


# datasets.
from .goro import GoroDataset

__all__ = [
    'GoroDataset'
]
