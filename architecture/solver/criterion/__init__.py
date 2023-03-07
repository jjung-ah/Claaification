# Coding by SUNN(01139138@hyundai-autoever.com)

from fvcore.common.registry import Registry

CRITERION_REGISTRY = Registry("CRITERION")
CRITERION_REGISTRY.__doc__ = "Registry for Criterion."


# criterions.
from .functions import MultiLossFusion

__all__ = [
    'MultiLossFusion',
]
