# Coding by SUNN(01139138@hyundai-autoever.com)

from fvcore.common.registry import Registry

EVALUATOR_REGISTRY = Registry("EVALUATOR")
EVALUATOR_REGISTRY.__doc__ = "Registry for Evaluator."


# evaluators.
from .basics import F1MaeTorch

__all__ = [
    'F1MaeTorch',
]
