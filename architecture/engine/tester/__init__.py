# Coding by BAEK(01153450@hyundai-autoever.com)

from fvcore.common.registry import Registry

TESTER_REGISTRY = Registry("TESTER")
TESTER_REGISTRY.__doc__ = "Registry for Tester."


# testers.
from .default import DefaultTester

__all__ = [
    'DefaultTester'
]
