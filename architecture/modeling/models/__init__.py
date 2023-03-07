# Coding by BAEK(01153450@hyundai-autoever.com)

from fvcore.common.registry import Registry

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = "Registry for Model."


# models.
from .resnet import ResNet, BasicBlock, Bottleneck  # , ResidualBlock

__all__ = [
    'ResNet',
    # 'ResidualBlock',
    'BasicBlock',
    'Bottleneck'
]
