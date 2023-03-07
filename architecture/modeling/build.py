# Coding by SUNN(01139138@hyundai-autoever.com)

import torch
import torch.nn as nn

from utils.types import Dictconfigs
from architecture.modeling.models import MODEL_REGISTRY


def build_model(configs: Dictconfigs):
    model_name = configs.meta_arch.trainer.model.name
    model = MODEL_REGISTRY.get(model_name)()

    # todo : 이게 여기 있어야 하는지 고민
    # > 추후에 모델 구조 정리하면서 모델 클래스 안으로 옮기면 좋을듯
    model_digit = configs.meta_arch.trainer.model.digit
    if model_digit == 'half':
        model.half()
        for layer in model.moduels():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()
    # if torch.cuda.is_available():
    #     return model.cuda()
    return model
