# Coding by SUNN(01139138@hyundai-autoever.com)
# todo : 여러 함수를 한 번에 사용하는 경우는 어떻게 할 것인지 > 함수별로 분리할지 고민

import torch.nn as nn
import torch.nn.functional as F

from . import CRITERION_REGISTRY
from utils.types import Dictconfigs


@CRITERION_REGISTRY.register()
class MultiLossFusion(object):
    def __init__(self, configs: Dictconfigs):
        self.bce_loss = nn.BCELoss(size_average=configs.parameters.size_average)

    def __call__(self, preds, target):
        loss0, loss = 0.0, 0.0

        for i in range(0, len(preds)):
            if preds[i].shape[2] != target.shape[2] or preds[i].shape[3] != target.shape[3]:
                tmp_target = F.interpolate(target, size=preds[i].size()[2:], mode='bilinear', align_corners=True)
                loss = loss + self.bce_loss(preds[i], tmp_target)
            else:
                loss = loss + self.bce_loss(preds[i], target)
            if i == 0:
                loss0 = loss

        return loss0, loss
