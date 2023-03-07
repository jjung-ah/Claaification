# Coding by BAEK(01153450@hyundai-autoever.com)

from typing import List, Dict
from torchvision import transforms
from architecture.data.transforms.functions import Compose # , ToTensor , Normalize
from utils.types import Dictconfigs
# from .functions import TRANSFORMS_REGISTRY


data_transforms = {
    'train': Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def build_train_transforms(mode: str):
    return data_transforms[mode]


def build_val_transforms(mode: str):
    return data_transforms[mode]


