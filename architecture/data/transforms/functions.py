# Coding by BAEK(01153450@hyundai-autoever.com)
# todo : to_tensor 추가
# todo : 어떻게 하면 parameter 들을 깔끔 + 통일 되게 가져올 수 있을까 고민

import torch
import random
import numpy as np
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

# from . import TRANSFORMS_REGISTRY
from utils.types import Dictconfigs
from torchvision import transforms


class Compose(object):
    """
        call 로 전달받은 image 와 mask에 동시에 같은 augmentation function 적용할 수 있게 변경
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image


class ToTensor(object):
    """
        기본 ToTensor 함수에, call function parameters 만 변경
    """
    def __call__(self, image):
        return F.to_tensor(image)

class ToPILImage(object):
    """
        기본 ToPILIMage 함수에, call function parameters 만 변경
    """
    def __init__(self, mode=None):
        self.mode = mode
    def __call__(self, image):
        return F.to_pil_image(image)


class Normalize(object):
    def __init__(self):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, image):
        image = normalize(image, self.mean, self.std)
        return image

'''
# TEST
from fvcore.common.registry import Registry
TRANSFORMS_REGISTRY = Registry("TRANSFORMS")


class Compose(object):
    """
        call 로 전달받은 image 와 mask에 동시에 같은 augmentation function 적용할 수 있게 변경
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image


class ToTensor(object):
    """
        기본 ToTensor 함수에, call function parameters 만 변경
    """
    def __call__(self, image, mask):
        return F.to_tensor(image), F.to_tensor(mask)


class ToPILImage(object):
    """
        기본 ToPILIMage 함수에, call function parameters 만 변경
    """
    def __init__(self, mode=None):
        self.mode = mode
    def __call__(self, image, mask):
        return F.to_pil_image(image), F.to_pil_image(mask)



@TRANSFORMS_REGISTRY.register()
class Resize(object):
    """
        기본 Resize 함수에, call function parameters 만 변경
    """
    def __init__(self, cfg):
        self.size = cfg.training.transforms.resize.shape
        self.mode = cfg.training.transforms.resize.mode
        self.interpolation = _interpolation_modes_from_int(self.mode)

    def __call__(self, image, mask):
        return F.resize(image, (self.size, self.size), self.interpolation), F.resize(mask, (self.size, self.size), self.interpolation)



@TRANSFORMS_REGISTRY.register()
class RandomVerticalFlip(object):
    """
        기본 RandomVerticalFlip 함수에, call function parameters 만 변경
    """
    def __init__(self, cfg):
        self.ratio = cfg.training.transforms.random_vertical_flip.probability

    def __call__(self, image, mask):
        if random.random() > self.ratio:
            image = F.vflip(image)
            mask = F.vflip(mask)
        return image, mask

@TRANSFORMS_REGISTRY.register()
class RandomHorizontalFlip(object):
    def __init__(self, cfg):
        self.ratio = cfg.training.transforms.random_horizontal_flip.probability

    def __call__(self, image, mask):
        if random.random() > self.ratio:
            image = F.hflip(image)
            mask = F.hflip(mask)
        return image, mask

@TRANSFORMS_REGISTRY.register()
class RandomCrop(object):
    """
        only change call function parameters
    """

    def __init__(self, cfg):
        self.size = (cfg.training.transforms.resize.shape - cfg.training.transforms.random_crop.margin)

    def __call__(self, image, mask):

        top, left, height, width = T.RandomCrop.get_params(image, (self.size, self.size))

        image = F.crop(image, top, left, height, width)
        mask = F.crop(mask, top, left, height, width)

        return image, mask
'''




# @TRANSFORMS_REGISTRY.register()
# # originally from https://github.com/xuebinqin/DIS
# class GOSRandomHFlip(object):
#     def __init__(self, parameters: Dictconfigs):
#         self.prob = parameters.prob  # 0.5
#
#     def __call__(self, img, gt):
#         if random.random() >= self.prob:
#             img = torch.flip(img, dims=[2])
#             # gt = torch.flip(gt, dims=[2])
#         return img  #, gt
#
#
# @TRANSFORMS_REGISTRY.register()
# # originally from https://github.com/xuebinqin/DIS
# class GOSResize(object):
#     def __init__(self, parameters: Dictconfigs):
#         self.size = (parameters.width, parameters.height)
#
#     def __call__(self, img, gt):
#         img = torch.squeeze(F.upsample(torch.unsqueeze(img, 0), self.size, mode='bilinear'), dim=0)
#         # gt = torch.squeeze(F.upsample(torch.unsqueeze(gt, 0), self.size, mode='bilinear'), dim=0)
#         return img  # , gt
#
#
# @TRANSFORMS_REGISTRY.register()
# # originally from https://github.com/xuebinqin/DIS
# class GOSRandomCrop(object):
#     def __init__(self, parameters: Dictconfigs):
#         self.size = (parameters.weight, parameters.height)
#
#     def __call__(self, img, gt):
#         h, w = img.shape[1:]
#         new_w, new_h = self.size
#         top = np.random.randint(0, h - new_h)
#         left = np.random.randint(0, w - new_w)
#
#         img = img[:, top:top+new_h, left:left+new_w]
#         # gt = gt[:, top:top+new_h, left:left+new_w]
#         return img  # , gt
#
#
# @TRANSFORMS_REGISTRY.register()
# # originally from https://github.com/xuebinqin/DIS
# class GOSNormalize(object):
#     def __init__(self, parameters: Dictconfigs):
#         self.mean = parameters.mean  # [0.485, 0.456, 0.406]
#         self.std = parameters.std  # [0.229, 0.224, 0.225]
#
#     def __call__(self, img, gt):
#         img = normalize(img, self.mean, self.std)
#         return img  # , gt
