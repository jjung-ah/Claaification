# Coding by BAEK(01153450@hyundai-autoever.com)
import os.path

import torch
import numpy as np
from typing import Dict
from skimage import io
from PIL import Image

from torchvision import transforms as T
from . import DATASETS_REGISTRY
from utils.types import Tensor


def read_img(img_dir: str) -> Tensor:
    img = io.imread(img_dir)
    # img = Image.open(img_dir).convert('RGB')

    if len(img.shape) < 3:
        img = img[:, :, np.newaxis]
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

    # np.array to torch.tensor
    img = torch.tensor(img, dtype=torch.float32)
    img = torch.transpose(torch.transpose(img, 1, 2), 0, 1)
    return img


def read_gt(img_dir: str) -> str:
    gt = os.path.dirname(img_dir).split(os.path.sep)[-1]
    return gt


@DATASETS_REGISTRY.register()
class GoroDataset(object):
    '''
    Basic Dataset-Class of this repo.
    '''
    def __init__(self, datasets: Dict, transform=None):
        self.datasets = datasets
        self.transform = transform

    def __len__(self) -> int:
        return len(self.datasets)

    def __getitem__(self, idx: int) -> Dict:
        '''
        return Dict : set(str), img_dir(str), img(tensor), gt(str)
        '''
        data = self.datasets[idx]

        img = read_img(img_dir=data['img_dir'])
        # img = Image.open(data['img_dir']).convert('RGB')
        gt = read_gt(img_dir=data['img_dir'])

        # apply transform.
        img = torch.divide(img, 255.0)
        if self.transform:
            # # img = Image.fromarray(np.asarray(img).astype(np.float32))
            # img = np.array(img).astype(np.uint8)
            # # img = torch.tensor(img).squeeze().cpu().numpy()
            # img = torch.tensor(img)
            img = torch.tensor(img).squeeze()
            img = self.transform(img)
        data.update(dict(img=img, gt=gt))
        return data


