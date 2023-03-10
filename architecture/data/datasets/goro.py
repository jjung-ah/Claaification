# Coding by BAEK(01153450@hyundai-autoever.com)
import os.path

import torch
import numpy as np
from typing import Dict
from PIL import Image
from . import DATASETS_REGISTRY
from utils.types import Tensor


def read_img(img_dir: str) -> Tensor:
    img = Image.open(img_dir).convert('RGB')
    img = np.array(img)

    # np.array to torch.tensor
    img = torch.tensor(img, dtype=torch.float32)
    img = torch.transpose(torch.transpose(img, 1, 2), 0, 1)
    return img


def read_gt(img_dir: str) -> str:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    gt = os.path.dirname(img_dir).split(os.path.sep)[-2].split('_')[0]
    return torch.tensor(int(gt)).to(device)  # torch.tensor(gt)


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

    def __getitem__(self, idx: int):
        '''
        return Dict : set(str), img_dir(str), img(tensor), gt(str)
        '''
        data = self.datasets[idx]

        img = read_img(img_dir=data['img_dir'])
        gt = read_gt(img_dir=data['img_dir'])

        # apply transform.
        img = torch.divide(img, 255.0)
        if self.transform:
            # img = torch.tensor(img).squeeze()
            # img = torch.tensor(img).squeeze().requires_grad_(True)
            img = torch.tensor(img).squeeze().clone().detach()
            img = self.transform(img)
        # data.update(img=img, gt=gt)
        return img, gt  # data


