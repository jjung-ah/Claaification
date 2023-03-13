# Coding by BAEK(01153450@hyundai-autoever.com)

import glob
import numpy as np
from PIL import Image

import argparse
import os
import time
import torch
import torch.nn as nn
from hydra import initialize_config_dir, compose
from architecture.modeling.models import resnet
from utils.types import Dictconfigs, Tensor
from typing import List
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from architecture.data.transforms.build import data_transforms


def get_all_items(directory: str, extension=['jpg', 'png']) -> List:
    if not os.path.exists(directory):
        raise FileNotFoundError(f'There is no image or folder for {directory}.')

    if os.path.isfile(directory):
        return [directory]
    else:
        file_list = []
        for ext in extension:
            file_list += glob.glob(f'{directory}/**/*.{ext}', recursive=True)
        return list(set(file_list))


def read_img(img_dir: str) -> Tensor:
    img = Image.open(img_dir).convert('RGB')
    img = np.array(img)

    # np.array to torch.tensor
    img = torch.tensor(img, dtype=torch.float32)
    img = torch.transpose(torch.transpose(img, 1, 2), 0, 1)
    return img



class Inference():

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.CrossEntropyLoss()


    def running(self):
        input_size = (224, 224)
        # trans = data_transforms['val']
        model = resnet.modeltype(args.model_name)
        model.load_state_dict(torch.load(args.model_dir))
        # model.to(self.device)
        model.eval()

        image_list = get_all_items(args.test_dir)
        for img in image_list:
            img = read_img(img)
            im_tensor = torch.tensor(img)
            # im_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
            im_tensor = F.upsample(torch.unsqueeze(im_tensor, 0), input_size, mode="bilinear").type(torch.uint8)
            image = torch.divide(im_tensor, 255.0)
            image = normalize(image, [0.5], [1.0])
            output = model(image)
            _, predicted = output.max(1)
            print('predict: ', predicted.item())



def main(args) -> None:
    torch.multiprocessing.set_start_method('spawn')
    Inference.running(args)


if __name__ == '__main__':
    abs_config_dir = os.path.abspath('./configs')
    with initialize_config_dir(version_base=None, config_dir=abs_config_dir):
        cfg = compose(config_name='defaults.yaml')

    parser = argparse.ArgumentParser(description='Image Classification')
    parser.add_argument('--model_dir',
                        default='/mnt/hdd1/datasets/hyundai-steel-goro/datasets/04_results/3rd/resnet18_best.pth',
                        type=str, help='model path')
    parser.add_argument('--test_dir',
                        default='/mnt/hdd1/datasets/hyundai-steel-goro/datasets/02_inspected_datasets/test', type=str,
                        help='save test results')
    parser.add_argument("--model_name", default='resnet18', type=str, help='model name')

    args = parser.parse_args()
    main(args)
