# Coding by BAEK(01153450@hyundai-autoever.com)

import os
import torch
import random
import numpy as np
import argparse
from hydra import initialize_config_dir, compose

from utils.types import Dictconfigs
from architecture.engine import training
# from architecture.data.datasets import datasets
from architecture.data.build import build_dataloader


def fix_seed(seed: int) -> None:
    # for control randomness.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setting_cfgs(configs: Dictconfigs) -> Dictconfigs:
    fix_seed(seed=configs.mode.seed)

    if torch.cuda.is_available():
        configs.mode.mp.device = 'cuda'
        # todo : gpu 수량을 전부 사용하고 싶지 않은 경우는 ?
        configs.mode.mp.num_gpus = torch.cuda.device_count()
    else:
        configs.mode.mp.device = 'cpu'
        configs.mode.mp.num_gpus = 0

    # make output-directory.
    os.makedirs(configs.mode.output_dir, exist_ok=True)

    return configs


def main(configs: Dictconfigs) -> None:
    configs = setting_cfgs(configs)

    # 데이터 불러오기
    trainloader = build_dataloader(configs, train=True)
    valloader = build_dataloader(configs, train=False)
    print('Completed loading datasets')


    # 모델 불러오기
    learning = training.SupervisedLearning(trainloader, valloader, args.model_name, args.pretrained)

    if args.train == 'train':
        learning.train(args.epoch, args.lr, args.l2)
    else:
        train_acc = learning.eval(trainloader)
        val_acc = learning.eval(valloader)
        print(f' Train Accuracy: {train_acc}, Test Accuracy: {val_acc}')



if __name__ == '__main__':
    abs_config_dir = os.path.abspath('./configs')
    with initialize_config_dir(version_base=None, config_dir=abs_config_dir):
        cfg = compose(config_name='defaults.yaml')

    parser = argparse.ArgumentParser(description='Image Classification')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--epoch', default=300, type=int, help='training epoch')
    parser.add_argument('--lr', default=0.00001, type=float, help='learning rate')
    parser.add_argument('--l2', default=0.000001, type=float, help='weight decay')
    parser.add_argument('--model_name', default='resnet18', type=str, help='model name')
    parser.add_argument('--pretrained', default=None, type=str, help='model path')
    parser.add_argument('--train', default='train', type=str, help='train and eval')
    args = parser.parse_args()

    # todo : add launch(=multi-gpu)
    main(configs=cfg)