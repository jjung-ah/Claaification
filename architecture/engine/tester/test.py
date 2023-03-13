# Coding by BAEK(01153450@hyundai-autoever.com)

import argparse
import os
import time
import torch
import torch.nn as nn
from hydra import initialize_config_dir, compose
from collections import defaultdict
from torch.utils.data import DataLoader
from architecture.modeling.models import resnet
from utils.types import Dictconfigs
from architecture.data.datasets.goro import GoroDataset
from architecture.data.utils import load_yaml, get_all_items
from architecture.data.transforms.build import build_train_transforms, build_val_transforms


def build_dataloader(configs: Dictconfigs, train: bool):

    data_configs = configs.data

    # (1) build transforms.
    if train == True:
        transform = build_train_transforms(mode='train')
    else:
        transform = build_val_transforms(mode='val')

    # (2) get file-list.
    # todo : 추후에 다른 함수로 뺄 수 있도록
    data_list = data_configs.datasets.test
    datasets_dict = defaultdict(dict)
    idx = 0
    for data_info in data_list:
        info = load_yaml(config_path=data_info)
        set_name = info['name']
        img_list = get_all_items(info['img_dir'], img_ext=info['img_ext'])

        # 여기서 gt를 저장할지, dataset에서 gt를 부를지 고민
        for img_dir in img_list:
            datasets_dict[idx] = dict(
                set=set_name,
                img_dir=img_dir,
            )
            idx += 1

    # (3) call datasets.
    datasets = GoroDataset(datasets_dict, transform=transform)

    # (4) call data-loader.
    batch_size = configs.mode.solver.batch_train if train else configs.mode.solver.batch_val
    num_workers = 8 if batch_size > 8 else 4 if batch_size > 4 else 1
    # num_workers = 0
    dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, drop_last=True)  # shuffle=True : train
    return dataloader


class Test():

    def __init__(self, testloader):

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.testloader = testloader
        self.criterion = nn.CrossEntropyLoss()


    def test(self):
        test_loss = 0
        correct = 0
        total = 0

        model = resnet.modeltype(args.model_name)
        model.load_state_dict(torch.load(args.model_dir))
        model.to(self.device)
        model.eval()

        with torch.no_grad():
            for batch_idx, (input, gt) in enumerate(self.testloader):
                input, gt = input.to(self.device), gt.to(self.device)
                output = model(input)
                loss = self.criterion(output, gt.long())

                test_loss += loss.data.cpu().numpy()
                _, predicted = output.max(1)
                print(predicted)
                total += gt.size(0)
                correct += predicted.eq(gt).sum().item()

            epoch_loss = test_loss / len(self.testloader)
            epoch_acc = correct / total
            print('test loss: {:.4f}, {:.4f}'.format(epoch_loss, epoch_acc * 100))



def main(configs: Dictconfigs) -> None:
    torch.multiprocessing.set_start_method('spawn')

    # 데이터 불러오기
    testloader = build_dataloader(configs, train=False)
    print('Completed loading test datasets')
    test = Test(testloader)
    test.test()



if __name__ == '__main__':
    abs_config_dir = os.path.abspath('../../../configs')
    print(abs_config_dir)
    with initialize_config_dir(version_base=None, config_dir=abs_config_dir):
        cfg = compose(config_name='defaults.yaml')

    parser = argparse.ArgumentParser(description='Image Classification')
    parser.add_argument('--model_dir', default='/mnt/hdd1/datasets/hyundai-steel-goro/datasets/04_results/3rd/resnet18_best.pth', type=str, help='model path')
    parser.add_argument('--test_dir', default='/mnt/hdd1/datasets/hyundai-steel-goro/datasets/02_inspected_datasets/test', type=str, help='save test results')
    parser.add_argument("--model_name", default='resnet18', type=str, help='model name')

    args = parser.parse_args()

    main(cfg)


