# Coding by BAEK(01153450@hyundai-autoever.com)

from collections import defaultdict
from torch.utils.data import DataLoader

from utils.types import Dictconfigs
from architecture.data.datasets import DATASETS_REGISTRY
from .utils import load_yaml, get_all_items  #, get_label_items
from .transforms.build import build_train_transforms, build_val_transforms


def build_dataloader(configs: Dictconfigs, train: bool):
    data_configs = configs.data

    # (1) build transforms.
    if train == True:
        transform = build_train_transforms(mode='train')
    else:
        transform = build_val_transforms(mode='val')

    # (2) get file-list.
    # todo : 추후에 다른 함수로 뺄 수 있도록
    data_list = data_configs.datasets.train if train else data_configs.datasets.val
    datasets_dict = defaultdict(dict)
    idx = 0
    for data_info in data_list:
        info = load_yaml(config_path=data_info)
        set_name = info['name']
        img_list = get_all_items(info['img_dir'], img_ext=info['img_ext'])
        # gt_list = get_label_items(info['img_dir'])
        # assert len(img_list) == len(gt_list)

        # 여기서 gt를 저장할지, dataset에서 gt를 부를지 고민
        for img_dir in img_list:
            datasets_dict[idx] = dict(
                set=set_name,
                img_dir=img_dir,
            )
            idx += 1

    # (3) call datasets.
    factory_name = data_configs.datasets.factory_name
    datasets = DATASETS_REGISTRY.get(factory_name)(datasets_dict, transform=transform)
    # val_datasets = DATASETS_REGISTRY.get(factory_name)(datasets_dict, transform=val_transform)

    # (4) call data-loader.
    batch_size = configs.mode.solver.batch_train if train else configs.mode.solver.batch_val
    num_workers = 8 if batch_size > 8 else 4 if batch_size > 4 else 1
    dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers)  # shuffle=True : train
    # valloader = DataLoader(val_datasets, batch_size=batch_size, shuffle=False,
    #                         num_workers=num_workers)  # shuffle=False : val
    return dataloader
