# Coding by SUNN(01139138@hyundai-autoever.com)

import os
import glob
import yaml
from typing import List, Dict


def load_yaml(config_path: str) -> Dict:
    with open(config_path, encoding='utf-8') as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)
    return configs


def get_all_items(directory: str, img_ext: List) -> List:
    if not os.path.exists(directory):
        raise FileNotFoundError(f'There is no folder for {directory}.')

    file_list = []
    for ext in img_ext:
        file_list += glob.glob(f'{directory}/**/*.{ext}', recursive=True)
    return sorted(list(set(file_list)))


def get_label_items(directory: str, img_ext: List) -> List:
    if not os.path.exists(directory):
        raise FileNotFoundError(f'There is no folder for {directory}.')

    file_list, gt_list = [], []
    for ext in img_ext:
        file_list += glob.glob(f'{directory}/**/*.{ext}', recursive=True)
    for file in sorted(list(set(file_list))):
        gt = os.path.dirname(file).split(os.path.sep)[-1]
        gt_list += glob.glob(gt)
    return gt_list


# def get_label_items(directory: str) -> List:
#     if not os.path.exists(directory):
#         raise FileNotFoundError(f'There is no folder for {directory}.')
#
#     file_list = []
#     sub = [os.path.dirname(i) for i in subfolders if i.find(image) != -1]
#     for ext in img_ext:
#         file_list += glob.glob(f'{directory}/**/*.{ext}', recursive=True)
#     return sorted(list(set(file_list)))

