import os
import glob
from typing import List, Dict

def get_label_items(directory: str) -> List:
    if not os.path.exists(directory):
        raise FileNotFoundError(f'There is no folder for {directory}.')

    file_list = []
    sub = [os.path.dirname(i) for i in subfolders if i.find(image) != -1]
    for ext in img_ext:
        file_list += glob.glob(f'{directory}/**/*.{ext}', recursive=True)
    return sorted(list(set(file_list)))



path = '/mnt/hdd1/datasets/hyundai-steel-goro/datasets/02_inspected_datasets/train/0_Normal/image'
# sub = [os.path.dirname(i) for i in subfolders if i.find(image) != -1]



print(path.find('image'))
