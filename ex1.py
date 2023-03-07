import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

#
# data_transforms = {
#     'train': transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#     'val': transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }
#
#
# data_dir = "/mnt/hdd1/datasets/hyundai-steel-goro/datasets/02_inspected_datasets"
# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
#                                           data_transforms[x])
#                   for x in ['train', 'val']}
# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=6,
#                                              shuffle=True, num_workers=4)
#               for x in ['train', 'val']}
# print(image_datasets['train'])
# inputs, classes = next(iter(dataloaders['train']))
# # print(inputs[1], classes[1])

path = '/mnt/hdd1/datasets/hyundai-steel-goro/datasets/02_inspected_datasets/train'
sub = [os.path.dirname(i) for i in path if i.find('image') != -1]
print(sub)