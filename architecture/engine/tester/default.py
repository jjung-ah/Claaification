# # Coding by SUNN(01139138@hyundai-autoever.com)
#
# import gc
# import torch
# from utils.types import Dictconfigs
# from architecture.engine.test_loop import TesterBase
# from . import TESTER_REGISTRY
#
# from architecture.modeling.build import build_model
# from architecture.solver.build import build_criterion, build_optimizer
# from architecture.evaluation.build import build_evaluator
# from architecture.data.build import build_dataloader
#
#
# @TESTER_REGISTRY.register()
# class DefaultTester(TesterBase):
#     '''
#     Default-class for tester.
#     '''
#     def __init__(self, configs: Dictconfigs):
#         super(DefaultTester, self).__init__()
#
#         # build for trainer.
#         self.test_loader = self.build_dataloader(configs, train=False)
#
#         self.model = build_model(configs)
#         self.criterion = build_criterion(configs)
#         self.optimizer = build_optimizer(configs, self.model)
#
#     def before_test(self):
#         # todo : add checkpointer
#         pass
#
#     def before_step(self):
#         pass
#
#     def run_step(self):
#         self.model.eval()
#         test_loss = 0
#         correct = 0
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#         with torch.no_grad():
#             data, target = next(iter(self.test_loader))
#             data, target = data.to(device), target.to(device)
#             output = self.model(data)
#             test_loss += self.criterion()  # todo: change def
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()
#
#         test_loss /= len(self.test_loader.dataset)
#
#         print("\nTest set: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
#             test_loss, correct, len(self.test_loader.dataset), 100 * correct / len(self.test_loader.dataset)
#         ))
#
#
#
#     def after_step(self):
#         # todo : add save weights
#         # todo : add logger
#         pass
#
#     def after_test(self):
#         # todo : add logger
#         pass
#
#     @classmethod
#     def build_model(cls):
#         # It now calls :func: 'architecture.modeling.build_model'.
#         return build_model()
#
#     @classmethod
#     def build_criterion(cls):
#         # It now calls :func: 'architecture.solver.build_criterion'.
#         return build_criterion()
#
#     @classmethod
#     def build_optimizer(cls):
#         # It now calls :func: 'architecture.solver.build_optimizer'.
#         return build_optimizer()
#
#     @classmethod
#     def build_evaluator(cls):
#         # It now calls :func: 'architecture.evaluation.build_evaluator'.
#         return build_evaluator()
#
#     @classmethod
#     def build_dataloader(cls, configs: Dictconfigs, train: bool):
#         # It now calls :func: 'architecture.data.build_dataloader'.
#         return build_dataloader(configs, train)



'''
from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet


# -- fixtures -------------------------------------------------------------------------------------

@pytest.fixture(scope='module', params=[x for x in range(4)])
def model(request):
    return 'efficientnet-b{}'.format(request.param)


@pytest.fixture(scope='module', params=[True, False])
def pretrained(request):
    return request.param


@pytest.fixture(scope='function')
def net(model, pretrained):
    return EfficientNet.from_pretrained(model) if pretrained else EfficientNet.from_name(model)


# -- tests ----------------------------------------------------------------------------------------

@pytest.mark.parametrize('img_size', [224, 256, 512])
def test_forward(net, img_size):
    """Test `.forward()` doesn't throw an error"""
    data = torch.zeros((1, 3, img_size, img_size))
    output = net(data)
    assert not torch.isnan(output).any()


def test_dropout_training(net):
    """Test dropout `.training` is set by `.train()` on parent `nn.module`"""
    net.train()
    assert net._dropout.training == True


def test_dropout_eval(net):
    """Test dropout `.training` is set by `.eval()` on parent `nn.module`"""
    net.eval()
    assert net._dropout.training == False


def test_dropout_update(net):
    """Test dropout `.training` is updated by `.train()` and `.eval()` on parent `nn.module`"""
    net.train()
    assert net._dropout.training == True
    net.eval()
    assert net._dropout.training == False
    net.train()
    assert net._dropout.training == True
    net.eval()
    assert net._dropout.training == False


@pytest.mark.parametrize('img_size', [224, 256, 512])
def test_modify_dropout(net, img_size):
    """Test ability to modify dropout and fc modules of network"""
    dropout = nn.Sequential(OrderedDict([
        ('_bn2', nn.BatchNorm1d(net._bn1.num_features)),
        ('_drop1', nn.Dropout(p=net._global_params.dropout_rate)),
        ('_linear1', nn.Linear(net._bn1.num_features, 512)),
        ('_relu', nn.ReLU()),
        ('_bn3', nn.BatchNorm1d(512)),
        ('_drop2', nn.Dropout(p=net._global_params.dropout_rate / 2))
    ]))
    fc = nn.Linear(512, net._global_params.num_classes)

    net._dropout = dropout
    net._fc = fc

    data = torch.zeros((2, 3, img_size, img_size))
    output = net(data)
    assert not torch.isnan(output).any()


@pytest.mark.parametrize('img_size', [224, 256, 512])
def test_modify_pool(net, img_size):
    """Test ability to modify pooling module of network"""

    class AdaptiveMaxAvgPool(nn.Module):

        def __init__(self):
            super().__init__()
            self.ada_avgpool = nn.AdaptiveAvgPool2d(1)
            self.ada_maxpool = nn.AdaptiveMaxPool2d(1)

        def forward(self, x):
            avg_x = self.ada_avgpool(x)
            max_x = self.ada_maxpool(x)
            x = torch.cat((avg_x, max_x), dim=1)
            return x

    avg_pooling = AdaptiveMaxAvgPool()
    fc = nn.Linear(net._fc.in_features * 2, net._global_params.num_classes)

    net._avg_pooling = avg_pooling
    net._fc = fc

    data = torch.zeros((2, 3, img_size, img_size))
    output = net(data)
    assert not torch.isnan(output).any()


@pytest.mark.parametrize('img_size', [224, 256, 512])
def test_extract_endpoints(net, img_size):
    """Test `.extract_endpoints()` doesn't throw an error"""
    data = torch.zeros((1, 3, img_size, img_size))
    endpoints = net.extract_endpoints(data)
    assert not torch.isnan(endpoints['reduction_1']).any()
    assert not torch.isnan(endpoints['reduction_2']).any()
    assert not torch.isnan(endpoints['reduction_3']).any()
    assert not torch.isnan(endpoints['reduction_4']).any()
    assert not torch.isnan(endpoints['reduction_5']).any()
    assert endpoints['reduction_1'].size(2) == img_size // 2
    assert endpoints['reduction_2'].size(2) == img_size // 4
    assert endpoints['reduction_3'].size(2) == img_size // 8
    assert endpoints['reduction_4'].size(2) == img_size // 16
    assert endpoints['reduction_5'].size(2) == img_size // 32

'''
