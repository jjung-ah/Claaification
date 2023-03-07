# Coding by SUNN(01139138@hyundai-autoever.com)

import torch.optim as optim

# from . import OPTIMIZER_REGISTRY
# from utils.types import Dictconfigs
#
#
# @OPTIMIZER_REGISTRY.register()
# def Adam(configs: Dictconfigs, model):
#     return optim.Adam(
#         model.parameters(),
#         lr=configs.parameters.lr,
#         betas=tuple(configs.parameters.betas),
#         eps=configs.parameters.eps,
#         weight_decay=configs.parameters.weight_decay
#     )


def Adam(model, lr, weight_decay):
    return optim.Adam(
        model.parameters(),
        lr=lr,
        # betas=tuple(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay
    )