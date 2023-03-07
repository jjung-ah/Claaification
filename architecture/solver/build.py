# Coding by SUNN(01139138@hyundai-autoever.com)

from utils.types import Dictconfigs
from architecture.solver.criterion import CRITERION_REGISTRY
from architecture.solver.optimizer import OPTIMIZER_REGISTRY


def build_criterion(configs: Dictconfigs):
    criterion_config = configs.solver.criterion
    criterion_name = criterion_config.name
    return CRITERION_REGISTRY.get(criterion_name)(criterion_config)


def build_optimizer(configs: Dictconfigs, model):
    optimizer_config = configs.solver.optimizer
    optimizer_name = optimizer_config.name
    return OPTIMIZER_REGISTRY.get(optimizer_name)(optimizer_config, model)
