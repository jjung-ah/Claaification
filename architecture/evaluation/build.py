# Coding by SUNN(01139138@hyundai-autoever.com)

from utils.types import Dictconfigs
from architecture.evaluation import EVALUATOR_REGISTRY


def build_evaluator(configs: Dictconfigs):
    evaluator_config = configs.solver.evaluator
    evaluator_name = evaluator_config.name
    return EVALUATOR_REGISTRY.get(evaluator_name)(evaluator_config)
