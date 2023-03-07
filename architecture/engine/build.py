# Coding by SUNN(01139138@hyundai-autoever.com)

from utils.types import Dictconfigs
from architecture.engine.trainer import TRAINER_REGISTRY
from architecture.engine.tester import TESTER_REGISTRY


def build_trainer(configs: Dictconfigs):
    factory_name = configs.meta_arch.trainer.factory_name
    trainer = TRAINER_REGISTRY.get(factory_name)(configs)
    return trainer

def build_tester(configs: Dictconfigs):
    factory_name = configs.meta_arch.tester.factory_name
    tester = TESTER_REGISTRY.get(factory_name)(configs)
    return tester