# Coding by SUNN(01139138@hyundai-autoever.com)

import os
import torch
import random
import numpy as np
from hydra import initialize_config_dir, compose

from utils.types import Dictconfigs
from architecture.engine.build import build_trainer


def fix_seed(seed: int) -> None:
    # for control randomness.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setting_cfgs(configs: Dictconfigs) -> Dictconfigs:
    fix_seed(seed=configs.mode.seed)

    if torch.cuda.is_available():
        configs.mode.mp.device = 'cuda'
        # todo : gpu 수량을 전부 사용하고 싶지 않은 경우는 ?
        configs.mode.mp.num_gpus = torch.cuda.device_count()
    else:
        configs.mode.mp.device = 'cpu'
        configs.mode.mp.num_gpus = 0

    # make output-directory.
    os.makedirs(configs.mode.output_dir, exist_ok=True)

    return configs


def main(configs: Dictconfigs) -> None:
    configs = setting_cfgs(configs)

    trainer = build_trainer(configs)
    trainer.train()


if __name__ == '__main__':
    abs_config_dir = os.path.abspath('./configs')
    with initialize_config_dir(version_base=None, config_dir=abs_config_dir):
        cfg = compose(config_name='defaults.yaml')

    # todo : add launch(=multi-gpu)
    main(configs=cfg)
