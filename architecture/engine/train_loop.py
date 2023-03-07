# Coding by SUNN(01139138@hyundai-autoever.com)

from abc import ABCMeta, abstractmethod


class TrainerBase(metaclass=ABCMeta):
    '''
    Base-class for iterative trainer.
        +. combine with hook-system.
    '''
    def __init__(self):
        self.epoch: int = 0
        self.start_epoch: int = 0
        self.max_epoch: int = 0

    def train(self, start_epoch: int, max_epoch: int):
        self.before_train()
        for epoch in range(start_epoch, max_epoch):
            self.before_step()
            self.run_step()
            self.after_step()
        self.after_train()

    @abstractmethod
    def before_train(self):
        pass

    @abstractmethod
    def before_step(self):
        pass

    @abstractmethod
    def run_step(self):
        pass

    @abstractmethod
    def after_step(self):
        pass

    @abstractmethod
    def after_train(self):
        pass
