from abc import ABC, abstractmethod


class Algorithm(ABC):
    Modules = []

    def __init__(self, *args, **kwargs): ...

    @abstractmethod
    def train_step(self, *args, **kwargs) -> dict: ...
