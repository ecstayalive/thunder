from abc import ABC, abstractmethod

from thunder.core import Algorithm


class Agent(ABC):
    def __init__(self):
        super().__init__()
