from abc import ABC, abstractmethod


class Algos(ABC):
    """_summary_

    Args:
        ABC (_type_): _description_
    """

    @abstractmethod
    def act(self, *args, **kwargs):
        pass

    @abstractmethod
    def store(self, *args, **kwargs):
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass
