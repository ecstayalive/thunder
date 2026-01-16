from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Mapping, Tuple


class Actor(ABC):
    """ """

    @abstractmethod
    def explore(self, *args, **kwargs) -> Tuple[Any, Tuple[Any, Any], Any]:
        """
        Args:
            *args: Additional arguments.
            **kwargs: Additional arguments.
        """
        raise NotImplementedError()

    @abstractmethod
    def decision(self, *args, **kwargs) -> Any:
        """_summary_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()
