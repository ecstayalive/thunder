import copy
from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional, Tuple

import thunder
import torch
import torch.nn as nn

from .info_processor import ModulesInfoProcessor, NetInfo
from .net_constructor import NetConstructor

# TODO: Make stricter constraints on `NetFactory`
# TODO: Optimize the produced network
# NOTE: For complex network architectures, we do not
#       recommend to use `NetFactory` to construct the network

__all__ = ["NetFactory"]


class NetFactory:
    """This class is used to construct a network from yaml node.
    For example, given a yaml node(noticing: the node must comply with the specification)
        net:
          - lstm_mlp:
              shape: [16, 17, 18]
              rnn_num_layer: 1
    For the list of configuring the network, the following statements need to be made:
        1: This class only supports the configuration of some network modules.
        See `FACTORY_SUPPORTED_NET` to determine the supported network
        2: For each sub-network, the key presents the network's type,
        the user needs to specify the shape of the network, (mainly the
        shape of the hidden layer, because if the sub-network is not
        located at the input or output position, the code will automatically
        analyze the input and output size), the activation function and some
        parameters consistent with the sub-network's specification.
        TODO: For each sub-network, a document needs to be written describing
        the supported configurations
        3: And for the output and input features of the network, the user
        needs to specify.
        4: Users need to ensure the rationality of the sequence structure
        of the network in the information. For example, `mlp` cannot be
        directly followed by `lstm`. If you want to construct this architecture,
        you can directly use the `lstm_mlp` network type. For another example,
        if there is no `flatten()` after a separate `convolutional layer`
        or `capsule layer`, you cannot use `mlp` directly after it.
        On the contrary, users can use the modules already written to
        replace this structure.
    Args:
        modules_info: A iterable sequence of modules' information
    """

    def __init__(
        self, modules_info: Iterator[Dict], net_name: Optional[str] = "CustomNet"
    ) -> None:
        self.modules_info_processor = ModulesInfoProcessor(copy.deepcopy(modules_info))
        self.net_constructor = NetConstructor(net_name)

    @classmethod
    def make(
        cls,
        in_features: Optional[int | Tuple[int, ...]] = None,
        out_features: Optional[int] = None,
        modules_info: Iterator[Dict] = None,
        net_name: str = None,
        only_return_modules: bool = True,
    ) -> Tuple[nn.Module, NetInfo]:
        if modules_info is not None:
            obj = cls(modules_info, net_name)
        else:
            raise ValueError(
                "You must provide modules information, "
                "so the factory can construct a network according to the information!"
            )
        return obj(in_features, out_features, only_return_modules)

    def __call__(
        self,
        in_features: Optional[int | Tuple[int, ...]] = None,
        out_features: Optional[int] = None,
        only_return_modules: bool = True,
    ) -> Tuple[nn.Module, NetInfo]:
        modules_config, net_info = self.modules_info_processor(
            in_features, out_features
        )
        return self.net_constructor(modules_config, net_info, only_return_modules)
