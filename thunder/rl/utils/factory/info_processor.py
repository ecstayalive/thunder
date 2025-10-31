from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional, Tuple

from .supported_net_info import *


@dataclass
class NetInfo:
    num_modules: int = None
    in_features: int = None
    out_features: int = None
    net_type: str = "mlp"
    num_rnn_net: int = 0
    rnn_num_layers: int = 1
    rnn_hidden_size: int = None
    is_recurrent: bool = False
    rnn_net_id: Iterator[int] = field(default_factory=list)
    arch: Iterator[Dict] = field(default_factory=list)


class ModulesInfoProcessor:
    def __init__(self, modules_info: Iterator[Dict]) -> None:
        self.num_modules = len(modules_info)
        assert (
            self.num_modules >= 1
        ), "Error, there is no module information in input parameters!!"
        self.net_info = NetInfo(self.num_modules)
        self.modules_info = tuple(map(self._preprocess_modules_info, modules_info))

    def __call__(
        self,
        in_features: Optional[int | Tuple[int, ...]] = None,
        out_features: Optional[int] = None,
    ) -> Tuple[Iterator[Dict], NetInfo]:
        if in_features is not None:
            self._add_input_features(in_features)
        if out_features is not None:
            self._add_output_features(out_features)
        self.modules_config = self._get_modules_config(self.modules_info)
        return self.modules_config, self.net_info

    def _preprocess_modules_info(self, module_info: Dict) -> Dict:
        """Use the module's name in package to replace the
        module type in the given information
        """
        # for each module, the dict only has one key and value
        module_type, module_params_info = module_info.popitem()
        if module_type not in FACTORY_SUPPORTED_NET_MAP:
            raise KeyError(f"Unsupported module type: {module_type}")
        module_name = FACTORY_SUPPORTED_NET_MAP[module_type.lower()]
        return {module_name: module_params_info}

    def _get_modules_config(self, modules_info: Tuple[Dict]) -> Iterator[Dict]:
        """Preprocess the module inventory, make it acceptable on the network
        TODO: optimize this part of code
        """
        modules_config = []
        self.prev_module_out_features = None
        for module_idx, module_info in enumerate(modules_info):
            # for each module, the dict only has one key and value
            module_name, module_params_info = module_info.popitem()
            # get networks input features and output features
            if module_idx == 0:
                self.net_info.in_features = module_params_info["shape"][0]
            if module_idx == self.num_modules - 1:
                self.net_info.out_features = module_params_info["shape"][-1]
            # preprocess module params information according its type
            if module_name in LINEAR_BLOCK:
                module_params = self._get_linear_block_params(
                    module_idx, module_params_info
                )
                self.net_info.rnn_net_id.append(None)
            elif module_name in RECURRENT_MLP:
                self.net_info.is_recurrent = True
                module_params = self._get_recurrent_mlp_params(
                    module_name, module_idx, module_params_info
                )
                self.net_info.rnn_net_id.append(self.net_info.num_rnn_net)
                self.net_info.num_rnn_net += 1
            elif module_name in RECURRENT:
                self.net_info.is_recurrent = True
                module_params = self._get_rnn_params(
                    module_name, module_idx, module_params_info
                )
                self.net_info.rnn_net_id.append(self.net_info.num_rnn_net)
                self.net_info.num_rnn_net += 1

            modules_config.append({module_name: module_params})
            self.net_info.arch.append({module_name: module_params})

        return modules_config

    def _get_linear_block_params(
        self, module_idx: int, module_params_info: Dict
    ) -> Dict:
        module_params = {}
        if self.prev_module_out_features is not None:
            origin_shape_info = module_params_info["shape"]
            module_params_info["shape"] = [
                self.prev_module_out_features,
                *origin_shape_info,
            ]
        module_params["in_features"] = module_params_info["shape"][0]
        module_params["hidden_features"] = module_params_info["shape"][1:-1]
        module_params["out_features"] = module_params_info["shape"][-1]
        self.prev_module_out_features = module_params["out_features"]
        if module_idx < self.num_modules - 1:
            # the module is not the last module
            module_params["activate_output"] = True

        module_params_info.pop("shape")
        module_params.update(module_params_info)

        return module_params

    def _get_recurrent_mlp_params(
        self,
        module_name: str,
        module_idx: int,
        module_params_info: Dict,
    ) -> Dict:
        assert (
            len(module_params_info["shape"]) >= 2
        ), "Module configuration does not follow the specification!"
        module_params = {}
        if module_name == "RecurrentMlp":
            self.net_info.net_type = module_params_info["rnn_type"]
        else:
            if module_name in LSTM_NET:
                self.net_info.net_type = "lstm"
            else:
                self.net_info.net_type = "gru"

        if self.prev_module_out_features is not None:
            origin_shape_info = module_params_info["shape"]
            module_params_info["shape"] = [
                self.prev_module_out_features,
                *origin_shape_info,
            ]
        module_params["in_features"] = module_params_info["shape"][0]
        module_params["rnn_hidden_size"] = module_params_info["shape"][1]
        self.net_info.rnn_hidden_size = module_params["rnn_hidden_size"]
        module_params["mlp_shape"] = module_params_info["shape"][2:-1]
        module_params["out_features"] = module_params_info["shape"][-1]
        self.prev_module_out_features = module_params["out_features"]
        self.net_info.rnn_num_layers = module_params_info.get("rnn_num_layers", 1)
        if module_idx < self.num_modules - 1:
            # mean the module is not the last
            module_params["activate_output"] = True

        module_params_info.pop("shape")
        module_params.update(module_params_info)
        return module_params

    def _get_rnn_params(
        self,
        module_name: str,
        module_idx: int,
        module_params_info: Dict,
    ) -> Dict:
        if module_name in LSTM_NET:
            self.net_info.net_type = "lstm"
        else:
            self.net_info.net_type = "gru"
        module_params = {}
        if self.prev_module_out_features is not None:
            origin_shape_info = module_params_info["shape"]
            module_params_info["shape"] = [
                self.prev_module_out_features,
                *origin_shape_info,
            ]
        module_params["input_size"] = module_params_info["shape"][0]
        module_params["hidden_size"] = module_params_info["shape"][-1]
        self.net_info.rnn_hidden_size = module_params["hidden_size"]
        self.prev_module_out_features = module_params["hidden_size"]
        self.net_info.rnn_num_layers = module_params_info.get("num_layers", 1)
        module_params_info.pop("shape")
        module_params.update(module_params_info)

        return module_params

    def _get_conv_block_params(
        self, module_idx: int, module_config: Dict, next_module_type: str
    ) -> Dict:
        """ """
        ...

    def _add_input_features(self, in_features: int, module_idx: int = 0) -> None:
        """Add input information to one module information"""
        ((module_name, module_params_info),) = self.modules_info[module_idx].items()
        original_shape = module_params_info["shape"]
        if isinstance(original_shape, int):
            self.modules_info[module_idx][module_name]["shape"] = [
                in_features,
                original_shape,
            ]
        else:
            self.modules_info[module_idx][module_name]["shape"] = [
                in_features,
                *original_shape,
            ]

    def _add_output_features(self, out_features: int, module_idx: int = -1) -> None:
        """Add input information to one module information"""
        ((module_name, module_params_info),) = self.modules_info[module_idx].items()
        original_shape = module_params_info["shape"]
        if isinstance(original_shape, int):
            self.modules_info[module_idx][module_name]["shape"] = [
                original_shape,
                out_features,
            ]
        else:
            self.modules_info[module_idx][module_name]["shape"] = [
                *original_shape,
                out_features,
            ]
