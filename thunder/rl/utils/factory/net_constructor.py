import ast
import types
from inspect import Parameter
from typing import Dict, Iterator, Tuple

import thunder
import torch
import torch.nn as nn

from .info_processor import NetInfo


class NetConstructor:
    def __init__(self, net_name: str) -> None:
        self.net_name = net_name

    def __call__(
        self,
        modules_config: Iterator[Dict],
        net_info: NetInfo,
        only_return_modules: bool = True,
    ) -> Tuple[nn.Module, NetInfo]:
        self.net_info = net_info
        modules = map(self._get_module, modules_config)

        if net_info.num_modules == 1:
            modules = tuple(modules)[0]
        else:
            modules = nn.ModuleList(modules)

        if only_return_modules:
            return modules, net_info

        def __init__(
            obj, modules: nn.Module | nn.ModuleList, net_info: NetInfo
        ) -> None:
            super(obj.__class__, obj).__init__()
            obj.networks = modules
            obj.net_info = net_info
            if obj.net_info.num_modules == 1:
                obj.forward = obj.signal_forward

        def signal_forward(obj, *args):
            return obj.networks(*args)

        forward = self._create_forward_fn()

        Net = type(
            self.net_name,
            (nn.Module,),
            {
                "__init__": __init__,
                "forward": forward,
                "signal_forward": signal_forward,
            },
        )
        net = Net(modules, net_info)
        return net

    def _get_module(self, module: Dict) -> nn.Module:
        module_name, module_params = module.popitem()
        try:
            module_instance = getattr(thunder.nn, module_name)
        except AttributeError:
            module_instance = getattr(nn, module_name)
        return module_instance(**module_params)

    def _create_forward_fn(self):
        forward_fn_params, forward_default_arg_values = self._generate_forward_params()
        mod_co_argcount = len(forward_fn_params)
        mod_co_nlocals = len(forward_fn_params)
        mod_co_name = "forward"
        mod_co_varnames = tuple(map(lambda param: param.name, forward_fn_params))
        logic_forward_code = self._generate_logic_forward()
        modified_code = logic_forward_code.replace(
            co_argcount=mod_co_argcount,
            co_nlocals=mod_co_nlocals,
            co_varnames=mod_co_varnames,
            co_name=mod_co_name,
        )
        forward = types.FunctionType(
            modified_code,
            {"locals": locals},
            name="forward",
            argdefs=forward_default_arg_values,
        )
        return forward

    def _generate_logic_forward(self):
        def _logic_forward():
            kwargs = locals()
            self = kwargs["self"]
            kwargs.pop("self")
            for idx in range(self.net_info.num_modules):
                if self.net_info.rnn_net_id[idx] is None:
                    kwargs["input"] = self.networks[idx](kwargs["input"])
                elif self.net_info.num_rnn_net == 1:
                    kwargs["input"], kwargs["hidden"] = self.networks[idx](
                        kwargs["input"], kwargs["hidden"]
                    )
                else:
                    (
                        kwargs["input"],
                        kwargs[f"hidden{self.net_info.rnn_net_id[idx]}"],
                    ) = self.networks[idx](
                        kwargs["input"],
                        kwargs[f"hidden{self.net_info.rnn_net_id[idx]}"],
                    )
            return tuple(kwargs.values())

        return _logic_forward.__code__

    def _generate_forward_params(self):
        forward_fn_params = [Parameter("self", Parameter.POSITIONAL_OR_KEYWORD)]
        forward_fn_params.append(Parameter("input", Parameter.POSITIONAL_OR_KEYWORD))
        if self.net_info.num_rnn_net == 1:
            forward_fn_params.append(
                Parameter("hidden", Parameter.POSITIONAL_OR_KEYWORD, default=None)
            )
        else:
            for id in range(self.net_info.num_rnn_net):
                forward_fn_params.append(
                    Parameter(
                        f"hidden{id}", Parameter.POSITIONAL_OR_KEYWORD, default=None
                    )
                )
        forward_default_arg_values = tuple(
            p.default for p in forward_fn_params if p.default != Parameter.empty
        )

        return forward_fn_params, forward_default_arg_values
