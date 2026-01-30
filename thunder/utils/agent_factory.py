from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple, Type

from thunder.core import ModelPack


@dataclass
class AgentSpec(dict):
    agent: Any
    models: ModelPack


@dataclass
class AlgoEntry:
    name: str
    algo_cls: Type[Any]
    default_model_name: str


MODEL_REGISTRY: Dict[Tuple[str, str], ModelBuilder] = {}
ALGOS_REGISTRY: Dict[str, AlgoEntry] = {}


def register_model(name: str, supported_algos: Optional[Iterable[str]] = None):
    """_summary_

    Args:
        name (str): _description_
        supported_algos (Optional[Iterable[str]], optional): _description_. Defaults to None.
    """

    def decorator(builder_cls: Type[ModelBuilder]):
        builder_instance = builder_cls()
        algos = [algo_name.lower() for algo_name in supported_algos]
        for algo in algos:
            key = (algo, name)
            if key in MODEL_REGISTRY:
                raise ValueError(
                    "Conflict: Model '{name}' for algo '{algo}' is already registered."
                )
            MODEL_REGISTRY[key] = builder_instance
        return builder_cls

    return decorator


def register_algo(name: str, algo_cls: Type[Any], default_model_name: str):
    """_summary_

    Args:
        name (str): _description_
        algo_cls (Type[Any]): _description_
        default_model_name (str): _description_
    """

    def decorator(param_builder):
        algo_name = name.lower()
        ALGOS_REGISTRY[algo_name] = AlgoEntry(
            name=algo_name,
            algo_cls=algo_cls,
            param_builder=param_builder(),
            default_model_name=default_model_name,
        )
        return param_builder

    return decorator


class ModelBuilder(ABC):
    """
    Need a unified interface to build models for different algorithms.
    """

    @abstractmethod
    def build(self) -> ModelPack:
        raise NotImplementedError


class AgentFactory:
    """ """

    def __init__(self):
        self._model_cache: Dict[str, ModelPack] = {}

    def create(
        self,
        algo_name: str,
        env: Any,
        cfg: Dict[str, Any],
        algo_cfg: Dict[str, Any],
        optimize_model: bool = True,
        pretrained_model_path: Optional[str] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> AgentSpec:
        """
        Creates an agent pack containing the agent, actor, and critic.
        Args:
        Returns:
            An AgentPack containing the instantiated agent, actor, and critic.
        """
        algo_name = algo_name.lower()
        algo_entry = ALGOS_REGISTRY.get(algo_name)
        if not algo_entry:
            raise ValueError(f"Algorithm '{algo_name}' is not registered.")
        model_factory_name = cfg.get("model_factory", algo_entry.default_model_name)
        # print((algo_name, model_factory_name))
        model_builder = MODEL_REGISTRY.get((algo_name, model_factory_name))
        if model_builder is None:
            model_builder = MODEL_REGISTRY.get(model_factory_name)
        if model_builder is None:
            raise ValueError(
                f"Model factory '{model_factory_name}' not found for algo '{algo_name}'."
            )
        cache_key = f"{algo_name}_{model_factory_name}"
        if cache_key not in self._model_cache:
            models = model_builder.build(
                env,
                cfg,
                algo_cfg,
                optimize_model,
                pretrained_model_path,
                extra_info,
            )
            self._model_cache[cache_key] = models
        else:
            models = self._model_cache[cache_key]
        agent_kwargs = algo_entry.param_builder.build(env, cfg, algo_cfg, extra_info)
        agent_kwargs.update()
        AgentClass = algo_entry.algo_cls
        agent = AgentClass(**agent_kwargs)
        return AgentSpec(agent=agent, models=models)
