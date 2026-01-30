from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Type

import numpy as np
import thunder.algorithms as algo
import torch
import torch.nn as nn
from thunder.models import *
from thunder.nn import *
from thunder.rl import DecActor, GeneralActor, GeneralVNet, NetFactory, RoaActor
from thunder.rl.distributions import ConsistentGaussian
from thunder.rl.utils import DimAdaptRMlp

GLOBAL_ALGO_KEY = "GlobalAlgo"


class ModelPack(dict):
    """ """

    def __init__(self, **kwargs: nn.Module):
        super().__init__(kwargs)

    def __getattr__(self, name: str) -> nn.Module:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'ModelPack' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: nn.Module):
        self[name] = value

    def to(self, device: torch.device):
        """ """
        for v in self.values():
            if isinstance(v, torch.nn.Module):
                v.to(device)
        return self

    def state_dict(self):
        """ """
        return {k: v.state_dict() for k, v in self.items() if isinstance(v, torch.nn.Module)}

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            if k in self and isinstance(self[k], torch.nn.Module):
                self[k].load_state_dict(v)


@dataclass
class AgentPack(dict):
    agent: Any
    models: ModelPack


@dataclass
class AlgoEntry:
    name: str
    algo_cls: Type[Any]
    param_builder: SpecBuilder
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
        algos = (
            [algo_name.lower() for algo_name in supported_algos]
            if supported_algos
            else [GLOBAL_ALGO_KEY]
        )
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

    def decorator(param_builder: Type[SpecBuilder]):
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
    def build(
        self,
        env: Any,
        cfg: Dict[str, Any],
        algo_cfg: Dict[str, Any],
        device: torch.device,
        optimize_model: bool = True,
        pretrained_model_path: Optional[str] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> ModelPack:
        raise NotImplementedError


class SpecBuilder(ABC):
    @abstractmethod
    def build(
        self,
        env: Type[Any],
        cfg: Dict[str, Any],
        algo_cfg: Dict[str, Any],
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
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
        device: torch.device,
        optimize_model: bool = True,
        pretrained_model_path: Optional[str] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> AgentPack:
        """
        Creates an agent pack containing the agent, actor, and critic.

        Args:
            algo_name: The name of the algorithm to create (e.g., "ppo").
            env: The environment object, used to extract necessary dimensions and info.
            cfg: General configuration dictionary.
            algo_cfg: Algorithm-specific hyperparameters.
            device: The torch device.
            actor: An optional pre-existing actor module.
            critic: An optional pre-existing critic module.
            optimize_model: Flag to compile the models.
            pretrained_model_path: Path to a pre-trained model.
            extra_info: Dictionary for optional observation IDs.

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
            model_builder = MODEL_REGISTRY.get((GLOBAL_ALGO_KEY, model_factory_name))
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
                device,
                optimize_model,
                pretrained_model_path,
                extra_info,
            )
            self._model_cache[cache_key] = models
        else:
            models = self._model_cache[cache_key]
        agent_kwargs = algo_entry.param_builder.build(env, cfg, algo_cfg, extra_info)
        agent_kwargs.update(models | {"device": device})
        AgentClass = algo_entry.algo_cls
        agent = AgentClass(**agent_kwargs)
        return AgentPack(agent=agent, models=models)


@register_model("general", ("ppo", "sac", "roa"))
class GeneralModelFactory:
    def build(
        self,
        env: Any,
        cfg: Dict[str, Any],
        algo_cfg: Dict[str, Any],
        device: torch.device,
        optimize_model: bool = True,
        pretrained_model_path: Optional[str] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> ModelPack:
        arch_cfg = cfg["architecture"]
        actor_obs_dim = env.getObDim(extra_info.get("actor_obs_id", 0))
        critic_obs_dim = env.getObDim(extra_info.get("critic_obs_id", 0))
        action_dim = env.action_dim
        critic_kernel, _ = NetFactory.make(critic_obs_dim, 1, arch_cfg["critic"]["kernel"])
        critic = GeneralVNet(DimAdaptRMlp(critic_kernel))
        actor = GeneralActor.make(arch_cfg["actor"], actor_obs_dim, action_dim)
        orthogonal_modules_(actor, critic)
        if pretrained_model_path is not None:
            actor.restore(pretrained_model_path)
        if optimize_model:
            actor = torch.compile(actor, mode="max-autotune")
            critic = torch.compile(critic)
        return ModelPack(actor=actor, critic=critic)


@register_model("test_perception", ("ppo", "sac"))
class PerceptionModelFactory:
    def build(
        self,
        env: Any,
        cfg: Dict[str, Any],
        algo: Dict[str, Any],
        device: torch.device,
        optimize_model: bool = True,
        pretrained_model_path: Optional[str] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> ModelPack:
        arch_cfg = cfg["architecture"]
        actor_obs_dim = env.getObDim(extra_info.get("actor_obs_id", 0))
        critic_obs_dim = env.getObDim(extra_info.get("critic_obs_id", 0))
        action_dim = env.action_dim
        heightmap_shape = env.getLocalHeightMapShape()
        foot_heightmap_shape = env.getLocalFootHeightMapShape()
        # c_conv_head = Conv2dBlock(foot_heightmap_shape, (8, 16), ((1, 5), (1, 5)), ((1, 3), (1, 3)))
        c_conv_head = Conv2dBlock(heightmap_shape, (8, 16), (5, (5, 3)), (3, (3, 2)))
        critic_kernel = Perception(critic_obs_dim, 1, 256, c_conv_head, [256, 256])
        critic = GeneralVNet(DimAdaptRMlp(critic_kernel))
        # Actor Network
        a_conv_head = Conv2dBlock(heightmap_shape, (8, 8), (5, (5, 3)), (3, (3, 2)))
        enc = BeliefPerception(actor_obs_dim, 256, 256, a_conv_head)
        dec = ConsistentGaussian(256, action_dim, [256])
        actor = GeneralActor(DimAdaptRMlp(enc), dec)
        orthogonal_modules_(actor, critic)
        if pretrained_model_path is not None:
            actor.restore(pretrained_model_path)
        if optimize_model:
            actor = torch.compile(actor, mode="max-autotune")
            critic = torch.compile(critic)
        return ModelPack(actor=actor, critic=critic)


@register_model("test_mha", ("ppo", "sac"))
class MhaModelFactory:
    def build(
        self,
        env: Any,
        cfg: Dict[str, Any],
        algo: Dict[str, Any],
        device: torch.device,
        optimize_model: bool = True,
        pretrained_model_path: Optional[str] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> ModelPack:
        arch_cfg = cfg["architecture"]
        actor_obs_dim = env.getObDim(extra_info.get("actor_obs_id", 0))
        critic_obs_dim = env.getObDim(extra_info.get("critic_obs_id", 0))
        action_dim = env.action_dim
        heightmap_shape = env.getLocalHeightMapShape()
        foot_heightmap_shape = env.getLocalFootHeightMapShape()
        # c_conv_head = Conv2dBlock(foot_heightmap_shape, (8, 16), ((1, 5), (1, 5)), ((1, 3), (1, 3)))
        c_conv_head = Conv2dBlock(heightmap_shape, (8, 16), (5, 5), (3, 3))
        critic_kernel = Perception(critic_obs_dim, 1, 256, c_conv_head, [256, 256])
        critic = GeneralVNet(DimAdaptRMlp(critic_kernel))
        # Actor Network
        a_conv_head = Conv2dBlock(heightmap_shape, (8, 16), (3, 3), (2, 2))
        enc = MhaBelief(actor_obs_dim, 256, 256, a_conv_head, num_heads=2)
        dec = ConsistentGaussian(256, action_dim, [256])
        actor = GeneralActor(DimAdaptRMlp(enc), dec)
        orthogonal_modules_(actor, critic)
        if pretrained_model_path is not None:
            actor.restore(pretrained_model_path)
        if optimize_model:
            actor = torch.compile(actor, mode="max-autotune")
            critic = torch.compile(critic)
        return ModelPack(actor=actor, critic=critic)


@register_model("test_spatial", ("ppo", "sac"))
class SpatialAttentionModelFactory:
    def build(
        self,
        env: Any,
        cfg: Dict[str, Any],
        algo: Dict[str, Any],
        device: torch.device,
        optimize_model: bool = True,
        pretrained_model_path: Optional[str] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> ModelPack:
        arch_cfg = cfg["architecture"]
        actor_obs_dim = env.getObDim(extra_info.get("actor_obs_id", 0))
        critic_obs_dim = env.getObDim(extra_info.get("critic_obs_id", 0))
        action_dim = env.action_dim
        heightmap_shape = env.getLocalHeightMapShape()
        foot_heightmap_shape = env.getLocalFootHeightMapShape()
        # c_conv_head = Conv2dBlock(foot_heightmap_shape, (8, 16), ((1, 5), (1, 5)), ((1, 3), (1, 3)))
        c_conv_head = Conv2dBlock(heightmap_shape, (8, 16), (5, 5), (3, 3))
        critic_kernel = Perception(critic_obs_dim, 1, 256, c_conv_head, [256, 256])
        critic = GeneralVNet(DimAdaptRMlp(critic_kernel))
        # Actor Network
        a_conv_head = Conv2dBlock(heightmap_shape, (8, 16), (3, 3), (2, 2))
        enc = SpatialBelief(actor_obs_dim, 256, 256, a_conv_head)
        dec = ConsistentGaussian(256, action_dim, [256])
        actor = GeneralActor(DimAdaptRMlp(enc), dec)
        orthogonal_modules_(actor, critic)
        if pretrained_model_path is not None:
            actor.restore(pretrained_model_path)
        if optimize_model:
            actor = torch.compile(actor, mode="max-autotune")
            critic = torch.compile(critic)
        return ModelPack(actor=actor, critic=critic)


@register_algo("ppo", algo.PPO, "general")
class PPOSpecBuilder(SpecBuilder):
    def build(self, env, cfg, algo_cfg, extra_info=None):
        q_limits = env.getRobotJointLimits()
        q0 = env.getRobotJointPos0()
        init_std = 0.5 * np.minimum(q0 - q_limits[:, 0], q_limits[:, 1] - q0)
        init_std = np.minimum(0.5 * env.getRobotMaxTorDq(), init_std)
        params = {
            "num_envs": getattr(env, "num_envs", 1),
            "num_collects": cfg["update_every_n_steps"],
            "init_std": init_std,
        }
        params.update(algo_cfg)
        # if extra_info:
        #     params.update(extra_info)
        return params


class SetupAgent:
    """ """

    def __init__(self, algo_name: str, device=None):
        self.alg_name = algo_name
        try:
            factory_name = algo_name.lower()
            self._factory = globals()[f"setup_{factory_name}"]
        except KeyError:
            raise ValueError(f"Unknown algorithm: {algo_name}")
        self._info = {}
        self._device = device

    def __call__(
        self,
        env,
        cfg: dict,
        algo_cfg: dict,
        actor: nn.Module = None,
        critic: nn.Module = None,
        optimize_model: bool = True,
        pretrained_model_path: str = None,
        **kwargs,
    ):
        self._info.update(kwargs)
        if actor is not None:
            self._info["actor"] = actor
        if critic is not None:
            self._info["critic"] = critic
        return self._factory(
            env,
            cfg.copy(),
            algo_cfg.copy(),
            self._device,
            self._info,
            optimize_model,
            pretrained_model_path,
        )


def setup_ppo(
    env,
    cfg: dict,
    algo_cfg: dict,
    device: torch.device,
    info: dict,
    optimize_model: bool = True,
    pretrained_model_path: str = None,
):
    Agent = algo.PPO
    if "actor" in info and "critic" in info:
        actor, critic = info["actor"], info["critic"]
        init_std = None
    else:
        q_limits = env.single_obj.getRobotJointLimits()
        q0 = env.single_obj.getRobotJointPos0()
        init_std = 0.5 * np.minimum(q0 - q_limits[:, 0], q_limits[:, 1] - q0)
        init_std = np.minimum(0.5 * env.single_obj.getRobotMaxTorDq(), init_std)
        arch_cfg = cfg["architecture"]
        heightmap_shape = env.local_heightmap_shape
        foot_heightmap_shape = env.local_foot_heightmap_shape
        critic_kernel, _ = NetFactory.make(
            env.getObDim(info.get("critic_obs_id", 0)), 1, arch_cfg["critic"]["kernel"]
        )
        critic = GeneralVNet(DimAdaptRMlp(critic_kernel))
        actor = GeneralActor.make(
            arch_cfg["actor"],
            env.getObDim(info.get("actor_obs_id", 0)),
            env.action_dim,
        )
        orthogonal_modules_(actor, critic)
        if pretrained_model_path is not None:
            actor.restore(pretrained_model_path)
        if optimize_model:
            actor = torch.compile(actor, mode="max-autotune")
            critic = torch.compile(critic)
        info["actor"], info["critic"] = actor, critic

    return Agent(
        actor,
        critic,
        num_envs=env.num_envs,
        num_collects=cfg["update_every_n_steps"],
        init_std=init_std,
        device=device,
        **algo_cfg,
    )


def setup_am_ppo(
    env,
    cfg: dict,
    algo_cfg: dict,
    device: torch.device,
    info: dict,
    optimize_model: bool = True,
    pretrained_model_path: str = None,
):
    Agent = algo.AdvantageMixPPO
    if "actor" in info and "critic" in info:
        actor, critic = info["actor"], info["critic"]
    else:
        arch_cfg = cfg["architecture"]
        critic_kernel, _ = NetFactory.make(
            env.getObDim(info.get("critic_obs_id", 0)), 1, arch_cfg["critic"]["kernel"]
        )
        critic = GeneralVNet(DimAdaptRMlp(critic_kernel))
        actor = GeneralActor.make(
            arch_cfg["actor"],
            env.getObDim(info.get("actor_obs_id", 0)),
            env.action_dim,
        )
        orthogonal_modules_(actor, critic)
        if pretrained_model_path is not None:
            actor.restore(pretrained_model_path)
        if optimize_model:
            actor = torch.compile(actor, mode="max-autotune")
            critic = torch.compile(critic)
        info["actor"], info["critic"] = actor, critic
    return Agent(
        actor,
        critic,
        num_envs=env.num_envs,
        num_collects=cfg["update_every_n_steps"],
        device=device,
        **algo_cfg,
    )


def setup_ggf_ppo(
    env,
    cfg: dict,
    algo_cfg: dict,
    device: torch.device,
    info: dict,
    optimize_model: bool = True,
    pretrained_model_path: str = None,
):
    Agent = algo.GgfPPO
    if "actor" in info and "critic" in info:
        actor, critic = info["actor"], info["critic"]
    else:
        arch_cfg = cfg["architecture"]
        critic_kernel, _ = NetFactory.make(
            env.getObDim(info.get("critic_obs_id", 0)), 2, arch_cfg["critic"]["kernel"]
        )
        critic = GeneralVNet(DimAdaptRMlp(critic_kernel))
        actor = GeneralActor.make(
            arch_cfg["actor"],
            env.getObDim(info.get("actor_obs_id", 0)),
            env.action_dim,
        )
        orthogonal_modules_(actor, critic)
        if pretrained_model_path is not None:
            actor.restore(pretrained_model_path)
        if optimize_model:
            actor = torch.compile(actor, mode="max-autotune")
            critic = torch.compile(critic)
        info["actor"], info["critic"] = actor, critic
    return Agent(
        actor,
        critic,
        num_envs=env.num_envs,
        num_collects=cfg["update_every_n_steps"],
        device=device,
        **algo_cfg,
    )


def setup_denoise_ppo(
    env,
    cfg: dict,
    algo_cfg: dict,
    device: torch.device,
    info: dict,
    optimize_model: bool = True,
    pretrained_model_path: str = None,
):
    """ """
    Agent = algo.DenoisePPO
    if "actor" in info and "critic" in info:
        actor, critic = info["actor"], info["critic"]
    else:
        arch_cfg = cfg["architecture"]
        critic_kernel, _ = NetFactory.make(
            env.getObDim(info.get("critic_obs_id", 0)), 1, arch_cfg["critic"]["kernel"]
        )
        critic = GeneralVNet(DimAdaptRMlp(critic_kernel))
        state_encoder = LinearBlock(
            env.getObDim(info.get("actor_obs_id", 0)), 256, [192], activate_output=True
        )
        latent_encoder = DimAdaptRMlp(LstmMlp(256, 128, 256, [192]))
        action_decoder = ConsistentGaussian(128, env.action_dim, [])
        state_decoder = LinearBlock(256, env.getObDim(info.get("critic_obs_id", 0)), [192])
        actor = DecActor(state_encoder, latent_encoder, action_decoder, state_decoder)
        orthogonal_modules_(actor, critic)
        if pretrained_model_path is not None:
            actor.restore(pretrained_model_path)
        if optimize_model:
            actor = torch.compile(actor, mode="max-autotune")
            critic = torch.compile(critic)
        info["actor"], info["critic"] = actor, critic

    return Agent(
        actor,
        critic,
        num_envs=env.num_envs,
        num_collects=cfg["update_every_n_steps"],
        device=device,
        **algo_cfg,
    )


def setup_roa_ppo(
    env,
    cfg: dict,
    algo_cfg: dict,
    device: torch.device,
    info: dict,
    optimize_model: bool = True,
    pretrained_model_path: str = None,
):
    Agent = algo.RoaPPO

    if "actor" in info and "critic" in info:
        init_std = None
        actor, critic = info["actor"], info["critic"]
    else:
        arch_cfg = cfg["architecture"]
        q_limits = env.single_obj.getRobotJointLimits()
        q0 = env.single_obj.getRobotJointPos0()
        init_std = 0.5 * np.minimum(q0 - q_limits[:, 0], q_limits[:, 1] - q0)
        height_map_shape = env.local_heightmap_shape
        critic_kernel, _ = NetFactory.make(
            env.getObDim(info.get("critic_obs_id", 0)), 1, arch_cfg["critic"]["kernel"]
        )
        critic = GeneralVNet(DimAdaptRMlp(critic_kernel))
        obs_enc = DimAdaptRMlp(
            LstmMlp(
                env.getObDim(info.get("actor_obs_id", 0)), 128, 256, [192], activate_output=True
            )
        )
        state_enc = DimAdaptRMlp(
            LstmMlp(
                env.getObDim(info.get("critic_obs_id", 0)), 128, 256, [192], activate_output=True
            )
        )
        action_dec = ConsistentGaussian(128, env.action_dim, [])
        actor = RoaActor(obs_enc, state_enc, action_dec)
        orthogonal_modules_(actor, critic)
        if pretrained_model_path is not None:
            actor.restore(pretrained_model_path)
        if optimize_model:
            actor = torch.compile(actor, mode="max-autotune")
            critic = torch.compile(critic)
        info["actor"], info["critic"] = actor, critic
    return Agent(
        actor,
        critic,
        num_envs=env.num_envs,
        num_collects=cfg["update_every_n_steps"],
        init_std=init_std,
        device=device,
        **algo_cfg,
    )


def setup_mix_roa_ppo(
    env,
    cfg: dict,
    algo_cfg: dict,
    device: torch.device,
    info: dict,
    optimize_model: bool = True,
    pretrained_model_path: str = None,
):
    Agent = algo.MixRoaPPO
    if "actor" in info and "critic" in info:
        actor, critic = info["actor"], info["critic"]
    else:
        arch_cfg = cfg["architecture"]
        critic_kernel, _ = NetFactory.make(
            env.getObDim(info.get("critic_obs_id", 0)), 2, arch_cfg["critic"]["kernel"]
        )
        critic = GeneralVNet(DimAdaptRMlp(critic_kernel))
        obs_enc = DimAdaptRMlp(
            LstmMlp(
                env.getObDim(info.get("actor_obs_id", 0)), 128, 256, [192], activate_output=True
            )
        )
        state_enc = DimAdaptRMlp(
            LstmMlp(
                env.getObDim(info.get("critic_obs_id", 0)), 128, 256, [192], activate_output=True
            )
        )
        action_dec = ConsistentGaussian(128, env.action_dim, [])
        actor = RoaActor(obs_enc, state_enc, action_dec)
        orthogonal_modules_(actor, critic)
        if pretrained_model_path is not None:
            actor.restore(pretrained_model_path)
        if optimize_model:
            actor = torch.compile(actor, mode="max-autotune")
            critic = torch.compile(critic)
        info["actor"], info["critic"] = actor, critic
    return Agent(
        actor,
        critic,
        num_envs=env.num_envs,
        num_collects=cfg["update_every_n_steps"],
        device=device,
        **algo_cfg,
    )


def setup_x_ppo(
    env,
    cfg: dict,
    algo_cfg: dict,
    device: torch.device,
    info: dict,
    optimize_model: bool = True,
    pretrained_model_path: str = None,
):
    Agent = algo.DenoisePPO
    if "actor" in info and "critic" in info:
        actor, critic = info["actor"], info["critic"]
    else:
        arch_cfg = cfg["architecture"]
        critic_kernel, _ = NetFactory.make(
            env.getObDim(info.get("critic_obs_id", 0)), 1, arch_cfg["critic"]
        )
        state_encoder = DimAdaptRMlp(nn.LSTM(97, 256, 1))
        state_decoder = LinearBlock(256, 185, [192])
        action_decoder = ConsistentGaussian(256, 18, shape=[192, 128])
        actor = DecActor(state_encoder, state_decoder, action_decoder)
        critic = algo.GeneralVNet(DimAdaptRMlp(critic_kernel))
        orthogonal_modules_(actor, critic)
        if pretrained_model_path is not None:
            actor.restore(pretrained_model_path)
        if optimize_model:
            actor = torch.compile(actor, fullgraph=True, mode="max-autotune")
            critic = torch.compile(critic, fullgraph=True, mode="max-autotune")
        info["actor"], info["critic"] = actor, critic

    return Agent(
        actor,
        critic,
        num_envs=env.num_envs,
        num_collects=cfg["update_every_n_steps"],
        device=device,
        **algo_cfg,
    )


def setup_sac(env, cfg: dict, algo_cfg: dict, device, info: dict):
    Agent = algo.SAC

    if "actor" in info and "critic" in info:
        actor, critic = info["actor"], info["critic"]
    else:
        arch_cfg = cfg["architecture"]
        actor = GeneralActor.make(arch_cfg["actor"], env.ob_dim, env.action_dim)
        critic = Agent.Critic.make(arch_cfg["critic"], env.ob_dim, env.action_dim)
        info["actor"], info["critic"] = actor, critic

    return Agent(actor, critic, num_envs=env.num_envs, device=device, **algo_cfg)
