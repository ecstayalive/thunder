import numpy as np
import torch
import torch.nn as nn

import thunder.algorithms as algo
from thunder.nn import *
from thunder.rl import DecActor, GeneralActor, GeneralVNet, NetFactory, RoaActor
from thunder.rl.distributions import ConsistentGaussian
from thunder.rl.utils import DimAdaptRMlp, EmbedConvRMlp

__all__ = ["SetupAgent"]


class AgentFactory:
    """ """

    def __init__(self, algo_name: str, device=None):
        self._agent_mapping = {
            "ppo": self.ppo_agent,
            "roa_ppo": self.roa_ppo_agent,
            "sac": self.sac_agent,
        }

    def ppo_agent(self): ...

    def roa_ppo_agent(self): ...

    def sac_agent(self): ...

    def actor(self, actor_cfg: dict): ...

    def critic(self, critic_cfg: dict): ...


class SetupAgent:
    """ """

    def __init__(
        self,
        algo_name: str,
        device=None,
    ):
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
        height_map_shape = env.local_heightmap_shape
        conv_head = Conv2dBlock(
            height_map_shape, (16, 16, 32), (3, 3, 3), pool_kernel=(2, 2), gap=True
        )
        rmlp_head = EmbedLstmMlp(
            env.getObDim(info.get("critic_obs_id", 0))
            - conv_head.in_features
            + conv_head.out_features,
            conv_head.out_features,
            1,
            256,
            [256, 256],
            embed_type="normal",
        )
        critic_kernel = EmbedConvRMlp(conv_head, rmlp_head)
        # critic_kernel, _ = NetFactory.make(
        #     env.getObDim(info.get("critic_obs_id", 0)), 1, arch_cfg["critic"]["kernel"]
        # )
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
        conv_head = Conv2dBlock(
            height_map_shape, (16, 16, 32), (3, 3, 3), pool_kernel=(2, 2), gap=True
        )
        rmlp_head = EmbedLstmMlp(
            env.getObDim(info.get("critic_obs_id", 0))
            - conv_head.in_features
            + conv_head.out_features,
            conv_head.out_features,
            1,
            256,
            [512, 256],
        )
        critic_kernel = EmbedConvRMlp(conv_head, rmlp_head)
        # critic_kernel, _ = NetFactory.make(
        #     env.getObDim(info.get("critic_obs_id", 0)), 1, arch_cfg["critic"]["kernel"]
        # )
        critic = GeneralVNet(DimAdaptRMlp(critic_kernel))
        obs_enc = DimAdaptRMlp(
            LstmMlp(
                env.getObDim(info.get("actor_obs_id", 0)), 128, 256, [192], activate_output=True
            )
        )
        conv_head = Conv2dBlock(
            height_map_shape, (16, 16, 32), (3, 3, 3), pool_kernel=(2, 2), gap=True
        )
        rmlp_head = EmbedLstmMlp(
            env.getObDim(info.get("critic_obs_id", 0))
            - conv_head.in_features
            + conv_head.out_features,
            conv_head.out_features,
            128,
            256,
            [192],
            activate_output=True,
        )
        state_enc = DimAdaptRMlp(EmbedConvRMlp(conv_head, rmlp_head))
        # state_enc = DimAdaptRMlp(
        #     LstmMlp(
        #         env.getObDim(info.get("critic_obs_id", 0)), 128, 256, [192], activate_output=True
        #     )
        # )
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
        action_decoder = ConsistentGaussian(256, 18, [192, 128])
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
