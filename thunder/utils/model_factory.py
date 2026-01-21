from thunder.nn.torch.models import recursive_instantiate

from .agent_factory import ModelBuilder


class DynamicModelBuilder(ModelBuilder):
    def build(self, env, cfg, *args, **kwargs):
        full_cfg = cfg.copy()
        if "action_dim" not in full_cfg:
            full_cfg["action_dim"] = env.action_dim
        return recursive_instantiate(full_cfg)
