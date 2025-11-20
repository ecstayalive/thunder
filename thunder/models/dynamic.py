import importlib


def recursive_instantiate(cfg):
    """ """
    if isinstance(cfg, dict) and "_target_" in cfg:
        class_path = cfg["_target_"]
        kwargs = {k: v for k, v in cfg.items() if k != "_target_"}

        for k, v in kwargs.items():
            kwargs[k] = recursive_instantiate(v)

        module_name, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls(**kwargs)
    elif isinstance(cfg, list):
        return [recursive_instantiate(item) for item in cfg]
    else:
        return cfg
