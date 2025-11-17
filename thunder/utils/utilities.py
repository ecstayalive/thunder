import torch


def default_device(device: torch.device | str = "cpu"):
    if device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
