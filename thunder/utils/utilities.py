from typing import Optional

import torch


def default_device(device: Optional[str] = None):
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
