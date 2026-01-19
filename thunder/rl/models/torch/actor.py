from dataclasses import dataclass

import torch
import torch.nn as nn

from thunder.nn.distributions import Distributions

from ..protocol import Actor, ActorStep
