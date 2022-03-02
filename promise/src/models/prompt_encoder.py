import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Tuple, Type
import math
from segment_anything.modeling.common import MLPBlock


class LayerNorm3d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, k