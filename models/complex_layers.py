from typing import Union
import math

import numpy as np
from einops import rearrange
import torch
import torch.nn as nn

from utils.utils_general import stable_angle


class ComplexConv2d(nn.Module):
    """ Complex conv2d layer which applies matrix-vector product op (in cartesian form) 
    and performs addition op of magnitude and phase-component biases (in polar form). 
    Additional option to specify whether layer acts in transposed conv. form. """
    def __init__(self, 
        in_channels: int,
        out_channels: int, 
        kernel_size: int,
        stride: int,
        padding: Union[str, int],
        phase_init_min: float = -math.pi,
        phase_init_max: float = math.pi
        ):
        super(ComplexConv2d, self).__init__()

        self.in_channels = in_channels
        self.stride = stride 
        self.padding = padding
        self.phase_init_min = phase_init_min
        self.phase_init_max = phase_init_max

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False, dtype=torch.complex64
        )
        self.conv.apply(self._init_complex_weights)
        self.magnitude_bias = nn.Parameter(torch.empty((1, out_channels, 1, 1)))
        fan_in = in_channels * kernel_size**2
        self.magnitude_bias = self._init_magnitude_bias(fan_in, self.magnitude_bias)
        self.phase_bias = nn.Parameter(torch.empty((1, out_channels, 1, 1)))
        self.phase_bias = self._init_phase_bias(self.phase_bias)

    def _init_complex_weights(self, module):
        w_scale = 1 / np.sqrt(np.prod(module.weight.shape[1:]))
        w_magnitude = np.random.rayleigh(scale=w_scale, size=module.weight.shape)
        w_phase = np.random.uniform(
            low=self.phase_init_min, high=self.phase_init_max, 
            size=module.weight.shape
        )
        with torch.no_grad():
            module.weight.real.copy_(torch.from_numpy(w_magnitude * np.cos(w_phase)))
            module.weight.imag.copy_(torch.from_numpy(w_magnitude * np.sin(w_phase)))
        print(f'Initialized weights of layer {module}')

    def _init_magnitude_bias(self, fan_in, magnitude_bias):
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(magnitude_bias, -bound, bound)
        return magnitude_bias
    
    def _init_phase_bias(self, phase_bias):
        return nn.init.constant_(phase_bias, val=0)
    
    def forward(self, x, use_symmetric=False):
        if not use_symmetric:
            x = torch.nn.functional.conv2d(
                x, self.conv.weight, bias=None, stride=self.stride, padding=self.padding
            )
            x_magnitude = x.abs() + self.magnitude_bias
            x_phase = stable_angle(x) + self.phase_bias
        else:
            x = torch.nn.functional.conv2d(
                x, torch.conj(self.conv.weight), bias=None, stride=1, padding="same"
            )
            x_magnitude = x.abs() + torch.conj(self.magnitude_bias)
            x_phase = stable_angle(x) + torch.conj(self.phase_bias)
        return x_magnitude * torch.exp(x_phase * 1j)


class ComplexMaxPool2d(nn.Module):
    """MaxPool2d layer that performs max pooling on magnitude-components and 
    copies the corresponding phase-components of complex-valued activations. """
    def __init__(self, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.mag_pool = nn.MaxPool2d(kernel_size, stride, padding, return_indices=True)

    def forward(self, x):
        magnitude, indices = self.mag_pool(x.abs())
        bsz, c, h, w = magnitude.shape
        flat_phase = rearrange(x.angle(), 'b c h w -> b c (h w)')
        flat_indices = rearrange(indices, 'b c h w -> b c (h w)')
        flat_phase_pool = torch.gather(flat_phase, dim=-1, index=flat_indices)
        phase = rearrange(flat_phase_pool, 'b c (h w) -> b c h w', h=h, w=w)
        return magnitude * torch.exp(phase * 1j)


class ComplexUpSample2d(nn.Module):
    """ Upsample2d layer for complex-valued activations that independently 
    upsamples the magnitude and phase-components. """
    def __init__(self, size: int, mode: str = 'nearest'):
        super().__init__()
        self.mag_upsample = nn.Upsample(size=size, mode=mode)
        self.phase_upsample = nn.Upsample(size=size, mode=mode)

    def forward(self, x):
        magnitude = self.mag_upsample(x.abs())
        phase = self.phase_upsample(x.angle())
        return magnitude * torch.exp(phase * 1j)
    

class Activation(nn.Module):
    def __init__(self, hidden_dim=None, activation_type="magic", eps=1e-8):
        """ Implements the following activation functions:
        1. modReLU used in https://arxiv.org/abs/1705.09792
        2. vanilla ReLU
        """
        super(Activation, self).__init__()
        self.activation_type = activation_type
        self.eps = eps
        if self.activation_type == "modrelu":
            self.deadzone_b = nn.Parameter(torch.Tensor(1, hidden_dim, 1, 1))
            nn.init.constant_(self.deadzone_b, -0.1)

    def forward(self, x):
        x_magnitude = x.abs()
        z = None
        if self.activation_type == "modrelu":
            z = nn.functional.relu(x_magnitude + self.deadzone_b) * (torch.exp(x.angle() * 1j)) 
        elif self.activation_type == "relu":
            z = nn.functional.relu(x_magnitude) * (torch.exp(x.angle() * 1j))
        else:
            raise ValueError("Invalid activation type specified!!!!")
        return z