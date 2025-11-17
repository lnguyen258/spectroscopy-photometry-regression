from typing import Optional
import math

import torch
from torch import nn

from .model_config import SineKAN_Config

def forward_step(i_n, grid_size, A, K, C):
    ratio = A * grid_size**(-K) + C
    i_n1 = ratio * i_n
    return i_n1

class SineKAN(nn.Module):
    def __init__(
            self, 
            config: Optional[SineKAN_Config] = None,
            input_dim: Optional[int] = None, 
            output_dim: Optional[int] = None, 
            grid_size: Optional[int] = None, 
            is_first: Optional[bool] = None, 
            add_bias: Optional[bool] = None, 
            norm_freq: Optional[bool] = None,
    ):
        super(SineKAN, self).__init__() 

        # Use config if provided 
        if config is not None:
            self.input_dim = input_dim if input_dim is not None else config.input_dim
            self.output_dim = output_dim if output_dim is not None else config.output_dim
            self.grid_size = grid_size if grid_size is not None else config.grid_size
            self.is_first = is_first if is_first is not None else config.is_first
            self.add_bias = add_bias if add_bias is not None else config.add_bias
            self.norm_freq = norm_freq if norm_freq is not None else config.norm_freq
        else:
            self.input_dim = input_dim if input_dim is not None else 5
            self.output_dim = output_dim if output_dim is not None else 3
            self.grid_size = grid_size if grid_size is not None else 3
            self.is_first = is_first if is_first is not None else True
            self.add_bias = add_bias if add_bias is not None else True
            self.norm_freq = norm_freq if norm_freq is not None else True
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.A, self.K, self.C = 0.9724108095811765, 0.9884401790754128, 0.999449553483052
        self.grid_norm_factor = (torch.arange(grid_size) + 1)
        self.grid_norm_factor = self.grid_norm_factor.reshape(1, 1, grid_size)

        if is_first:
            self.amplitudes = torch.nn.Parameter(torch.empty(output_dim, input_dim, 1).normal_(0, .4) / output_dim  / self.grid_norm_factor)
        else:
            self.amplitudes = torch.nn.Parameter(torch.empty(output_dim, input_dim, 1).uniform_(-1, 1) / output_dim  / self.grid_norm_factor)
        
        grid_phase = torch.arange(1, grid_size + 1).reshape(1, 1, 1, grid_size) / (grid_size + 1)
        self.input_phase = torch.linspace(0, math.pi, input_dim).reshape(1, 1, input_dim, 1).to(self.device)
        phase = grid_phase.to(self.device) + self.input_phase

        if norm_freq:
            self.freq = torch.nn.Parameter(torch.arange(1, grid_size + 1).float().reshape(1, 1, 1, grid_size) / (grid_size + 1)**(1 - is_first))
        else:
            self.freq = torch.nn.Parameter(torch.arange(1, grid_size + 1).float().reshape(1, 1, 1, grid_size))

        for i in range(1, self.grid_size):
            phase = forward_step(phase, i, self.A, self.K, self.C)
        self.register_buffer('phase', phase)
        
        if self.add_bias:
            self.bias  = torch.nn.Parameter(torch.ones(1, output_dim) / output_dim)

    def forward(self, x):
        x_shape = x.shape
        output_shape = x_shape[0:-1] + (self.output_dim,)
        x = torch.reshape(x, (-1, self.input_dim))

        x_reshaped = torch.reshape(x, (x.shape[0], 1, x.shape[1], 1))
        s = torch.sin(x_reshaped * self.freq + self.phase)
        y = torch.einsum('ijkl,jkl->ij', s, self.amplitudes)
        if self.add_bias:
            y += self.bias
        y = torch.reshape(y, output_shape)
        return y