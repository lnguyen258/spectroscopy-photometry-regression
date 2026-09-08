from typing import Optional, List

import torch
from torch import nn

from src.config import MLP_Config

ACTIVATIONS = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'leaky_relu': nn.LeakyReLU,
    'silu': nn.SiLU,
    'gelu': nn.GELU,
}


class MLP(nn.Module):
    def __init__(
            self,
            config: Optional[MLP_Config] = None,
            input_dim: Optional[int] = None,
            output_dim: Optional[int] = None,
            hidden_dims: Optional[List[int]] = None,
            dropout: Optional[float] = None,
            activation: Optional[str] = None,
    ):
        super(MLP, self).__init__()

        if config is not None:
            self.input_dim = input_dim if input_dim is not None else config.input_dim
            self.output_dim = output_dim if output_dim is not None else config.output_dim
            self.hidden_dims = hidden_dims if hidden_dims is not None else config.hidden_dims
            self.dropout = dropout if dropout is not None else config.dropout
            self.activation = activation if activation is not None else config.activation
        else:
            self.input_dim = input_dim if input_dim is not None else 7
            self.output_dim = output_dim if output_dim is not None else 1
            self.hidden_dims = hidden_dims if hidden_dims is not None else [32, 16]
            self.dropout = dropout if dropout is not None else 0.1
            self.activation = activation if activation is not None else 'relu'

        act_cls = ACTIVATIONS[self.activation]
        layer_dims = [self.input_dim] + self.hidden_dims + [self.output_dim]

        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                layers.append(nn.BatchNorm1d(layer_dims[i + 1]))
                layers.append(act_cls())
                if self.dropout > 0:
                    layers.append(nn.Dropout(self.dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
