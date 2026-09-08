from typing import Dict, List
from dataclasses import dataclass

@dataclass
class SineKAN_Config:
    input_dim: int
    output_dim: int
    hidden_dim: List[int]
    grid_size: int
    is_first: bool
    add_bias: bool
    norm_freq: bool

    @classmethod
    def from_dict(cls, d: Dict):
        return cls(**d)


@dataclass
class MLP_Config:
    input_dim: int
    output_dim: int
    hidden_dims: List[int]
    dropout: float
    activation: str

    @classmethod
    def from_dict(cls, d: Dict):
        return cls(**d)