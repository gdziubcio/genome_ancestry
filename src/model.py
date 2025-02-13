# src/model.py

import torch
import torch.nn as nn
from typing import List

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: List[int] = None,
        dropout_rate: float = 0.5,
    ) -> None:
        super(MLP, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128]
        
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
        

        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
