from __future__ import annotations

import torch
import torch.nn as nn

from .common import get_activation, make_norm


class Model(nn.Module):
    def __init__(self, cfg, data_info):
        super().__init__()
        input_dim = int(data_info["input_dim"])
        hidden_dim = int(cfg.model.hidden_dim)
        num_layers = int(cfg.model.num_layers)
        dropout = float(cfg.model.dropout)
        activation = str(cfg.model.get("activation", "relu"))
        norm_name = cfg.model.get("norm", "batchnorm")

        layers: list[nn.Module] = []
        dims = [input_dim] + [hidden_dim] * num_layers
        for layer_idx in range(num_layers):
            layers.append(nn.Linear(dims[layer_idx], dims[layer_idx + 1]))
            if layer_idx != num_layers - 1:
                layers.append(make_norm(norm_name, dims[layer_idx + 1]))
                layers.append(get_activation(activation))
                layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*layers)
        self.out_dim = hidden_dim

    def forward(self, x, edge_index=None):
        z = self.encoder(x)
        aux_loss = z.new_tensor(0.0)
        return z, None, None, aux_loss, {}
