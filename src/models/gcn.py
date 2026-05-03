from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

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

        dims = [input_dim] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList([GCNConv(dims[i], dims[i + 1]) for i in range(num_layers)])
        self.norms = nn.ModuleList([make_norm(norm_name, hidden_dim) for _ in range(max(num_layers - 1, 0))])
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden_dim

    def forward(self, x, edge_index):
        for layer_idx, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if layer_idx != len(self.convs) - 1:
                x = self.norms[layer_idx](x)
                x = self.activation(x)
                x = self.dropout(x)
        aux_loss = x.new_tensor(0.0)
        return x, None, None, aux_loss, {}
