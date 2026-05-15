from __future__ import annotations

import copy

import torch
from omegaconf import ListConfig


def clone_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def load_state_dict_cpu(module: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    module.load_state_dict(copy.deepcopy(state_dict))


def resolve_num_neighbors(cfg) -> list[int]:
    num_layers = int(cfg.model.get("num_layers", 1))
    if num_layers < 1:
        raise ValueError(f"model.num_layers must be >= 1, got {num_layers}")

    raw_neighbors = cfg.task.get("num_neighbors", -1)
    if isinstance(raw_neighbors, str):
        raw_neighbors = raw_neighbors.strip()
        if raw_neighbors.startswith("[") and raw_neighbors.endswith("]"):
            values = [int(item.strip()) for item in raw_neighbors[1:-1].split(",") if item.strip()]
        else:
            values = [int(raw_neighbors)]
    elif isinstance(raw_neighbors, (list, tuple, ListConfig)):
        values = [int(value) for value in raw_neighbors]
    else:
        values = [int(raw_neighbors)]

    if not values:
        raise ValueError("task.num_neighbors must contain at least one value")
    if len(values) >= num_layers:
        return values[:num_layers]
    return values + [values[-1]] * (num_layers - len(values))
