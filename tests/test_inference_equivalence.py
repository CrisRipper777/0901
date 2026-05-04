from __future__ import annotations

import torch

from src.models import gcn, mlp, sage


class _CfgNode(dict):
    def __getattr__(self, name: str):
        return self[name]


def _cfg() -> _CfgNode:
    return _CfgNode(
        model=_CfgNode(
            hidden_dim=5,
            num_layers=2,
            dropout=0.0,
            activation="relu",
            norm="none",
        )
    )


def _small_graph() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2],
            [1.3, 1.4, 1.5, 1.6],
            [1.7, 1.8, 1.9, 2.0],
            [2.1, 2.2, 2.3, 2.4],
        ],
        dtype=torch.float32,
    )
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 5],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 5, 0],
        ],
        dtype=torch.long,
    )
    return x, edge_index


def _build_model(model_cls):
    torch.manual_seed(123)
    model = model_cls(_cfg(), {"input_dim": 4})
    model.eval()
    return model


def _compare_full_and_inference(model, x: torch.Tensor, edge_index: torch.Tensor | None):
    with torch.no_grad():
        full = model(x, edge_index)[0].detach().cpu()
        inferred = model.inference(x, edge_index, device=torch.device("cpu"), batch_size=2)
    return full, inferred, float((full - inferred).abs().max().item())


def test_mlp_inference_matches_full_batch_forward() -> None:
    x, _ = _small_graph()
    model = _build_model(mlp.Model)

    full, inferred, max_abs_diff = _compare_full_and_inference(model, x, None)

    assert inferred.shape == full.shape
    assert max_abs_diff < 1e-4


def test_sage_layerwise_inference_matches_full_batch_forward() -> None:
    x, edge_index = _small_graph()
    model = _build_model(sage.Model)

    full, inferred, max_abs_diff = _compare_full_and_inference(model, x, edge_index)

    assert inferred.shape == full.shape
    assert max_abs_diff < 1e-4


def test_gcn_layerwise_inference_matches_full_batch_forward() -> None:
    x, edge_index = _small_graph()
    model = _build_model(gcn.Model)

    full, inferred, max_abs_diff = _compare_full_and_inference(model, x, edge_index)

    assert inferred.shape == full.shape
    assert max_abs_diff < 1e-4
