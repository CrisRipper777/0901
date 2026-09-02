from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from src.models.biaxis_p0 import Model

TEXT_DIM = 13
VISUAL_DIM = 19
NUM_NODES = 17
FACTOR_DIM = 128
HIDDEN_DIM = 256


def _make_cfg(**model_overrides) -> OmegaConf.ListConfig | object:
    model_cfg = {
        "name": "biaxis_p0",
        "hidden_dim": HIDDEN_DIM,
        "factor_dim": FACTOR_DIM,
        "dropout": 0.2,
        "activation": "gelu",
        "norm": "layernorm",
        "lambda_common": 0.1,
        "lambda_orth": 0.01,
        "lambda_recon": 0.1,
        "full_graph_training": False,
    }
    model_cfg.update(model_overrides)
    return OmegaConf.create({"model": model_cfg})


def _make_data_info(**overrides) -> dict:
    info = {
        "input_dim": TEXT_DIM + VISUAL_DIM,
        "num_nodes": NUM_NODES,
        "num_classes": 5,
        "text_dim": TEXT_DIM,
        "visual_dim": VISUAL_DIM,
    }
    info.update(overrides)
    return info


def _make_x(num_nodes: int = NUM_NODES, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(num_nodes, TEXT_DIM + VISUAL_DIM, generator=generator)


def _make_edge_index(num_edges: int = 5, num_nodes: int = NUM_NODES, seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, num_nodes, (2, num_edges), generator=generator)


def _make_model(**overrides) -> Model:
    return Model(_make_cfg(**overrides), _make_data_info())


# ---------------------------------------------------------------------------
# Shape / interface
# ---------------------------------------------------------------------------


def test_forward_returns_framework_tuple_and_shapes() -> None:
    model = _make_model()
    model.train()
    x = _make_x()
    z, second, third, aux_loss, aux_info = model(x, _make_edge_index())
    assert second is None
    assert third is None
    assert z.shape == (NUM_NODES, HIDDEN_DIM)
    assert aux_loss.ndim == 0
    assert torch.isfinite(aux_loss)
    assert isinstance(aux_info, dict)
    expected_keys = (
        "p0_common_loss",
        "p0_orth_loss",
        "p0_recon_loss",
        "p0_common_sim",
        "p0_private_sim",
        "p0_c_norm",
        "p0_pt_norm",
        "p0_pv_norm",
        "p0_cp_overlap_t",
        "p0_cp_overlap_v",
    )
    for key in expected_keys:
        assert key in aux_info, f"missing aux_info key {key}"
        assert torch.is_tensor(aux_info[key]) and aux_info[key].numel() == 1


def test_forward_accepts_none_edge_index() -> None:
    model = _make_model()
    model.train()
    z, _, _, _, _ = model(_make_x(), None)
    assert z.shape == (NUM_NODES, HIDDEN_DIM)


def test_factor_shapes() -> None:
    model = _make_model()
    factors = model.encode_factors(_make_x(), edge_index=None)
    for key in ("c", "c_t", "c_v", "p_t", "p_v"):
        assert factors[key].shape == (NUM_NODES, FACTOR_DIM)
    assert factors["z_local"].shape == (NUM_NODES, HIDDEN_DIM)


def test_modality_split_order() -> None:
    model = _make_model()
    x = _make_x()
    x_t, x_v = model._split_modalities(x)
    assert x_t.shape == (NUM_NODES, TEXT_DIM)
    assert x_v.shape == (NUM_NODES, VISUAL_DIM)
    assert torch.equal(x_t, x[:, :TEXT_DIM])
    assert torch.equal(x_v, x[:, TEXT_DIM : TEXT_DIM + VISUAL_DIM])


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def test_aux_loss_finite_over_random_batches() -> None:
    model = _make_model()
    model.train()
    for seed in range(3):
        _, _, _, aux_loss, aux_info = model(_make_x(seed=seed), None)
        assert torch.isfinite(aux_loss), f"non-finite aux_loss at seed={seed}"
        for key, value in aux_info.items():
            assert torch.isfinite(value), f"non-finite {key} at seed={seed}"


def test_aux_info_similarity_bounds() -> None:
    model = _make_model()
    model.train()
    _, _, _, _, aux_info = model(_make_x(), None)
    assert -1.0 - 1e-5 <= float(aux_info["p0_common_sim"]) <= 1.0 + 1e-5
    assert -1.0 - 1e-5 <= float(aux_info["p0_private_sim"]) <= 1.0 + 1e-5


def test_orth_loss_fallback_small_batch() -> None:
    model = _make_model()
    model.train()
    x = torch.randn(8, TEXT_DIM + VISUAL_DIM)  # batch < orth_fallback_batch
    _, _, _, aux_loss, aux_info = model(x, None)
    assert torch.isfinite(aux_loss)
    assert torch.isfinite(aux_info["p0_cp_overlap_t"])
    assert torch.isfinite(aux_info["p0_cp_overlap_v"])


# ---------------------------------------------------------------------------
# Gradients / parameter sharing
# ---------------------------------------------------------------------------


def test_gradients_flow_to_all_components() -> None:
    model = _make_model()
    model.train()
    x = _make_x()
    z, _, _, aux_loss, _ = model(x, _make_edge_index())
    (aux_loss + z.square().mean()).backward()
    components = [
        model.factorizer.text_projector,
        model.factorizer.visual_projector,
        model.factorizer.common_encoder,
        model.factorizer.private_text_encoder,
        model.factorizer.private_visual_encoder,
        model.recon_text_head,
        model.recon_visual_head,
        model.fusion,
    ]
    for component in components:
        params = list(component.parameters())
        assert params, "component has no parameters"
        assert all(p.grad is not None for p in params), f"missing gradient in {component}"
        assert all(torch.isfinite(p.grad).all() for p in params), f"non-finite gradient in {component}"


def test_common_encoder_parameter_sharing() -> None:
    model = _make_model()
    # The shared MLP must be registered exactly once: its state_dict key count
    # (2 Linears + 1 LayerNorm = 6 entries) must appear once in the model.
    common_keys = [key for key in model.state_dict() if key.startswith("factorizer.common_encoder.")]
    assert len(common_keys) == len(model.factorizer.common_encoder.state_dict())
    assert model.factorizer.common_encoder is not model.factorizer.private_text_encoder
    assert model.factorizer.private_text_encoder is not model.factorizer.private_visual_encoder


# ---------------------------------------------------------------------------
# Topology independence
# ---------------------------------------------------------------------------


def test_factorizer_is_topology_free() -> None:
    model = _make_model()
    x = _make_x()
    edge_index_a = _make_edge_index(num_edges=2, seed=1)
    edge_index_b = _make_edge_index(num_edges=50, seed=2)
    factors_a = model.encode_factors(x, edge_index=edge_index_a)
    factors_b = model.encode_factors(x, edge_index=edge_index_b)
    factors_c = model.encode_factors(x, edge_index=None)
    for key in factors_a:
        assert torch.equal(factors_a[key], factors_b[key]), f"factor {key} depends on edge_index"
        assert torch.equal(factors_a[key], factors_c[key]), f"factor {key} depends on edge_index"
    z_a, _, _, _, _ = model(x, edge_index_a)
    z_b, _, _, _, _ = model(x, edge_index_b)
    assert torch.equal(z_a, z_b), "forward output depends on edge_index"


# ---------------------------------------------------------------------------
# Chunked equivalence
# ---------------------------------------------------------------------------


def test_encode_factors_chunked_matches_full() -> None:
    model = _make_model()
    x = _make_x()
    full = model.encode_factors(x, edge_index=None)
    chunked = model.encode_factors(x, edge_index=None, batch_size=7)
    for key in full:
        # GEMM accumulation order differs across batch shapes -> ~1e-6 noise.
        assert torch.allclose(full[key], chunked[key], atol=1e-4), f"chunked {key} differs from full"


def test_inference_matches_full_forward() -> None:
    model = _make_model()
    model.eval()
    x = _make_x()
    z_full, _, _, _, _ = model(x, None)
    z_inf = model.inference(x, None, device=torch.device("cpu"), batch_size=7)
    assert z_inf.shape == (NUM_NODES, HIDDEN_DIM)
    assert torch.allclose(z_full, z_inf, atol=1e-4)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_modality_split_asserts_on_short_features() -> None:
    model = _make_model()
    x_short = torch.randn(NUM_NODES, TEXT_DIM + VISUAL_DIM - 1)
    with pytest.raises(AssertionError):
        model(x_short, None)


def test_missing_modality_dims_rejected() -> None:
    with pytest.raises(ValueError):
        Model(_make_cfg(), _make_data_info(text_dim=0, visual_dim=VISUAL_DIM, input_dim=VISUAL_DIM))
    with pytest.raises(ValueError):
        Model(_make_cfg(), _make_data_info(text_dim=TEXT_DIM, visual_dim=0, input_dim=TEXT_DIM))
