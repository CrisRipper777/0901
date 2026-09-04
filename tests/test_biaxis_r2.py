"""Unit tests for the R2-Design-1 model (plan §37-I, 15 required items).

Four variants share ONE implementation (src/models/biaxis_r2.py) and differ
only in semantic_refiner.enabled / functional_transfer.enabled:
    B0 = (false, false), F = (false, true), S = (true, false), J = (true, true).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from src.models.biaxis_r2 import Model

TEXT_DIM = 13
VISUAL_DIM = 19
NUM_NODES = 17
FACTOR_DIM = 128
HIDDEN_DIM = 256

VARIANTS = ("B0", "F", "S", "J")
SEMANTIC_ON = {"S", "J"}
FUNCTIONAL_ON = {"F", "J"}

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _make_cfg(variant: str = "B0", **overrides) -> object:
    model_cfg = {
        "name": "biaxis_r2",
        "hidden_dim": HIDDEN_DIM,
        "factor_dim": FACTOR_DIM,
        "dropout": 0.2,
        "activation": "gelu",
        "norm": "layernorm",
        "lambda_common": 0.02,
        "lambda_orth": 0.01,
        "lambda_recon": 0.3,
        "orth_fallback_batch": 16,
        "full_graph_training": True,
        "edge_chunk_size": None,
        "semantic_refiner": {
            "enabled": variant in SEMANTIC_ON,
            "gate_hidden": 64,
            "dropout": 0.2,
        },
        "functional_transfer": {
            "enabled": variant in FUNCTIONAL_ON,
            "type_dim": 8,
            "gate_hidden": 64,
            "rho_func_init": 0.01,
        },
    }
    model_cfg.update(overrides)
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


def _make_edge_index(num_edges: int = 40, num_nodes: int = NUM_NODES, seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, num_nodes, (2, num_edges), generator=generator)


def _make_model(variant: str = "B0", **overrides) -> Model:
    return Model(_make_cfg(variant, **overrides), _make_data_info())


# ---------------------------------------------------------------------------
# (1) forward shapes for all four variants + (2) aux loss finite
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", VARIANTS)
def test_forward_shapes_all_variants(variant: str) -> None:
    model = _make_model(variant)
    model.train()
    x = _make_x()
    z, second, third, aux_loss, aux_info = model(x, _make_edge_index())
    assert z.shape == (NUM_NODES, HIDDEN_DIM)
    assert second is None and third is None
    assert aux_loss.ndim == 0 and torch.isfinite(aux_loss)
    assert isinstance(aux_info, dict)
    for key in ("p0_common_loss", "p0_orth_loss", "p0_recon_loss", "p0_common_sim",
                "p0_private_sim", "p0_c_norm", "p0_pt_norm", "p0_pv_norm",
                "p0_cp_overlap_t", "p0_cp_overlap_v"):
        assert key in aux_info and aux_info[key].numel() == 1


@pytest.mark.parametrize("variant", VARIANTS)
def test_forward_accepts_none_edge_index(variant: str) -> None:
    model = _make_model(variant).eval()
    z, _, _, _, _ = model(_make_x(), None)
    assert z.shape == (NUM_NODES, HIDDEN_DIM)
    assert torch.isfinite(z).all()


# ---------------------------------------------------------------------------
# (3) isolated / no-edge graph: B0 graph residual exactly zero
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", VARIANTS)
def test_isolated_graph_residual_exactly_zero(variant: str) -> None:
    """N^a = 0 -> V_a(0) = 0 (bias=False) -> LN(0) = 0 (bias=False) ->
    F' = F* EXACTLY (plan §4.4). Must hold for every variant, including the
    functional path (m = g * V(0) = 0)."""
    model = _make_model(variant).eval()
    x = _make_x()
    x_t, x_v = model._split_modalities(x)
    factors = model.factorizer(x_t, x_v)
    f0, f_star, w = model._ownership_states(factors)
    empty_edges = torch.empty(2, 0, dtype=torch.long)
    f_out, _, _, _ = model._graph_update(f_star, empty_edges, NUM_NODES)
    assert torch.equal(f_out, f_star)


@pytest.mark.parametrize("variant", VARIANTS)
def test_isolated_nodes_inside_graph_unchanged(variant: str) -> None:
    """Nodes with in-degree 0 inside a real graph must also keep F' = F*."""
    model = _make_model(variant).eval()
    x = _make_x()
    edge_index = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.long)
    x_t, x_v = model._split_modalities(x)
    factors = model.factorizer(x_t, x_v)
    _, f_star, _ = model._ownership_states(factors)
    f_out, _, _, _ = model._graph_update(f_star, edge_index, NUM_NODES)
    # nodes 8..16 have in-degree 0 (edges only touch dst 4..7)
    assert torch.equal(f_out[8:], f_star[8:])


# ---------------------------------------------------------------------------
# (4)/(5) variant isolation: disabled paths are not instantiated / used
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["B0", "F"])
def test_semantic_disabled_no_semantic_modules(variant: str) -> None:
    model = _make_model(variant)
    assert not hasattr(model, "adaptive_common")
    assert not hasattr(model, "semantic_residual")


@pytest.mark.parametrize("variant", ["B0", "S"])
def test_functional_disabled_no_functional_modules(variant: str) -> None:
    model = _make_model(variant)
    for attr in ("func_scorer", "src_type_emb", "tgt_type_emb", "msg_norm_func", "rho_func"):
        assert not hasattr(model, attr), f"{attr} must not exist when functional is off"


# ---------------------------------------------------------------------------
# (6)/(7) zero-init degeneracies
# ---------------------------------------------------------------------------


def test_common_gate_zero_init_exact_half() -> None:
    model = _make_model("S")
    c_t = _rand_c(seed=1)
    c_v = _rand_c(seed=2)
    _, w = model.adaptive_common(c_t, c_v)
    assert torch.equal(w, torch.full((NUM_NODES, 2), 0.5))


def test_semantic_residual_heads_zero_init_exact_zero() -> None:
    model = _make_model("S")
    f0 = torch.stack([_rand_c(seed=s) for s in (1, 2, 3)], dim=1)
    delta = model.semantic_residual(f0)
    assert torch.equal(delta, torch.zeros_like(delta))


def _rand_c(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(NUM_NODES, FACTOR_DIM, generator=generator)


# ---------------------------------------------------------------------------
# (8)/(9) functional scorer finite + gate range (via model diagnostics)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["F", "J"])
def test_functional_path_finite_and_gate_range(variant: str) -> None:
    model = _make_model(variant).eval()
    x = _make_x()
    edge_index = _make_edge_index()
    z, _, _, _, _ = model(x, edge_index)
    assert torch.isfinite(z).all()
    diag = model.compute_r2_diagnostics(x, edge_index)
    gate = diag["functional"]["gate_matrix"]
    assert gate["rows"] == ["src_C", "src_Pt", "src_Pv"]
    assert gate["cols"] == ["tgt_C", "tgt_Pt", "tgt_Pv"]
    for key in ("mean", "p05", "p50", "p95"):
        for row in gate[key]:
            for v in row:
                assert 0.0 <= v <= 1.0, f"{key} cell out of range: {v}"
    contrib = diag["functional"]["contribution_matrix"]
    values = contrib["values"]  # rows = source, cols = target
    for b in range(3):  # each TARGET column sums to 1 (plan §18.2)
        assert sum(values[a][b] for a in range(3)) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# (10) rho_func init = 0.01 exactly; rho_base init = 0.5 exactly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["F", "J"])
def test_rho_func_init_exact(variant: str) -> None:
    model = _make_model(variant)
    assert torch.equal(model.rho_func, torch.full((3,), 0.01))


@pytest.mark.parametrize("variant", VARIANTS)
def test_rho_base_init_half(variant: str) -> None:
    model = _make_model(variant)
    assert torch.equal(torch.sigmoid(model.raw_rho_base), torch.full((3,), 0.5))


# ---------------------------------------------------------------------------
# (11) source/target factor order = [C, Pt, Pv]
# ---------------------------------------------------------------------------


def test_factor_order_cptv_semantic_off() -> None:
    model = _make_model("B0").eval()
    x = _make_x()
    x_t, x_v = model._split_modalities(x)
    factors = model.factorizer(x_t, x_v)
    f0, f_star, w = model._ownership_states(factors)
    assert w is None
    assert f_star is f0 or torch.equal(f_star, f0)
    assert torch.equal(f0[:, 0], factors["c"])  # C = (c_t + c_v) / 2
    assert torch.equal(f0[:, 1], factors["p_t"])
    assert torch.equal(f0[:, 2], factors["p_v"])


def test_factor_order_cptv_semantic_on() -> None:
    model = _make_model("S").eval()
    x = _make_x()
    x_t, x_v = model._split_modalities(x)
    factors = model.factorizer(x_t, x_v)
    f0, f_star, w = model._ownership_states(factors)
    assert w is not None
    assert torch.equal(f0[:, 0], 0.5 * (factors["c_t"] + factors["c_v"]))  # step-0 C0
    assert torch.equal(f0[:, 1], factors["p_t"])
    assert torch.equal(f0[:, 2], factors["p_v"])
    assert torch.equal(f_star, f0)  # zero-init heads -> F* == F0 at step 0


# ---------------------------------------------------------------------------
# (12) permutation of edge order does not change eval beyond tolerance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", VARIANTS)
def test_edge_permutation_invariance(variant: str) -> None:
    model = _make_model(variant).eval()
    x = _make_x()
    edge_index = _make_edge_index(seed=1)
    generator = torch.Generator().manual_seed(7)
    perm = torch.randperm(edge_index.size(1), generator=generator)
    z_a, _, _, _, _ = model(x, edge_index)
    z_b, _, _, _, _ = model(x, edge_index[:, perm])
    assert torch.allclose(z_a, z_b, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# (13) chunked neighbor_mean path identical (within float tolerance)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", VARIANTS)
def test_chunked_neighbor_mean_matches(variant: str) -> None:
    model_chunked = _make_model(variant, edge_chunk_size=3).eval()
    model_plain = _make_model(variant).eval()
    model_plain.load_state_dict(model_chunked.state_dict())
    x = _make_x()
    edge_index = _make_edge_index(seed=1)
    z_chunked, _, _, _, _ = model_chunked(x, edge_index)
    z_plain, _, _, _, _ = model_plain(x, edge_index)
    assert torch.allclose(z_chunked, z_plain, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# (14) old biaxis_final regression smoke unchanged
# ---------------------------------------------------------------------------


def _make_a0_cfg() -> object:
    return OmegaConf.create({
        "model": {
            "name": "biaxis_final",
            "hidden_dim": HIDDEN_DIM,
            "factor_dim": FACTOR_DIM,
            "dropout": 0.2,
            "activation": "gelu",
            "norm": "layernorm",
            "lambda_common": 0.02,
            "lambda_orth": 0.01,
            "lambda_recon": 0.3,
            "orth_fallback_batch": 16,
            "full_graph_training": True,
            "p1": {
                "factor_aware": True,
                "num_relations": 4,
                "relation_dim": 32,
                "relation_temperature": 0.5,
                "selector_hidden_dim": 64,
                "selector_input_norm": None,
                "budget_hidden_dim": 64,
                "use_graph_budget": True,
                "budget_shared": False,
                "eps": 1.0e-8,
                "relation_balance_weight": 0.0,
                "alpha_entropy_weight": 0.0,
                "budget_reg_weight": 0.0,
                "edge_chunk_size": 500000,
            },
            "p2": {
                "mode": "null_softmax",
                "score_hidden_dim": 64,
                "epsilon": 0.2,
                "tau_base": 1.0,
                "sinkhorn_iters": 10,
                "null_prior": 0.5,
                "null_score_init": 0.0,
                "deterministic": False,
                "detach_capacity_prior": True,
                "detach_relation_confidence": True,
                "eps": 1.0e-8,
            },
            "p3": {
                "operator_mode": "full_interaction",
                "operator_reg_weight": 0.0,
                "interaction_reg_weight": 0.0,
                "memory_checkpoint": False,
            },
        }
    })


def test_a0_regression_smoke_unchanged() -> None:
    from src.models.biaxis_final import Model as A0Model

    model = A0Model(_make_a0_cfg(), _make_data_info()).eval()
    x = _make_x(seed=3)
    edge_index = _make_edge_index(seed=4)
    z, second, third, aux_loss, aux_info = model(x, edge_index)
    assert z.shape == (NUM_NODES, HIDDEN_DIM)
    assert second is None and third is None
    assert torch.isfinite(z).all()


# ---------------------------------------------------------------------------
# (15) no Test logic inside the R2 model sources
# ---------------------------------------------------------------------------


def test_no_test_logic_in_r2_sources() -> None:
    sources = [
        PROJECT_ROOT / "src" / "models" / "biaxis_r2.py",
        PROJECT_ROOT / "src" / "models" / "biaxis_r2_components.py",
    ]
    forbidden = ("test_idx", "test_acc", "test_macro", "data.y[", "test_mask")
    for path in sources:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path.name} contains forbidden token {token!r}"


# ---------------------------------------------------------------------------
# inference interface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", VARIANTS)
def test_inference_returns_cpu_z(variant: str) -> None:
    model = _make_model(variant).eval()
    z = model.inference(_make_x(), _make_edge_index(), batch_size=8)
    assert z.shape == (NUM_NODES, HIDDEN_DIM)
    assert z.device.type == "cpu"
    assert torch.isfinite(z).all()
