"""Unit tests for the R2-Design-1.5 analysis layer (plan §33).

Covers: counterfactual mask mathematics, bitwise consistency of every
cf path with the model's own forward, rho_func=0 equivalence, fixed-common
exactness, B0 state extraction alignment, H1/H2/HP signals, deterministic
permutation, and the no-test-access guard.
"""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from src.analysis import perf_r2d15_utils as u
from src.models.biaxis_r2 import Model

TEXT_DIM = 13
VISUAL_DIM = 19
NUM_NODES = 17
FACTOR_DIM = 32  # small dim keeps tests fast
HIDDEN_DIM = 64


def _make_cfg(variant: str) -> object:
    sem = variant in ("S", "J")
    func = variant in ("F", "J")
    return OmegaConf.create({
        "model": {
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
            "semantic_refiner": {"enabled": sem, "gate_hidden": 16, "dropout": 0.0},
            "functional_transfer": {"enabled": func, "type_dim": 4, "gate_hidden": 16,
                                    "rho_func_init": 0.01},
        }
    })


def _make_model(variant: str) -> Model:
    info = {
        "input_dim": TEXT_DIM + VISUAL_DIM,
        "num_nodes": NUM_NODES,
        "num_classes": 5,
        "text_dim": TEXT_DIM,
        "visual_dim": VISUAL_DIM,
    }
    return Model(_make_cfg(variant), info).eval()


def _make_x(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(NUM_NODES, TEXT_DIM + VISUAL_DIM, generator=generator)


def _make_edges(seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, NUM_NODES, (2, 40), generator=generator)


# ---------------------------------------------------------------------------
# Counterfactual mask mathematics (plan §33)
# ---------------------------------------------------------------------------


def test_cell_masks_disjoint_and_union() -> None:
    diag = u._FUNC_CELLS["diag_only"]
    offdiag = u._FUNC_CELLS["offdiag_only"]
    assert bool(((diag > 0) & (offdiag > 0)).any()) is False  # disjoint
    assert torch.equal(diag + offdiag, torch.ones(3, 3))  # union = all cells
    assert torch.equal(diag, torch.eye(3))
    full = u._FUNC_CELLS["full"]
    for name, mask in u._FUNC_CELLS.items():
        assert mask.shape == (3, 3)
        assert bool(((mask == 0) | (mask == 1)).all()), f"{name} not 0/1"


def test_source_row_masks_correct() -> None:
    for idx, name in enumerate(("src_C", "src_Pt", "src_Pv")):
        mask = u._FUNC_CELLS[name]
        assert torch.equal(mask[idx], torch.ones(3))  # full row
        for other in range(3):
            if other != idx:
                assert torch.equal(mask[other], torch.zeros(3))


# ---------------------------------------------------------------------------
# Bitwise consistency: cf paths vs the model's own forward (plan §33)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["B0", "F", "S"])
def test_forward_cf_no_cf_matches_model_forward(variant: str) -> None:
    model = _make_model(variant)
    x, ei = _make_x(), _make_edges()
    z_model, _, _, _, _ = model(x, ei)
    z_cf, internals = u.forward_cf(model, x, ei, sem_cf=None, func_cf=None)
    assert torch.equal(z_model, z_cf)
    assert internals["f_star"].shape == (NUM_NODES, 3, FACTOR_DIM)


def test_masked_functional_message_all_ones_bitwise() -> None:
    model = _make_model("F")
    x, ei = _make_x(), _make_edges()
    z, internals = u.forward_cf(model, x, ei)
    f_star, n_block, v_block = internals["f_star"], internals["n_block"], internals["v_block"]
    reference = model._functional_message(f_star, n_block, v_block)
    masked = u.functional_message_masked(model, f_star, n_block, v_block, torch.ones(3, 3))
    assert torch.equal(masked, reference)


def test_func_off_equivalent_rho_func_zero() -> None:
    model = _make_model("F")
    x, ei = _make_x(), _make_edges()
    z_cf, _ = u.forward_cf(model, x, ei, func_cf="func_off")
    saved = model.rho_func.detach().clone()
    with torch.no_grad():
        model.rho_func.zero_()
    z_zeroed, _, _, _, _ = model(x, ei)
    with torch.no_grad():
        model.rho_func.copy_(saved)
    assert torch.equal(z_cf, z_zeroed)


def test_zero_mask_message_exactly_zero() -> None:
    """A zeroed functional message is EXACTLY zero (LN(0)=0 with bias=False,
    message multiplied by mask=0), so func_off via mask equals skipping the
    functional term."""
    model = _make_model("F")
    x, ei = _make_x(), _make_edges()
    _, internals = u.forward_cf(model, x, ei)
    f_star, n_block, v_block = internals["f_star"], internals["n_block"], internals["v_block"]
    zero_msg = u.functional_message_masked(model, f_star, n_block, v_block, torch.zeros(3, 3))
    assert torch.equal(zero_msg, torch.zeros_like(zero_msg))
    # and rho_func * zero == the func_off forward (already tested separately)


# ---------------------------------------------------------------------------
# S counterfactuals (plan §8)
# ---------------------------------------------------------------------------


def test_s_fixed_common_exact_half() -> None:
    model = _make_model("S")
    x = _make_x()
    factors = u.factorize(model, x)
    f0, f_star = u.ownership_states_cf(model, factors, "both_off")
    expected_c0 = 0.5 * (factors["c_t"] + factors["c_v"])
    assert torch.equal(f0[:, 0], expected_c0)
    assert torch.equal(f_star, f0)  # residual off


def test_s_common_only_matches_adaptive_common() -> None:
    model = _make_model("S")
    x = _make_x()
    factors = u.factorize(model, x)
    c0_ref, _w = model.adaptive_common(factors["c_t"], factors["c_v"])
    f0, f_star = u.ownership_states_cf(model, factors, "common_only")
    assert torch.equal(f0[:, 0], c0_ref)
    assert torch.equal(f_star, f0)  # residual off


def test_s_fixed_common_plus_residual_uses_trained_heads() -> None:
    model = _make_model("S")
    x = _make_x()
    factors = u.factorize(model, x)
    f0, f_star = u.ownership_states_cf(model, factors, "fixed_common_residual")
    assert torch.equal(f0[:, 0], 0.5 * (factors["c_t"] + factors["c_v"]))
    # residual heads are applied on the FIXED-common f0
    delta = model.semantic_residual(f0)
    assert torch.equal(f_star, f0 + delta)


# ---------------------------------------------------------------------------
# B0 state extraction (plan §22)
# ---------------------------------------------------------------------------


def test_extract_b0_states_aligned_with_forward() -> None:
    model = _make_model("B0")
    x, ei = _make_x(), _make_edges()
    z_model, _, _, _, _ = model(x, ei)
    states = u.extract_b0_states(model, x, ei)
    assert torch.equal(states["z"], z_model)
    assert states["f_pre"].shape == (NUM_NODES, 3, FACTOR_DIM)
    assert states["n"].shape == (NUM_NODES, 3, FACTOR_DIM)
    assert states["f_out"].shape == (NUM_NODES, 3, FACTOR_DIM)
    # f_out must equal the model's own graph-updated factors
    f_out_ref, _, _, _ = model._graph_update(states["f_pre"], ei, NUM_NODES)
    assert torch.equal(states["f_out"], f_out_ref)


def test_propagation_signals() -> None:
    model = _make_model("B0")
    x, ei = _make_x(), _make_edges()
    states = u.extract_b0_states(model, x, ei)
    h1, h2, hp = u.propagation_signals(model, states["f_pre"], ei, NUM_NODES)
    assert h1.shape == h2.shape == hp.shape == (NUM_NODES, 3, FACTOR_DIM)
    assert torch.isfinite(h1).all() and torch.isfinite(h2).all() and torch.isfinite(hp).all()
    # H1 == the B0 neighbor context (same aggregation)
    assert torch.equal(h1, states["n"])
    # HP == H0 - H1 exactly
    assert torch.equal(hp, states["f_pre"] - h1)


# ---------------------------------------------------------------------------
# Permutation determinism (plan §33)
# ---------------------------------------------------------------------------


def test_fixed_permutation_deterministic_and_valid() -> None:
    p1 = u.fixed_node_permutation(37)
    p2 = u.fixed_node_permutation(37)
    assert torch.equal(p1, p2)
    assert sorted(p1.tolist()) == list(range(37))
    assert not torch.equal(p1, torch.arange(37))  # not identity for this seed


# ---------------------------------------------------------------------------
# Metrics helpers (plan §4.6)
# ---------------------------------------------------------------------------


def test_cka_and_cosine_helpers() -> None:
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(50, 8, generator=generator)
    y = x + 0.1 * torch.randn(50, 8, generator=generator)
    assert 0.0 < u.linear_cka(x, y) <= 1.0 + 1e-6
    assert 0.5 < u.mean_cosine(x, y) <= 1.0
    z_orth = torch.randn(50, 8, generator=generator)
    assert u.linear_cka(x, x) == pytest.approx(1.0, abs=1e-6)
    assert u.mean_cosine(x, x) == pytest.approx(1.0, abs=1e-6)
    assert u.linear_cka(x, z_orth) < 0.3


def test_relative_l2_helper() -> None:
    generator = torch.Generator().manual_seed(4)
    x = torch.randn(20, 4, generator=generator)
    assert u.mean_relative_l2(x, x) == pytest.approx(0.0, abs=1e-6)
    y = 2.0 * x
    assert u.mean_relative_l2(x, y) > 0.5


def test_no_test_access_guard() -> None:
    class _Fake:
        train_idx = torch.tensor([0])
        val_idx = torch.tensor([1])

    u.assert_no_test_access(_Fake())  # no raise
    with pytest.raises(AssertionError):
        u.assert_no_test_access(type("X", (), {"train_idx": None, "val_idx": torch.tensor([0])})())
