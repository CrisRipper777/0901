"""Unit tests for the R2-Design-1.6 dual-parent layer (plan §51, 12 items).

Bitwise extraction consistency for BOTH parents, frozen-adapter plumbing,
matched-initialization save/load, propagation signals, permutation
determinism and the no-test guard.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from src.analysis import perf_r2d16_utils as u
from src.analysis.perf_r2d15_utils import propagation_signals
from src.models.biaxis_r2 import Model as B0Model

TEXT_DIM = 13
VISUAL_DIM = 19
NUM_NODES = 17
FACTOR_DIM = 32
HIDDEN_DIM = 64
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _make_b0_model() -> B0Model:
    cfg = OmegaConf.create({
        "model": {
            "name": "biaxis_r2", "hidden_dim": HIDDEN_DIM, "factor_dim": FACTOR_DIM,
            "dropout": 0.2, "activation": "gelu", "norm": "layernorm",
            "lambda_common": 0.02, "lambda_orth": 0.01, "lambda_recon": 0.3,
            "orth_fallback_batch": 16, "full_graph_training": True,
            "edge_chunk_size": None,
            "semantic_refiner": {"enabled": False, "gate_hidden": 16, "dropout": 0.0},
            "functional_transfer": {"enabled": False, "type_dim": 4, "gate_hidden": 16,
                                    "rho_func_init": 0.01},
        }
    })
    info = {"input_dim": TEXT_DIM + VISUAL_DIM, "num_nodes": NUM_NODES,
            "num_classes": 5, "text_dim": TEXT_DIM, "visual_dim": VISUAL_DIM}
    return B0Model(cfg, info).eval()


def _make_a0_model():
    from src.models.biaxis_perf_r1 import Model as R1Model

    cfg = OmegaConf.create({
        "model": {
            "name": "biaxis_perf_r1", "hidden_dim": HIDDEN_DIM, "factor_dim": FACTOR_DIM,
            "dropout": 0.2, "activation": "gelu", "norm": "layernorm",
            "lambda_common": 0.02, "lambda_orth": 0.01, "lambda_recon": 0.3,
            "orth_fallback_batch": 16, "full_graph_training": True,
            "p1": {
                "factor_aware": True, "num_relations": 4, "relation_dim": 32,
                "relation_temperature": 0.5, "selector_hidden_dim": 64,
                "selector_input_norm": None, "budget_hidden_dim": 64,
                "use_graph_budget": True, "budget_shared": False, "eps": 1.0e-8,
                "relation_balance_weight": 0.0, "alpha_entropy_weight": 0.0,
                "budget_reg_weight": 0.0, "edge_chunk_size": None,
            },
            "p2": {
                "mode": "null_softmax", "score_hidden_dim": 64, "epsilon": 0.2,
                "tau_base": 1.0, "sinkhorn_iters": 10, "null_prior": 0.5,
                "null_score_init": 0.0, "deterministic": False,
                "detach_capacity_prior": True, "detach_relation_confidence": True,
                "eps": 1.0e-8,
            },
            "p3": {
                "operator_mode": "full_interaction", "operator_reg_weight": 0.0,
                "interaction_reg_weight": 0.0, "memory_checkpoint": False,
            },
            "r1": {"mode": "baseline", "router_mode": "base"},
        }
    })
    info = {"input_dim": TEXT_DIM + VISUAL_DIM, "num_nodes": NUM_NODES,
            "num_classes": 5, "text_dim": TEXT_DIM, "visual_dim": VISUAL_DIM}
    return R1Model(cfg, info).eval()


def _make_x(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(NUM_NODES, TEXT_DIM + VISUAL_DIM, generator=generator)


def _make_edges(seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, NUM_NODES, (2, 40), generator=generator)


def _wrap(parent: str, model, x, ei) -> u.ParentSetup:
    data = SimpleNamespace(x=x, edge_index=ei)
    head = torch.nn.Linear(model.out_dim, 5)
    return u.ParentSetup(parent, "Movies", 42, None, data, model, head, torch.device("cpu"))


# ---------------------------------------------------------------------------
# (1) state extraction reproduces each parent's own forward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("parent", ["A0", "B0"])
def test_extraction_reproduces_parent_forward(parent: str) -> None:
    model = _make_a0_model() if parent == "A0" else _make_b0_model()
    x, ei = _make_x(), _make_edges()
    setup = _wrap(parent, model, x, ei)
    z_model = u.parent_forward_z(setup, x, ei)
    states = u.extract_parent_states(setup, x, ei)
    assert torch.equal(states["z"], z_model)
    assert states["f_pre"].shape == (NUM_NODES, 3, FACTOR_DIM)
    assert states["n"].shape == (NUM_NODES, 3, FACTOR_DIM)
    assert states["f_out"].shape == (NUM_NODES, 3, FACTOR_DIM)
    assert torch.equal(states["base_update"], states["f_out"] - states["f_pre"])


# ---------------------------------------------------------------------------
# (2) HEAD frozen path reproduces parent z
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("parent", ["A0", "B0"])
def test_head_frozen_path_reproduces_parent_z(parent: str) -> None:
    model = _make_a0_model() if parent == "A0" else _make_b0_model()
    x, ei = _make_x(), _make_edges()
    setup = _wrap(parent, model, x, ei)
    states = u.extract_parent_states(setup, x, ei)
    z_head = u.adapter_z(setup, states["f_out"], torch.zeros_like(states["f_out"]))
    assert torch.equal(z_head, states["z"])


# ---------------------------------------------------------------------------
# (6) fixed parent has zero parameter grads / (7) adapter gradients finite
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("parent", ["A0", "B0"])
def test_frozen_parent_zero_grads_adapter_finite(parent: str) -> None:
    from src.models.biaxis_r2d16_adapters import build_interaction_adapter

    model = _make_a0_model() if parent == "A0" else _make_b0_model()
    x, ei = _make_x(), _make_edges()
    setup = _wrap(parent, model, x, ei)
    for p in model.parameters():
        p.requires_grad_(False)
    states = u.extract_parent_states(setup, x, ei)
    adapter = build_interaction_adapter("PRODDIFF", FACTOR_DIM)
    with torch.no_grad():
        for p in adapter.parameters():
            p.add_(0.05 * torch.randn_like(p))
    delta = adapter(states["f_pre"], states["n"])
    z = u.adapter_z(setup, states["f_out"], delta)
    loss = z[:5].sum()
    loss.backward()
    assert all(p.grad is None for p in model.parameters())
    assert any(p.grad is not None for p in adapter.parameters())
    assert all(torch.isfinite(p.grad).all() for p in adapter.parameters() if p.grad is not None)


# ---------------------------------------------------------------------------
# (3)/(4)/(5) adapter zero-init / parameter matching / FiLM zero-init
# ---------------------------------------------------------------------------


def test_interaction_adapters_zero_init_exact() -> None:
    from src.models.biaxis_r2d16_adapters import build_interaction_adapter

    f = torch.randn(NUM_NODES, 3, FACTOR_DIM)
    n = torch.randn(NUM_NODES, 3, FACTOR_DIM)
    for name in ("CONCAT", "PRODDIFF", "FiLM"):
        adapter = build_interaction_adapter(name, FACTOR_DIM)
        assert torch.equal(adapter(f, n), torch.zeros_like(adapter(f, n))), name


def test_concat_proddiff_parameter_matched() -> None:
    from src.models.biaxis_r2d16_adapters import build_interaction_adapter

    c = build_interaction_adapter("CONCAT", FACTOR_DIM)
    p = build_interaction_adapter("PRODDIFF", FACTOR_DIM)
    assert sum(x.numel() for x in c.parameters()) == sum(x.numel() for x in p.parameters())
    for (nc, tc), (np_, tp) in zip(c.named_parameters(), p.named_parameters()):
        assert nc == np_ and tc.shape == tp.shape


def test_film_head_zero_init() -> None:
    from src.models.biaxis_r2d16_adapters import build_interaction_adapter

    adapter = build_interaction_adapter("FiLM", FACTOR_DIM)
    final = adapter.net[-1]
    assert torch.equal(final.weight, torch.zeros_like(final.weight))
    assert torch.equal(final.bias, torch.zeros_like(final.bias))


def test_semantic_residual_zero_init_and_block() -> None:
    from src.models.biaxis_r2d16_adapters import SemanticResidualAdapter

    adapter = SemanticResidualAdapter(FACTOR_DIM, hidden=32)
    f0 = torch.randn(NUM_NODES, 3, FACTOR_DIM)
    assert torch.equal(adapter(f0), f0)  # zero-init -> F* == F0 exactly
    block = adapter.interaction_block(f0)
    assert block.shape == (NUM_NODES, 6 * FACTOR_DIM)
    assert torch.equal(
        block[:, :2 * FACTOR_DIM],
        torch.cat([f0[:, 0] * f0[:, 1], f0[:, 0] * f0[:, 2]], dim=-1),
    )


# ---------------------------------------------------------------------------
# (8) classifier init save/reload bitwise / (9) matched t=0 states identical
# ---------------------------------------------------------------------------


def test_classifier_init_save_reload_bitwise(tmp_path) -> None:
    head = u.make_classifier_init(20260904, HIDDEN_DIM, 5, torch.device("cpu"))
    path = tmp_path / "head_init.pt"
    u.save_state(path, head)
    head2 = torch.nn.Linear(HIDDEN_DIM, 5)
    u.load_state_into(path, head2)
    for key in head.state_dict():
        assert torch.equal(head.state_dict()[key], head2.state_dict()[key])


def test_matched_schedule_states_identical(tmp_path) -> None:
    from src.models.biaxis_r2d16_adapters import build_interaction_adapter

    model = _make_b0_model()
    adapter = build_interaction_adapter("PRODDIFF", FACTOR_DIM)
    head = u.make_classifier_init(7, HIDDEN_DIM, 5, torch.device("cpu"))
    for name, module in (("parent", model), ("adapter", adapter), ("head", head)):
        u.save_state(tmp_path / f"{name}.pt", module)
    model2 = _make_b0_model()
    u.load_state_into(tmp_path / "parent.pt", model2)
    adapter2 = build_interaction_adapter("PRODDIFF", FACTOR_DIM)
    u.load_state_into(tmp_path / "adapter.pt", adapter2)
    head2 = torch.nn.Linear(HIDDEN_DIM, 5)
    u.load_state_into(tmp_path / "head.pt", head2)
    for a, b in ((model, model2), (adapter, adapter2), (head, head2)):
        for key in a.state_dict():
            assert torch.equal(a.state_dict()[key], b.state_dict()[key]), key


# ---------------------------------------------------------------------------
# (10) propagation signals finite / (11) permutation deterministic
# ---------------------------------------------------------------------------


def test_propagation_signals_finite() -> None:
    model = _make_b0_model()
    x, ei = _make_x(), _make_edges()
    setup = _wrap("B0", model, x, ei)
    states = u.extract_parent_states(setup, x, ei)
    h1, h2, hp = propagation_signals(model, states["f_pre"], ei, NUM_NODES)
    for t in (h1, h2, hp):
        assert t.shape == (NUM_NODES, 3, FACTOR_DIM)
        assert torch.isfinite(t).all()
    assert torch.equal(hp, states["f_pre"] - h1)


def test_mismatch_permutation_deterministic() -> None:
    from src.analysis.perf_r2d15_utils import fixed_node_permutation

    p1 = fixed_node_permutation(31)
    p2 = fixed_node_permutation(31)
    assert torch.equal(p1, p2)
    assert sorted(p1.tolist()) == list(range(31))


# ---------------------------------------------------------------------------
# (12) no Test access
# ---------------------------------------------------------------------------


def test_no_test_access_guard() -> None:
    class _Fake:
        train_idx = torch.tensor([0])
        val_idx = torch.tensor([1])

    u.assert_no_test_access(_Fake())
    with pytest.raises(AssertionError):
        u.assert_no_test_access(type("X", (), {"train_idx": None, "val_idx": torch.tensor([0])})())


def test_no_test_logic_in_d16_sources() -> None:
    for rel in ("src/analysis/perf_r2d16_utils.py", "src/models/biaxis_r2d16_adapters.py"):
        text = (PROJECT_ROOT / rel).read_text(encoding="utf-8")
        for token in ("test_idx", "test_accuracy", "data.y[", "test_mask", "data.test"):
            assert token not in text, f"{rel} contains {token!r}"
