"""Unit tests for R2-Design-2.8 v2 (v2 plan §5 + Rules I-V, Prompt 1).

    Repaired within-target shuffle: per-target histogram preserved, no
    cross-target movement, >=80% changed among degree>1 edges, >=95%
    non-identity among degree>1 targets, no-op shuffle fails validation.
    Per-target removal: per-target counts exact, random == top counts.
    COUPLED_EQUIV: explicit r*pi factorization reproduces the old
    null-augmented message to < 1e-6.
    New model family: E0 uniform message; real-neighbor simplex (no null);
    lambda simplex; NormMatch exact; zero-init operators == O0 at step 0;
    staged freezing flags; frozen params never move; A0 untouched;
    side_off bitwise; no Test access; no free node/edge tables; no
    RoleMAG/TMTE collisions.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.analysis.perf_r2d27_utils import train_utility_model
from src.models.biaxis_r2_neighbor_utility import CAUSAL_OVERRIDES as OLD_CAUSAL
from src.models.biaxis_r2_neighbor_utility import MODES as OLD_MODES
from src.models.biaxis_r2_relfunc import CAUSAL_OVERRIDES, Model
from src.models.biaxis_r2_relfunc_components import (
    chunked_coupled_message,
    norm_match,
    per_target_edge_mask,
    shuffle_scores_within_target,
    validate_shuffle,
    within_target_perm,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXT_DIM = 13
VISUAL_DIM = 19
NUM_NODES = 23
FACTOR_DIM = 32
HIDDEN_DIM = 64


def _make_parent() -> nn.Module:
    from src.models.biaxis_perf_r1 import Model as R1Model

    yaml_path = PROJECT_ROOT / "configs" / "model" / "biaxis_perf_r1.yaml"
    cfg = OmegaConf.create({"model": OmegaConf.load(yaml_path)})
    cfg.model.hidden_dim = HIDDEN_DIM
    cfg.model.factor_dim = FACTOR_DIM
    cfg.model.dropout = 0.0
    cfg.model.r1.mode = "baseline"
    info = {"input_dim": TEXT_DIM + VISUAL_DIM, "num_nodes": NUM_NODES,
            "num_classes": 5, "text_dim": TEXT_DIM, "visual_dim": VISUAL_DIM}
    model = R1Model(cfg, info)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _make_model(overrides: dict, parent: nn.Module | None = None) -> Model:
    cfg = OmegaConf.create({
        "model": {
            "name": "biaxis_r2_relfunc",
            "hidden_dim": HIDDEN_DIM, "factor_dim": FACTOR_DIM,
            "dropout": 0.0, "activation": "gelu", "norm": "layernorm",
            "type_dim": 4, "edge_chunk_size": 7,
            "exposure": "fixed_full", "composition": "uniform",
            "channel": "mean", "operator": "linear",
            "norm_match": True, "mean_dup": False,
            "uniform_router": False, "target_router": False, "basis_k": 4,
            "freeze_exposure": False, "freeze_composition": False,
            "freeze_channel": False, "freeze_operator": False,
            **overrides,
        }
    })
    info = {"input_dim": TEXT_DIM + VISUAL_DIM, "num_nodes": NUM_NODES,
            "num_classes": 5, "text_dim": TEXT_DIM, "visual_dim": VISUAL_DIM}
    return Model(cfg, info, parent if parent is not None else _make_parent()).eval()


def _make_x(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(NUM_NODES, TEXT_DIM + VISUAL_DIM, generator=generator)


def _make_edges(seed: int = 1, n_edges: int = 60) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, NUM_NODES, (2, n_edges), generator=generator)


def _make_scores(seed: int = 2) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(60, generator=generator)


# ---------------------------------------------------------------------------
# §5.1 repaired within-target shuffle
# ---------------------------------------------------------------------------


def test_shuffle_preserves_per_target_histogram_exactly() -> None:
    ei, s = _make_edges(), _make_scores()
    s_perm = shuffle_scores_within_target(s, ei)
    stats = validate_shuffle(s, s_perm, ei, NUM_NODES)
    assert stats["histogram_exact"] is True
    assert stats["sums_preserved"] is True


def test_shuffle_meets_change_thresholds() -> None:
    for seed in (0, 1, 2, 3, 4):
        ei = _make_edges(seed, 120)
        gen = torch.Generator().manual_seed(seed)
        s = torch.randn(120, generator=gen)
        s_perm = shuffle_scores_within_target(s, ei)
        stats = validate_shuffle(s, s_perm, ei, NUM_NODES)
        assert stats["histogram_exact"] is True, seed
        assert stats["frac_score_changed"] >= 0.80, (seed, stats)
        assert stats["frac_nonidentity_targets"] >= 0.95, (seed, stats)


def test_shuffle_never_crosses_targets() -> None:
    ei, s = _make_edges(), _make_scores()
    perm = within_target_perm(ei)
    assert torch.equal(ei[1], ei[1][perm])  # same dst for every mapped edge


def test_noop_shuffle_fails_validation() -> None:
    ei, s = _make_edges(), _make_scores()
    stats = validate_shuffle(s, s.clone(), ei, NUM_NODES)  # identity
    assert stats["frac_score_changed"] < 0.80
    assert stats["frac_nonidentity_targets"] < 0.95
    assert stats["histogram_exact"] is True  # identity trivially preserves


# ---------------------------------------------------------------------------
# §5.2 per-target removal
# ---------------------------------------------------------------------------


def test_per_target_removal_counts_exact() -> None:
    ei, s = _make_edges(), _make_scores()
    dst = ei[1]
    deg = torch.bincount(dst, minlength=NUM_NODES)
    for pct in (0.10, 0.25, 0.50):
        for op in ("remove_top", "remove_random", "remove_bottom"):
            mask = per_target_edge_mask(s, ei, NUM_NODES, op, pct)
            expected = sum(int(deg[i] * pct) for i in range(NUM_NODES))
            removed = int((~mask).sum().item())
            assert removed == expected, (op, pct, removed, expected)
            # removed edges are distributed per-target exactly
            kept_deg = torch.bincount(dst[mask], minlength=NUM_NODES)
            assert torch.equal(deg - kept_deg, (deg.float() * pct).to(torch.int64))


def test_per_target_random_removes_same_count_as_top() -> None:
    ei, s = _make_edges(), _make_scores()
    top = per_target_edge_mask(s, ei, NUM_NODES, "remove_top", 0.25)
    rnd = per_target_edge_mask(s, ei, NUM_NODES, "remove_random", 0.25)
    assert int((~top).sum().item()) == int((~rnd).sum().item())
    dst = ei[1]
    assert torch.equal(torch.bincount(dst[top], minlength=NUM_NODES),
                       torch.bincount(dst[rnd], minlength=NUM_NODES))


def test_per_target_top_removes_actually_top_scores() -> None:
    ei, s = _make_edges(), _make_scores()
    dst = ei[1]
    mask = per_target_edge_mask(s, ei, NUM_NODES, "remove_top", 0.50)
    for i in range(NUM_NODES):
        m = dst == i
        if m.sum() <= 1:
            continue
        kept_scores = s[mask & m]
        removed_scores = s[(~mask) & m]
        if len(removed_scores) and len(kept_scores):
            assert removed_scores.min() >= kept_scores.max()


def test_per_target_bottom_removes_lowest() -> None:
    ei, s = _make_edges(), _make_scores()
    dst = ei[1]
    mask = per_target_edge_mask(s, ei, NUM_NODES, "remove_bottom", 0.50)
    for i in range(NUM_NODES):
        m = dst == i
        if m.sum() <= 1:
            continue
        kept_scores = s[mask & m]
        removed_scores = s[(~mask) & m]
        if len(removed_scores) and len(kept_scores):
            assert removed_scores.max() <= kept_scores.min()


# ---------------------------------------------------------------------------
# §5.3 COUPLED_EQUIV exact factorization
# ---------------------------------------------------------------------------


def test_coupled_message_matches_old_softmax_exactly() -> None:
    from src.models.biaxis_r2_neighbor_utility_components import (
        chunked_pair_message,
    )

    torch.manual_seed(11)
    n = 40
    f_block = torch.randn(n, 3, FACTOR_DIM)
    ei = torch.randint(0, n, (2, 130))
    s = torch.randn(130)
    null = torch.randn(n)
    payload = torch.randn(n, FACTOR_DIM)
    m_old = chunked_pair_message(f_block, ei, n, s, null, payload,
                                 edge_chunk_size=17)
    m_coupled = chunked_coupled_message(f_block, ei, n, s, null, payload,
                                        edge_chunk_size=17)
    assert (m_old - m_coupled).abs().max().item() < 1e-6


def test_coupled_mode_reproduces_pair_edge_forward() -> None:
    from src.models.biaxis_r2_neighbor_utility import Model as OldModel

    torch.manual_seed(3)
    parent = _make_parent()
    cfg = OmegaConf.create({
        "model": {"name": "biaxis_r2_neighbor_utility", "mode": "pair_edge",
                  "hidden_dim": HIDDEN_DIM, "factor_dim": FACTOR_DIM,
                  "dropout": 0.0, "activation": "gelu", "norm": "layernorm",
                  "type_dim": 4, "edge_chunk_size": 7}})
    info = {"input_dim": TEXT_DIM + VISUAL_DIM, "num_nodes": NUM_NODES,
            "num_classes": 5, "text_dim": TEXT_DIM, "visual_dim": VISUAL_DIM}
    m_old = OldModel(cfg, info, parent).eval()
    cfg2 = OmegaConf.create({"model": {**cfg.model, "mode": "coupled_equiv"}})
    m_coupled = OldModel(cfg2, info, parent).eval()
    m_coupled.load_state_dict(m_old.state_dict())
    x, ei = _make_x(), _make_edges()
    with torch.no_grad():
        z_old, _, _, _, _ = m_old(x, ei)
        z_new, _, _, _, _ = m_coupled(x, ei)
    assert (z_old - z_new).abs().max().item() < 1e-5


# ---------------------------------------------------------------------------
# Rule I: exposure test structure (E0 == uniform real-neighbor aggregation)
# ---------------------------------------------------------------------------


def test_e0_message_is_uniform_neighbor_mean() -> None:
    from src.models.biaxis_p1_components import neighbor_mean

    m = _make_model({})
    x, ei = _make_x(), _make_edges()
    with torch.no_grad():
        f_block, _ = m._parent_forward(x, ei, NUM_NODES)
        msgs = m._side_messages(f_block, ei, NUM_NODES)
    payloads = [m.payload[a](f_block[:, a]) for a in range(3)]
    for b in range(3):
        ref = torch.zeros_like(payloads[0])
        for a in range(3):
            ref = ref + neighbor_mean(ei, payloads[a], NUM_NODES,
                                      edge_chunk_size=m.edge_chunk_size)
        assert torch.allclose(msgs[:, b], ref / 3.0, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Rule II: composition softmax is over real neighbors only (no null)
# ---------------------------------------------------------------------------


def test_composition_simplex_over_real_neighbors_only() -> None:
    m = _make_model({"exposure": "fixed_full", "composition": "pair"})
    x, ei = _make_x(), _make_edges()
    with torch.no_grad():
        f_block, _ = m._parent_forward(x, ei, NUM_NODES)
        num_edges = int(ei.size(1))
        dst = ei[1]
        for (a, b) in [(0, 0), (0, 1), (2, 1)]:
            s = m._comp_pair_scores(f_block, ei, a, b, num_edges)
            # real-neighbor simplex: sum_j pi_ji == 1 for every target with
            # degree > 0 (no null token consumes mass)
            max_i = torch.zeros(NUM_NODES, dtype=s.dtype)
            seg = torch.zeros(NUM_NODES, dtype=s.dtype)
            seg = seg.scatter_reduce(0, dst, s, reduce="amax", include_self=False)
            max_i = torch.maximum(max_i, seg)
            denom = torch.zeros(NUM_NODES, dtype=s.dtype)
            denom = denom.scatter_add(0, dst, torch.exp(s - max_i[dst]))
            pi = torch.exp(s - max_i[dst]) / denom[dst]
            total = torch.zeros(NUM_NODES, dtype=s.dtype)
            total = total.scatter_add(0, dst, pi)
            deg = torch.bincount(dst, minlength=NUM_NODES)
            assert torch.allclose(total[deg > 0],
                                  torch.ones_like(total[deg > 0]),
                                  atol=1e-4, rtol=1e-4), (a, b)


# ---------------------------------------------------------------------------
# Rule III: channel lambda is a simplex over sources
# ---------------------------------------------------------------------------


def test_channel_softmax_lambda_is_simplex() -> None:
    m = _make_model({"channel": "softmax"})
    x, ei = _make_x(), _make_edges()
    with torch.no_grad():
        f_block, _ = m._parent_forward(x, ei, NUM_NODES)
        for b in range(3):
            lam = m.channel_net(f_block[:, b], m.channel_emb.weight)
            assert lam.shape == (NUM_NODES, 3)
            assert torch.allclose(lam.sum(dim=-1),
                                  torch.ones(NUM_NODES), atol=1e-5)


# ---------------------------------------------------------------------------
# Rule IV: NormMatch keeps magnitude, changes only content
# ---------------------------------------------------------------------------


def test_norm_match_exact_magnitude() -> None:
    torch.manual_seed(5)
    v = torch.randn(37, FACTOR_DIM)
    v_out = v * torch.randn_like(v).abs() + 0.3
    v_hat = norm_match(v_out, v)
    assert torch.allclose(v_hat.norm(dim=-1), v.norm(dim=-1), atol=1e-4)
    # direction follows the operator output
    cos = torch.nn.functional.cosine_similarity(v_hat, v_out, dim=-1)
    assert torch.all(cos > 0.999)


# ---------------------------------------------------------------------------
# O1/O2/O3 zero-init step 0 == O0; O4 small-init close to O0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("operator_kind", ["static_pair", "target_film",
                                           "edge_film"])
def test_zero_init_operator_step0_equals_o0(operator_kind: str) -> None:
    torch.manual_seed(9)
    parent = _make_parent()  # same parent for both models (z_base identity)
    base = _make_model({}, parent)  # O0 reference
    m = _make_model({"operator": operator_kind}, parent)
    # copy base state into the shared prefixes (payload) so the only
    # difference is the zero-init operator module
    state = {k: v for k, v in base.state_dict().items() if not k.startswith(
        ("operator_net.", "operator_emb."))}
    m.load_state_dict({**m.state_dict(), **state})
    x, ei = _make_x(), _make_edges()
    with torch.no_grad():
        z0, _, _, _, _ = base(x, ei)
        z1, _, _, _, _ = m(x, ei)
    assert torch.allclose(z0, z1, atol=1e-5, rtol=1e-5)


def test_basis_operator_step0_close_to_o0() -> None:
    torch.manual_seed(10)
    parent = _make_parent()
    base = _make_model({}, parent)
    m = _make_model({"operator": "basis"}, parent)
    state = {k: v for k, v in base.state_dict().items() if not k.startswith(
        ("operator_net.", "operator_emb."))}
    m.load_state_dict({**m.state_dict(), **state})
    x, ei = _make_x(), _make_edges()
    with torch.no_grad():
        z0, _, _, _, _ = base(x, ei)
        z1, _, _, _, _ = m(x, ei)
    assert (z0 - z1).abs().max().item() < 5e-3


# ---------------------------------------------------------------------------
# v2 §2: no free node/edge tables
# ---------------------------------------------------------------------------


def test_no_free_node_or_edge_tables() -> None:
    for overrides in ({"exposure": "pair"}, {"composition": "pair"},
                      {"channel": "softmax"}, {"operator": "basis"}):
        m = _make_model(overrides)
        for mod in m.modules():
            if isinstance(mod, nn.Embedding):
                # only factor-type embeddings (3 rows) are allowed
                assert mod.num_embeddings <= 16, mod.num_embeddings
            if isinstance(mod, nn.Linear):
                assert mod.in_features < 100000  # no [E]- or [N]-indexed layers


# ---------------------------------------------------------------------------
# Rule V: staged freezing
# ---------------------------------------------------------------------------


def test_freeze_flags_set_requires_grad() -> None:
    m = _make_model({"exposure": "pair", "composition": "target",
                     "channel": "softmax", "operator": "target_film",
                     "freeze_exposure": True, "freeze_composition": True})
    for p in list(m.exposure_net.parameters()) + \
            list(m.exposure_emb.parameters()) + list(m.payload.parameters()):
        assert p.requires_grad is False
    for p in list(m.comp_net.parameters()) + list(m.comp_emb.parameters()):
        assert p.requires_grad is False
    for p in list(m.channel_net.parameters()) + list(m.channel_emb.parameters()):
        assert p.requires_grad is True
    for p in list(m.operator_net.parameters()) + list(m.operator_emb.parameters()):
        assert p.requires_grad is True


def test_frozen_params_unchanged_after_training() -> None:
    torch.manual_seed(21)
    parent = _make_parent()
    m = _make_model({"exposure": "target", "composition": "pair",
                     "freeze_exposure": True}, parent)
    head = nn.Linear(m.out_dim, 5)
    before = {k: v.detach().clone() for k, v in m.state_dict().items()}
    frozen_keys = {k for k in before
                   if k.startswith(("exposure_net.", "exposure_emb.", "payload."))}
    n = 40
    data = SimpleNamespace(
        x=torch.randn(n, TEXT_DIM + VISUAL_DIM),
        edge_index=torch.randint(0, n, (2, 120)),
        train_idx=torch.arange(0, 20), val_idx=torch.arange(20, 28),
        test_idx=torch.arange(28, 40), y=torch.randint(0, 5, (n,)),
        num_classes=5)
    train_utility_model(data, m, head, torch.device("cpu"), total_epochs=2)
    for k, v in m.state_dict().items():
        if k in frozen_keys:
            assert torch.equal(v, before[k]), k
    # something trainable actually moved
    moved = any(not torch.equal(before[k], m.state_dict()[k])
                for k in before if k not in frozen_keys)
    assert moved


def test_load_frozen_components_only_copies_requested_prefixes() -> None:
    from src.analysis.perf_r2d28_utils import load_frozen_components

    torch.manual_seed(12)
    src = _make_model({"exposure": "pair", "composition": "uniform"})
    dst = _make_model({"exposure": "pair", "composition": "pair",
                       "freeze_exposure": True})
    ckpt = PROJECT_ROOT / "tests" / "_tmp_r2d28_ckpt.pt"
    ckpt.parent.mkdir(exist_ok=True)
    torch.save({"model_state": src.state_dict()}, ckpt)
    try:
        info = load_frozen_components(dst, ckpt, ["exposure_net.", "exposure_emb.",
                                                  "payload."])
        assert info["copied_params"] > 0
        for k in src.state_dict():
            if k.startswith(("exposure_net.", "exposure_emb.", "payload.")):
                assert torch.equal(src.state_dict()[k], dst.state_dict()[k]), k
        # composition was NOT copied (fresh init)
        for k in dst.state_dict():
            if k.startswith(("comp_net.", "comp_emb.")):
                assert not torch.equal(src.state_dict().get(
                    k, torch.zeros(1)), dst.state_dict()[k])
    finally:
        ckpt.unlink()


# ---------------------------------------------------------------------------
# Parent / discipline invariants
# ---------------------------------------------------------------------------


def test_a0_parent_never_updated() -> None:
    torch.manual_seed(37)
    parent = _make_parent()
    theta0 = {k: v.detach().clone() for k, v in parent.state_dict().items()}
    m = _make_model({"exposure": "target"}, parent)
    head = nn.Linear(m.out_dim, 5)
    n = 40
    data = SimpleNamespace(
        x=torch.randn(n, TEXT_DIM + VISUAL_DIM),
        edge_index=torch.randint(0, n, (2, 120)),
        train_idx=torch.arange(0, 20), val_idx=torch.arange(20, 28),
        test_idx=torch.arange(28, 40), y=torch.randint(0, 5, (n,)),
        num_classes=5)
    train_utility_model(data, m, head, torch.device("cpu"), total_epochs=2)
    for k, v in parent.state_dict().items():
        assert torch.equal(v, theta0[k]), k


def test_side_off_reproduces_parent_bitwise() -> None:
    torch.manual_seed(7)
    parent = _make_parent()
    m = _make_model({"exposure": "pair", "composition": "pair",
                     "channel": "attn", "operator": "basis"}, parent)
    x, ei = _make_x(), _make_edges()
    with torch.no_grad():
        z_parent = parent(x, ei)[0]
        z_off, _, _, _, _ = m(x, ei, causal="side_off")
    assert torch.equal(z_off, z_parent)


class _GuardY:
    def __init__(self, y, test_idx) -> None:
        self._y = y
        self._test = set(int(i) for i in test_idx.tolist())

    def __getitem__(self, idx):
        idxs = idx.tolist() if torch.is_tensor(idx) else [idx]
        for i in idxs:
            if int(i) in self._test:
                raise RuntimeError("TEST LABEL ACCESS VIOLATION")
        return self._y[idx]


def test_no_test_access_in_training_loop() -> None:
    torch.manual_seed(29)
    parent = _make_parent()
    m = _make_model({"exposure": "target"}, parent)
    head = nn.Linear(m.out_dim, 5)
    n = 40
    data = SimpleNamespace(
        x=torch.randn(n, TEXT_DIM + VISUAL_DIM),
        edge_index=torch.randint(0, n, (2, 120)),
        train_idx=torch.arange(0, 20), val_idx=torch.arange(20, 28),
        test_idx=torch.arange(28, 40), y=torch.randint(0, 5, (n,)),
        num_classes=5)
    data.y = _GuardY(data.y, data.test_idx)
    res = train_utility_model(data, m, head, torch.device("cpu"), total_epochs=2)
    assert 0.0 <= res["best_val_acc"] <= 1.0


# ---------------------------------------------------------------------------
# Exposure values / forward sanity for all stage-representative configs
# ---------------------------------------------------------------------------


def test_exposure_value_counts_and_range() -> None:
    x, ei = _make_x(), _make_edges()
    for kind, n_vals in (("node", 1), ("target", 3), ("source", 3), ("pair", 9)):
        m = _make_model({"exposure": kind})
        with torch.no_grad():
            f_block, _ = m._parent_forward(x, ei, NUM_NODES)
            rvals = m._exposure_values(f_block)
        assert len(rvals) == n_vals, kind
        for v in rvals.values():
            assert torch.all((v > 0) & (v < 1)), kind


def test_forward_finite_for_stage_configs() -> None:
    x, ei = _make_x(), _make_edges()
    configs = [
        {},  # E0 / F0
        {"exposure": "node"}, {"exposure": "target"},
        {"exposure": "source"}, {"exposure": "pair"},
        {"exposure": "target", "composition": "generic"},
        {"exposure": "target", "composition": "target"},
        {"exposure": "target", "composition": "source"},
        {"exposure": "target", "composition": "pair"},
        {"exposure": "target", "composition": "pair", "channel": "softmax"},
        {"exposure": "target", "composition": "pair", "channel": "concat"},
        {"exposure": "target", "composition": "pair", "channel": "attn"},
        {"exposure": "target", "composition": "pair", "channel": "mean",
         "operator": "static_pair"},
        {"exposure": "target", "composition": "pair", "channel": "mean",
         "operator": "target_film"},
        {"exposure": "target", "composition": "pair", "channel": "mean",
         "operator": "edge_film"},
        {"exposure": "target", "composition": "pair", "channel": "mean",
         "operator": "basis"},
        {"exposure": "target", "composition": "pair", "channel": "mean",
         "operator": "basis", "uniform_router": True},
        {"exposure": "target", "composition": "pair", "channel": "mean",
         "operator": "basis", "target_router": True},
        {"channel": "concat", "mean_dup": True},
        {"channel": "attn", "mean_dup": True},
    ]
    for cfg in configs:
        m = _make_model(cfg)
        with torch.no_grad():
            z, _, _, _, _ = m(x, ei)
        assert torch.isfinite(z).all(), cfg
        assert z.size(-1) == HIDDEN_DIM + 3 * FACTOR_DIM, cfg


def test_operator_causal_guards() -> None:
    x, ei = _make_x(), _make_edges()
    m = _make_model({"exposure": "target", "composition": "pair",
                     "operator": "target_film"})
    with pytest.raises(ValueError):
        m(x, ei, causal="router_uniformize")
    m2 = _make_model({"exposure": "target", "composition": "pair",
                      "operator": "linear"})
    with pytest.raises(ValueError):
        m2(x, ei, causal="film_neutralize")
    m3 = _make_model({"exposure": "target", "composition": "uniform"})
    with pytest.raises(ValueError):
        m3(x, ei, causal="within_target_shuffle")


def test_exposure_stats_train_labels_only() -> None:
    m = _make_model({"exposure": "pair"})
    x, ei = _make_x(), _make_edges()
    train_idx = torch.arange(0, 12)
    train_y = torch.randint(0, 5, (12,))
    with torch.no_grad():
        stats = m.export_exposure_stats(x, ei, train_idx, train_y)
    assert stats["exposure_kind"] == "pair"
    assert len(stats["per_key"]) == 9
    assert len(stats["r_matrix_mean"]) == 3
    for key, v in stats["per_key"].items():
        assert len(v["mean_r_train_class"]) == 5, key


# ---------------------------------------------------------------------------
# Repaired machinery registered on the OLD model (D2.8-A will use it)
# ---------------------------------------------------------------------------


def test_old_model_registers_repaired_causal_keys() -> None:
    for key in ("within_target_shuffle_fixed",
                "remove_top_per_target_10", "remove_random_per_target_25",
                "remove_bottom_per_target_50", "keep_top_per_target_25"):
        assert key in OLD_CAUSAL, key
    assert "coupled_equiv" in OLD_MODES


def test_old_model_shuffle_now_meets_thresholds() -> None:
    from src.models.biaxis_r2_neighbor_utility import Model as OldModel

    torch.manual_seed(4)
    parent = _make_parent()
    cfg = OmegaConf.create({
        "model": {"name": "biaxis_r2_neighbor_utility", "mode": "pair_edge",
                  "hidden_dim": HIDDEN_DIM, "factor_dim": FACTOR_DIM,
                  "dropout": 0.0, "activation": "gelu", "norm": "layernorm",
                  "type_dim": 4, "edge_chunk_size": 7}})
    info = {"input_dim": TEXT_DIM + VISUAL_DIM, "num_nodes": NUM_NODES,
            "num_classes": 5, "text_dim": TEXT_DIM, "visual_dim": VISUAL_DIM}
    m = OldModel(cfg, info, parent).eval()
    ei = _make_edges(3, 120)
    gen = torch.Generator().manual_seed(8)
    s = torch.randn(120, generator=gen)
    s_perm = m._within_target_shuffle(s, ei, NUM_NODES, 120)
    stats = validate_shuffle(s, s_perm, ei, NUM_NODES)
    assert stats["frac_score_changed"] >= 0.80
    assert stats["frac_nonidentity_targets"] >= 0.95


# ---------------------------------------------------------------------------
# Collision guardrails (RoleMAG / TMTE / CoMAG)
# ---------------------------------------------------------------------------


def test_no_role_labels_or_topology_reconstruction() -> None:
    src = (PROJECT_ROOT / "src" / "models" / "biaxis_r2_relfunc.py").read_text()
    for banned in ("heterophilous", "complementary", "add_edge", "new_edge"):
        assert banned not in src.lower()
    src2 = (PROJECT_ROOT / "src" / "models" / "biaxis_r2_relfunc_components.py").read_text()
    for banned in ("heterophilous", "complementary"):
        assert banned not in src2.lower()
    m = _make_model({"exposure": "pair", "composition": "pair"})
    x, ei = _make_x(), _make_edges()
    ei_before = ei.clone()
    with torch.no_grad():
        m(x, ei)
    assert torch.equal(ei, ei_before)
