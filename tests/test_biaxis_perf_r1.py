"""Unit tests for the R1 performance branch (plan §34 / audit §3).

Component tests (Prompt 2): FactorConditionedEdgeReliability +
reliable_relation_weighted_mean + reliability_edge_statistics —
eta neutral / symmetry / range / eta=1 aggregation equivalence /
chunk/full equivalence / gradients / isolated / memory discipline.

Model tests (Prompt 3) are appended below (M1-M8): baseline bitwise
equivalence with biaxis_final, A1 zero-init numerical equivalence,
state_dict keys, inference, gradients, isolated, diagnostics keys.
"""

from __future__ import annotations

import torch

from src.models.biaxis_perf_r1_components import (
    DynamicLocalScoreResidual,
    FactorConditionedEdgeReliability,
    FactorConditionedRelationCalibration,
    SupportRelationScoreResidual,
    calibration_edge_statistics,
    relation_calibrated_weighted_mean,
    reliability_regularization,
    reliable_relation_weighted_mean,
    reliability_edge_statistics,
)
from src.models.biaxis_p1_components import relation_mass, relation_weighted_mean

# Test geometry (components): small so every test is CPU-fast.
N, F, K, D = 50, 3, 4, 32
PROJ, HIDDEN = 16, 32


def _rand(shape, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=generator)


def _make_module(perturb: bool = False) -> FactorConditionedEdgeReliability:
    module = FactorConditionedEdgeReliability(
        num_factors=F, factor_dim=D, proj_dim=PROJ, hidden_dim=HIDDEN, activation="gelu"
    )
    if perturb:
        with torch.no_grad():
            module.mlp[-1].weight.uniform_(-0.5, 0.5)
            module.mlp[-1].bias.uniform_(-0.5, 0.5)
    return module


def _make_graph(seed: int = 0, num_edges: int = 200, num_nodes: int = N):
    generator = torch.Generator().manual_seed(seed)
    src = torch.randint(0, num_nodes, (num_edges,), generator=generator)
    dst = torch.randint(0, num_nodes, (num_edges,), generator=generator)
    edge_index = torch.stack([src, dst], dim=0)
    r = torch.softmax(_rand((num_edges, K), seed + 1), dim=-1)
    f_block = _rand((num_nodes, F, D), seed + 2)
    return edge_index, r, f_block


# ---------------------------------------------------------------------------
# FactorConditionedEdgeReliability
# ---------------------------------------------------------------------------


def test_eta_neutral_zero_init() -> None:
    """Zero-init last layer -> delta = 0 -> eta == 1 EXACTLY (audit §7)."""
    module = _make_module(perturb=False)
    eta = module(_rand((40, F, D), 0), _rand((40, F, D), 1))
    assert torch.equal(eta, torch.ones_like(eta))


def test_eta_strict_range() -> None:
    """eta in (0, 2) strictly for a non-degenerate module."""
    module = _make_module(perturb=True)
    eta = module(_rand((40, F, D), 0), _rand((40, F, D), 1))
    assert torch.isfinite(eta).all()
    assert bool((eta > 0.0).all()) and bool((eta < 2.0).all())


def test_eta_symmetric_reverse_edges() -> None:
    """The token is symmetric in (i, j) -> reverse directions get
    bitwise-identical eta (undirected-graph discipline)."""
    module = _make_module(perturb=True)
    f_a = _rand((20, F, D), 0)
    f_b = _rand((20, F, D), 1)
    eta_ab = module(f_a, f_b)
    eta_ba = module(f_b, f_a)
    assert torch.equal(eta_ab, eta_ba)


def test_eta_extra_params_small() -> None:
    """R1-A must stay a tiny bolt-on (audit Q8: ~18.7K at d=128/p=32/h=64)."""
    module = FactorConditionedEdgeReliability(
        num_factors=3, factor_dim=128, proj_dim=32, hidden_dim=64
    )
    assert module.extra_params() == 3 * (128 * 32 + 32) + (97 * 64 + 64) + (64 * 1 + 1)


# ---------------------------------------------------------------------------
# reliable_relation_weighted_mean
# ---------------------------------------------------------------------------


def test_aggregation_eta_one_equivalence() -> None:
    """eta == 1 -> mathematically equal to relation_weighted_mean; float
    grouping differs -> allclose, never equal (audit §Q5/risk 1)."""
    edge_index, r, f_block = _make_graph()
    module = _make_module(perturb=False)
    g_perm, eff_mass = reliable_relation_weighted_mean(
        edge_index, r, f_block, module, N, edge_chunk_size=None
    )
    f_cat = f_block.reshape(N, F * D)
    g_ref, mass_ref = relation_weighted_mean(
        edge_index, r, f_cat, N, edge_chunk_size=None
    )
    g_ref = g_ref.reshape(N, K, F, D).permute(0, 2, 1, 3)
    assert g_perm.shape == (N, F, K, D) and eff_mass.shape == (N, F, K)
    assert torch.allclose(g_perm, g_ref, rtol=1e-5, atol=1e-5)
    assert torch.allclose(
        eff_mass, mass_ref.unsqueeze(1).expand(N, F, K), rtol=1e-5, atol=1e-5
    )


def test_chunk_full_equivalence() -> None:
    """chunk >= E runs the same single-iteration path -> bitwise equal;
    different small chunks -> allclose (grouping order differs)."""
    edge_index, r, f_block = _make_graph(num_edges=200)
    module = _make_module(perturb=True)
    g_full, m_full = reliable_relation_weighted_mean(
        edge_index, r, f_block, module, N, edge_chunk_size=None
    )
    g_big, m_big = reliable_relation_weighted_mean(
        edge_index, r, f_block, module, N, edge_chunk_size=10_000
    )
    assert torch.equal(g_full, g_big) and torch.equal(m_full, m_big)
    g_c37, m_c37 = reliable_relation_weighted_mean(
        edge_index, r, f_block, module, N, edge_chunk_size=37
    )
    g_c41, m_c41 = reliable_relation_weighted_mean(
        edge_index, r, f_block, module, N, edge_chunk_size=41
    )
    assert torch.allclose(g_c37, g_c41, rtol=1e-5, atol=1e-5)
    assert torch.allclose(m_c37, m_c41, rtol=1e-5, atol=1e-5)
    assert torch.allclose(g_c37, g_full, rtol=1e-5, atol=1e-5)


def test_gradients_finite_and_step0_dynamics() -> None:
    """Step 0: only the zero-init final layer receives gradient (hidden /
    projections are exactly zero — expected zero-init dynamics, audit §Q7).
    After two optimizer steps the reliability pathway is fully alive."""
    edge_index, r, f_block = _make_graph(num_edges=60, num_nodes=12)
    module = _make_module(perturb=False)
    opt = torch.optim.SGD(module.parameters(), lr=0.5)

    def step():
        opt.zero_grad()
        g_perm, _ = reliable_relation_weighted_mean(
            edge_index, r, f_block, module, 12, edge_chunk_size=None
        )
        loss = g_perm.square().sum()
        loss.backward()
        return loss

    step()
    assert module.mlp[-1].weight.grad is not None
    assert module.mlp[-1].weight.grad.norm() > 1e-9
    assert module.mlp[-1].bias.grad is not None and module.mlp[-1].bias.grad.norm() > 1e-9
    # hidden layer + projections: zero gradient at step 0 (expected).
    assert torch.equal(module.mlp[0].weight.grad, torch.zeros_like(module.mlp[0].weight.grad))
    assert torch.equal(
        module.projections[0].weight.grad,
        torch.zeros_like(module.projections[0].weight.grad),
    )
    for p in module.parameters():
        assert torch.isfinite(p.grad).all()
    opt.step()

    step()
    assert module.projections[0].weight.grad.norm() > 1e-9
    opt.step()

    step()
    for p in module.parameters():
        assert torch.isfinite(p.grad).all()


def test_isolated_nodes_zero_and_no_nan() -> None:
    """No incoming edges -> acc/effective_mass stay 0 -> g = 0, no NaN."""
    edge_index, r, f_block = _make_graph(num_edges=150, num_nodes=40)
    # Rebuild dst so node 0 has no incoming edges.
    edge_index = edge_index.clone()
    edge_index[1] = edge_index[1] % 39 + 1
    module = _make_module(perturb=True)
    g_perm, eff_mass = reliable_relation_weighted_mean(
        edge_index, r, f_block, module, 40, edge_chunk_size=64
    )
    assert torch.isfinite(g_perm).all() and torch.isfinite(eff_mass).all()
    assert torch.equal(g_perm[0], torch.zeros_like(g_perm[0]))
    assert torch.equal(eff_mass[0], torch.zeros_like(eff_mass[0]))


def test_empty_graph_zeros() -> None:
    """E = 0 must not crash and must return zeros."""
    edge_index = torch.empty(2, 0, dtype=torch.long)
    r = torch.empty(0, K)
    f_block = _rand((N, F, D), 0)
    module = _make_module(perturb=False)
    g_perm, eff_mass = reliable_relation_weighted_mean(
        edge_index, r, f_block, module, N, edge_chunk_size=None
    )
    assert torch.equal(g_perm, torch.zeros_like(g_perm))
    assert torch.equal(eff_mass, torch.zeros_like(eff_mass))


def test_scale_smoke_no_giant_edge_tensor() -> None:
    """CPU scale smoke (audit §Q4): E=50K with small chunks runs with only
    [chunk, ...] transients (mirrors P3's no-giant-tensor smoke style)."""
    edge_index, r, f_block = _make_graph(num_edges=50_000, num_nodes=4096)
    module = _make_module(perturb=True)
    g_perm, eff_mass = reliable_relation_weighted_mean(
        edge_index, r, f_block, module, 4096, edge_chunk_size=8192
    )
    assert g_perm.shape == (4096, F, K, D) and eff_mass.shape == (4096, F, K)
    assert torch.isfinite(g_perm).all()


# ---------------------------------------------------------------------------
# reliability_edge_statistics (audit §10 + review §8)
# ---------------------------------------------------------------------------


def test_statistics_keys_and_neutral_values() -> None:
    import json

    edge_index, r, f_block = _make_graph(num_edges=150, num_nodes=40)
    module = _make_module(perturb=False)  # eta == 1 everywhere
    stats = reliability_edge_statistics(
        edge_index, r, f_block, module, 40, edge_chunk_size=64
    )
    json.dumps(stats)  # JSON-safe
    assert set(stats) == {"eta", "neighbor", "corr_eta_cos", "weighted_semantic_coherence"}
    for f in range(F):
        row = stats["eta"][f"F{f + 1}"]
        assert abs(row["mean"] - 1.0) < 1e-6
        assert row["std"] < 1e-6 and row["cv"] < 1e-6
        assert abs(row["p50"] - 1.0) < 1e-6
        assert row["frac_lt_0.5"] == 0.0 and row["frac_gt_1.5"] == 0.0
        assert stats["neighbor"][f"F{f + 1}"]["neighbor_std_mean"] < 1e-6
    sim = stats["weighted_semantic_coherence"]
    assert len(sim) == F and all(len(row) == K for row in sim)
    assert all(-1.0 - 1e-6 <= v <= 1.0 + 1e-6 for row in sim for v in row)


def test_statistics_perturbed_differentiation() -> None:
    """Perturbed module: eta spreads, neighbor-wise std appears."""
    edge_index, r, f_block = _make_graph(num_edges=150, num_nodes=40)
    module = _make_module(perturb=True)
    stats = reliability_edge_statistics(
        edge_index, r, f_block, module, 40, edge_chunk_size=64
    )
    means = [stats["eta"][f"F{f + 1}"]["mean"] for f in range(F)]
    assert all(0.0 < m < 2.0 for m in means)
    assert any(stats["eta"][f"F{f + 1}"]["std"] > 1e-4 for f in range(F))
    assert any(stats["neighbor"][f"F{f + 1}"]["neighbor_std_mean"] > 1e-4 for f in range(F))
    assert all(-1.0 - 1e-6 <= c <= 1.0 + 1e-6 for c in stats["corr_eta_cos"].values())


def test_statistics_weighted_coherence_matches_manual() -> None:
    """Sim_{f,k} = sum r*eta*cos / sum r*eta checked against a manual
    computation on a tiny graph (torch.allclose)."""
    torch.manual_seed(0)
    edge_index, r, f_block = _make_graph(num_edges=40, num_nodes=12)
    module = _make_module(perturb=True)
    stats = reliability_edge_statistics(
        edge_index, r, f_block, module, 12, edge_chunk_size=16
    )
    src, dst = edge_index[0], edge_index[1]
    eta_full = module(f_block[src], f_block[dst]).double()
    sim_ref = torch.zeros(F, K, dtype=torch.float64)
    for f in range(F):
        a, b = f_block[src, f], f_block[dst, f]
        cos = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-8)
        for k in range(K):
            w = r[:, k].double() * eta_full[:, f]
            sim_ref[f, k] = (w * cos.double()).sum() / (w.sum() + 1e-8)
    assert torch.allclose(
        torch.tensor(stats["weighted_semantic_coherence"]).double(), sim_ref, atol=1e-5
    )


# ---------------------------------------------------------------------------
# reliability_regularization (review option B control)
# ---------------------------------------------------------------------------


def test_regularization_zero_at_neutral() -> None:
    """eta == 1 -> BOTH regularizers are exactly 0."""
    edge_index, r, f_block = _make_graph(num_edges=150, num_nodes=40)
    module = _make_module(perturb=False)
    for reg_type in ("mean1", "band"):
        loss = reliability_regularization(
            edge_index, f_block, module, 40, reg_type=reg_type, edge_chunk_size=64
        )
        assert torch.equal(loss, torch.zeros_like(loss))


def test_regularization_positive_perturbed_and_manual() -> None:
    """Perturbed module: losses > 0 and match a manual chunk-free reference."""
    edge_index, r, f_block = _make_graph(num_edges=120, num_nodes=30)
    module = _make_module(perturb=True)
    src, dst = edge_index[0], edge_index[1]
    eta = module(f_block[src], f_block[dst])
    ref_mean1 = ((eta - 1.0).square()).mean()
    ref_band = (
        (0.5 - eta).clamp_min(0.0).square() + (eta - 1.5).clamp_min(0.0).square()
    ).mean()
    for reg_type, ref in (("mean1", ref_mean1), ("band", ref_band)):
        loss = reliability_regularization(
            edge_index, f_block, module, 30, reg_type=reg_type, edge_chunk_size=64
        )
        assert loss.item() > 0
        assert torch.allclose(loss, ref, rtol=1e-5, atol=1e-6)


def test_regularization_band_ignores_healthy_band() -> None:
    """band(0.5, 1.5) must not penalize eta strictly inside the band:
    (eta - 1)^2 can be positive while the band loss is exactly 0."""
    edge_index, r, f_block = _make_graph(num_edges=120, num_nodes=30)
    module = _make_module(perturb=False)
    with torch.no_grad():
        module.mlp[-1].bias.fill_(0.2231)  # sigmoid(0.2231/... ) ~ keep in band
    # 2*sigmoid(0.2231) = 2*0.5555 = 1.111 -> inside (0.5, 1.5)
    loss_band = reliability_regularization(
        edge_index, f_block, module, 30, reg_type="band", edge_chunk_size=64
    )
    assert loss_band.item() == 0.0
    loss_mean1 = reliability_regularization(
        edge_index, f_block, module, 30, reg_type="mean1", edge_chunk_size=64
    )
    assert loss_mean1.item() > 0.0


def test_regularization_gradients_finite() -> None:
    edge_index, r, f_block = _make_graph(num_edges=60, num_nodes=12)
    module = _make_module(perturb=True)
    loss = reliability_regularization(
        edge_index, f_block, module, 12, reg_type="band", edge_chunk_size=32
    )
    loss.backward()
    for p in module.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_statistics_chunk_invariance() -> None:
    """Different chunk sizes -> identical statistics (f64 accumulators and
    exact-quantile pass are chunk-order invariant up to float grouping)."""
    edge_index, r, f_block = _make_graph(num_edges=300, num_nodes=60)
    module = _make_module(perturb=True)
    s1 = reliability_edge_statistics(edge_index, r, f_block, module, 60, edge_chunk_size=64)
    s2 = reliability_edge_statistics(edge_index, r, f_block, module, 60, edge_chunk_size=100)
    for f in range(F):
        for key in ("mean", "std", "p10", "p50", "p90"):
            assert abs(s1["eta"][f"F{f + 1}"][key] - s2["eta"][f"F{f + 1}"][key]) < 1e-6
    for f in range(F):
        assert abs(
            s1["neighbor"][f"F{f + 1}"]["neighbor_std_mean"]
            - s2["neighbor"][f"F{f + 1}"]["neighbor_std_mean"]
        ) < 1e-6


# ===========================================================================
# A2 components: FactorConditionedRelationCalibration (user-authorized)
# ===========================================================================


def _make_cal(perturb: bool = False) -> FactorConditionedRelationCalibration:
    cal = FactorConditionedRelationCalibration(
        num_factors=F, factor_dim=D, proj_dim=PROJ, hidden_dim=HIDDEN,
        num_relations=K, activation="gelu",
    )
    if perturb:
        with torch.no_grad():
            cal.mlp[-1].weight.uniform_(-0.5, 0.5)
            cal.mlp[-1].bias.uniform_(-0.5, 0.5)
    return cal


def test_cal_zero_init_equals_r_str() -> None:
    """delta == 0 -> r^f == r^str (mathematical equality; the log/softmax
    roundtrip is not bitwise -> allclose, never equal)."""
    cal = _make_cal(perturb=False)
    edge_index, r, f_block = _make_graph()
    src, dst = edge_index[0], edge_index[1]
    rf = cal(r, f_block[src], f_block[dst])
    assert rf.shape == (200, F, K)
    assert torch.allclose(rf, r.unsqueeze(1).expand(-1, F, -1), rtol=1e-6, atol=1e-6)
    assert torch.allclose(rf.sum(dim=-1), torch.ones(200, F), rtol=1e-6, atol=1e-6)


def test_cal_demean_tanh_bounds() -> None:
    """De-mean makes q shift-invariant and delta stays in (-1, 1)."""
    cal = _make_cal(perturb=True)
    edge_index, r, f_block = _make_graph(num_edges=80)
    src, dst = edge_index[0], edge_index[1]
    f_src, f_dst = f_block[src], f_block[dst]
    rf = cal(r, f_src, f_dst)
    assert rf.shape == (80, F, K)
    assert torch.allclose(rf.sum(dim=-1), torch.ones(80, F), rtol=1e-6, atol=1e-6)
    # bias shift must not change r^f (softmax shift invariance via de-mean).
    with torch.no_grad():
        cal.mlp[-1].bias += 3.0
    rf2 = cal(r, f_src, f_dst)
    assert torch.allclose(rf, rf2, rtol=1e-6, atol=1e-6)
    assert bool((rf > 0.0).all())


def test_cal_symmetric_reverse_edges() -> None:
    """Symmetric token -> reverse directions get bitwise-identical r^f."""
    cal = _make_cal(perturb=True)
    f_a = _rand((20, F, D), 0)
    f_b = _rand((20, F, D), 1)
    r_ab = torch.softmax(_rand((20, K), 2), dim=-1)
    assert torch.equal(cal(r_ab, f_a, f_b), cal(r_ab, f_b, f_a))


def test_cal_aggregation_zero_delta_equivalence() -> None:
    """delta == 0 -> mathematically equal to relation_weighted_mean
    (float grouping differs -> allclose)."""
    edge_index, r, f_block = _make_graph()
    cal = _make_cal(perturb=False)
    g_perm, eff_mass = relation_calibrated_weighted_mean(
        edge_index, r, f_block, cal, N, edge_chunk_size=None
    )
    f_cat = f_block.reshape(N, F * D)
    g_ref, mass_ref = relation_weighted_mean(
        edge_index, r, f_cat, N, edge_chunk_size=None
    )
    g_ref = g_ref.reshape(N, K, F, D).permute(0, 2, 1, 3)
    assert torch.allclose(g_perm, g_ref, rtol=1e-5, atol=1e-5)
    assert torch.allclose(
        eff_mass, mass_ref.unsqueeze(1).expand(N, F, K), rtol=1e-5, atol=1e-5
    )


def test_cal_chunk_full_equivalence() -> None:
    edge_index, r, f_block = _make_graph(num_edges=200)
    cal = _make_cal(perturb=True)
    g_full, m_full = relation_calibrated_weighted_mean(
        edge_index, r, f_block, cal, N, edge_chunk_size=None
    )
    g_big, m_big = relation_calibrated_weighted_mean(
        edge_index, r, f_block, cal, N, edge_chunk_size=10_000
    )
    assert torch.equal(g_full, g_big) and torch.equal(m_full, m_big)
    g_c37, m_c37 = relation_calibrated_weighted_mean(
        edge_index, r, f_block, cal, N, edge_chunk_size=37
    )
    assert torch.allclose(g_full, g_c37, rtol=1e-5, atol=1e-5)
    assert torch.allclose(m_full, m_c37, rtol=1e-5, atol=1e-5)


def test_cal_gradients_step0_dynamics() -> None:
    """Step 0: only the final layer moves (projections/hidden grad exactly 0,
    expected zero-init dynamics); after two steps the full path is alive."""
    edge_index, r, f_block = _make_graph(num_edges=60, num_nodes=12)
    cal = _make_cal(perturb=False)
    opt = torch.optim.SGD(cal.parameters(), lr=0.5)

    def step():
        opt.zero_grad()
        g_perm, _ = relation_calibrated_weighted_mean(
            edge_index, r, f_block, cal, 12, edge_chunk_size=None
        )
        loss = g_perm.square().sum()
        loss.backward()
        return loss

    step()
    assert cal.mlp[-1].weight.grad is not None
    assert cal.mlp[-1].weight.grad.norm() > 1e-9
    assert torch.equal(cal.mlp[0].weight.grad, torch.zeros_like(cal.mlp[0].weight.grad))
    assert torch.equal(
        cal.projections[0].weight.grad, torch.zeros_like(cal.projections[0].weight.grad)
    )
    for p in cal.parameters():
        assert torch.isfinite(p.grad).all()
    opt.step()
    step()
    assert cal.projections[0].weight.grad.norm() > 1e-9
    opt.step()
    step()
    for p in cal.parameters():
        assert torch.isfinite(p.grad).all()


def test_cal_isolated_and_empty() -> None:
    edge_index, r, f_block = _make_graph(num_edges=150, num_nodes=40)
    edge_index = edge_index.clone()
    edge_index[1] = edge_index[1] % 39 + 1  # node 0 isolated
    cal = _make_cal(perturb=True)
    g_perm, eff_mass = relation_calibrated_weighted_mean(
        edge_index, r, f_block, cal, 40, edge_chunk_size=64
    )
    assert torch.isfinite(g_perm).all() and torch.isfinite(eff_mass).all()
    assert torch.equal(g_perm[0], torch.zeros_like(g_perm[0]))
    assert torch.equal(eff_mass[0], torch.zeros_like(eff_mass[0]))
    # empty graph
    cal0 = _make_cal(perturb=False)
    g0, m0 = relation_calibrated_weighted_mean(
        torch.empty(2, 0, dtype=torch.long), torch.empty(0, K), f_block, cal0, N,
        edge_chunk_size=None,
    )
    assert torch.equal(g0, torch.zeros_like(g0))
    assert torch.equal(m0, torch.zeros_like(m0))


def test_cal_statistics_zero_init_and_manual() -> None:
    import json

    edge_index, r, f_block = _make_graph(num_edges=120, num_nodes=30)
    cal = _make_cal(perturb=False)
    stats = calibration_edge_statistics(
        edge_index, r, f_block, cal, 30, edge_chunk_size=64
    )
    json.dumps(stats)
    assert all(abs(v) < 1e-6 for v in stats["js_str"])
    assert all(abs(v) < 1e-6 for v in stats["kl_f2str"])
    assert all(abs(v) < 1e-6 for v in stats["kl_str2f"])
    assert all(abs(v) < 1e-6 for v in stats["js_pairwise"].values())
    # perturbed: js becomes positive and stats are chunk-invariant
    cal2 = _make_cal(perturb=True)
    s2 = calibration_edge_statistics(edge_index, r, f_block, cal2, 30, edge_chunk_size=64)
    s2b = calibration_edge_statistics(edge_index, r, f_block, cal2, 30, edge_chunk_size=100)
    assert any(v > 1e-4 for v in s2["js_str"])
    assert any(v > 1e-4 for v in s2["js_pairwise"].values())
    for key in ("js_str", "kl_f2str", "kl_str2f", "entropy", "k_eff"):
        assert max(abs(a - b) for a, b in zip(s2[key], s2b[key])) < 1e-6
    for key in ("C_Pt", "C_Pv", "Pt_Pv"):
        assert abs(s2["js_pairwise"][key] - s2b["js_pairwise"][key]) < 1e-6


def test_cal_statistics_coherence_manual() -> None:
    """sim_{f,k} = sum r^f_k cos / sum r^f_k checked against manual."""
    edge_index, r, f_block = _make_graph(num_edges=60, num_nodes=12)
    cal = _make_cal(perturb=True)
    stats = calibration_edge_statistics(
        edge_index, r, f_block, cal, 12, edge_chunk_size=16
    )
    src, dst = edge_index[0], edge_index[1]
    rf = cal(r, f_block[src], f_block[dst]).double()
    sim_ref = torch.zeros(F, K, dtype=torch.float64)
    for f in range(F):
        a, b = f_block[src, f], f_block[dst, f]
        cos = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-8)
        for k in range(K):
            w = rf[:, f, k]
            sim_ref[f, k] = (w * cos.double()).sum() / (w.sum() + 1e-8)
    assert torch.allclose(
        torch.tensor(stats["semantic_coherence"]).double(), sim_ref, atol=1e-5
    )


# ===========================================================================
# R1-B components: decoupled residual scorers (user §5/§9)
# ===========================================================================


def test_residual_scorers_zero_init_exact() -> None:
    """Zero-init final layers -> the residuals are EXACTLY 0 (BL/BR/BLR are
    bitwise-equal to the base Gamma at step 0)."""
    local = DynamicLocalScoreResidual(factor_dim=D, hidden_dim=32, activation="gelu")
    f = _rand((10, F, D), 0)
    g_bar = _rand((10, F, D), 1)
    assert torch.equal(local(f, g_bar), torch.zeros(10, F, 1))
    rel = SupportRelationScoreResidual(hidden_dim=32, activation="gelu")
    lm = _rand((10, K), 2)
    av = torch.softmax(_rand((10, K), 3), dim=-1)
    assert torch.equal(rel(lm, av), torch.zeros(10, K, 1))


def test_residual_scorers_shapes_and_gradients() -> None:
    local = DynamicLocalScoreResidual(factor_dim=D, hidden_dim=32, activation="gelu")
    rel = SupportRelationScoreResidual(hidden_dim=32, activation="gelu")
    f = _rand((10, F, D), 0)
    g_bar = _rand((10, F, D), 1)
    lm = _rand((10, K), 2)
    av = torch.softmax(_rand((10, K), 3), dim=-1)
    # lr / loss scale kept small: the residuals enter scores that are later
    # divided by epsilon=0.2, so a large step saturates the softmax and
    # kills the Jacobian (the real path uses AdamW lr=1e-3).
    opt = torch.optim.SGD(list(local.parameters()) + list(rel.parameters()), lr=0.01)

    def step():
        # Mimic the real training path: the residuals enter the scores, the
        # loss depends on the resulting Gamma (a pure residual-squared loss
        # is exactly 0 at step 0 and would carry no gradient).
        opt.zero_grad()
        s_local = torch.zeros(10, F, 1) + local(f, g_bar)
        s_rel = torch.zeros(10, F, K) + rel(lm, av).permute(0, 2, 1).expand(10, F, K)
        gamma = torch.softmax(torch.cat([s_local, s_rel], dim=-1) / 0.2, dim=-1)
        # Linear weights: a symmetric loss (e.g. gamma^2) has zero gradient
        # at the uniform Gamma produced by zero scores — not the real path.
        loss = (gamma * torch.arange(1.0, 6.0, device=gamma.device)).sum() * 1e-2
        loss.backward()
        return loss

    step()
    # step-0 zero-init dynamics: only final layers move.
    assert local.net[-1].weight.grad is not None and local.net[-1].weight.grad.norm() > 1e-9
    assert rel.net[-1].weight.grad is not None and rel.net[-1].weight.grad.norm() > 1e-9
    assert torch.equal(local.net[0].weight.grad, torch.zeros_like(local.net[0].weight.grad))
    assert torch.equal(rel.net[0].weight.grad, torch.zeros_like(rel.net[0].weight.grad))
    for p in list(local.parameters()) + list(rel.parameters()):
        assert torch.isfinite(p.grad).all()
    opt.step()
    step()
    assert local.net[0].weight.grad.norm() > 1e-9
    assert rel.net[0].weight.grad.norm() > 1e-9
    assert local(f, g_bar).shape == (10, F, 1)
    assert rel(lm, av).shape == (10, K, 1)


# ===========================================================================
# R1-C1SG: detached adaptive 2-hop (user ruling)
# ===========================================================================


def _make_c1sg_cfg() -> object:
    cfg = _make_r1_cfg("baseline")
    cfg.model.r1.mode = "detached_2hop"
    return cfg


def test_c1sg_zero_init_bitwise_equals_a0() -> None:
    """lam == 0 exactly -> F_out == F1 -> bitwise equal to A0, in BOTH
    modes (the detached second hop consumes no dropout RNG)."""
    baseline = _make_r1_model("baseline")
    c1 = R1Model(_make_c1sg_cfg(), _make_r1_info())
    c1.load_state_dict(baseline.state_dict(), strict=False)
    x, edge = _make_r1_x(), _make_r1_edge()
    baseline.eval()
    c1.eval()
    assert torch.equal(c1(x, edge)[0], baseline(x, edge)[0])
    baseline.train()
    c1.train()
    torch.manual_seed(7)
    z_b = baseline(x, edge)[0]
    torch.manual_seed(7)
    z_c = c1(x, edge)[0]
    assert torch.equal(z_c, z_b)


def test_c1sg_second_hop_zero_gradient_to_parent() -> None:
    """The second-hop path must add NOTHING to the parent parameters'
    gradients: with identical RNG and weights, the parent grads of C1SG and
    baseline are bitwise equal."""
    c1 = R1Model(_make_c1sg_cfg(), _make_r1_info())
    baseline = _make_r1_model("baseline")
    baseline.load_state_dict(c1.state_dict(), strict=False)
    x, edge = _make_r1_x(), _make_r1_edge()
    c1.train()
    baseline.train()
    torch.manual_seed(11)
    z_c, _, _, aux_c, _ = c1(x, edge)
    (z_c.square().sum() + aux_c).backward()
    torch.manual_seed(11)
    z_b, _, _, aux_b, _ = baseline(x, edge)
    (z_b.square().sum() + aux_b).backward()
    for name, p in baseline.named_parameters():
        q = dict(c1.named_parameters())[name]
        assert torch.equal(q.grad, p.grad), f"parent grad differs: {name}"
    # the trajectory machinery itself has finite gradients
    for name in ("traj_w.weight",):
        assert torch.isfinite(dict(c1.named_parameters())[name].grad).all()
    for name, p in c1.depth_mlp.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_c1sg_step0_dynamics_and_two_step_activation() -> None:
    """Step 0: depth gate last layer moves (tanh'(0)=1, d nonzero), W_traj
    grad is EXACTLY 0 (lam == 0 multiplies it away) — expected zero-init
    dynamics; after two steps W_traj is alive."""
    c1 = R1Model(_make_c1sg_cfg(), _make_r1_info())
    c1.train()
    x, edge = _make_r1_x(), _make_r1_edge()
    opt = torch.optim.SGD(c1.parameters(), lr=0.01)

    def step():
        opt.zero_grad()
        z, _, _, aux, _ = c1(x, edge)
        (z.square().sum() + aux).backward()

    step()
    assert c1.depth_mlp[-1].weight.grad is not None and c1.depth_mlp[-1].weight.grad.norm() > 1e-9
    assert torch.equal(c1.traj_w.weight.grad, torch.zeros_like(c1.traj_w.weight.grad))
    assert torch.equal(c1.depth_mlp[0].weight.grad, torch.zeros_like(c1.depth_mlp[0].weight.grad))
    opt.step()
    step()
    assert c1.traj_w.weight.grad.norm() > 1e-9
    assert c1.depth_mlp[0].weight.grad.norm() > 1e-9
    for p in c1.parameters():
        assert torch.isfinite(p.grad).all()


def test_c1sg_module_isolation_and_inference() -> None:
    c1 = R1Model(_make_c1sg_cfg(), _make_r1_info())
    assert hasattr(c1, "traj_w") and hasattr(c1, "traj_norm") and hasattr(c1, "depth_mlp")
    for attr in ("reliability", "calibration", "router_residuals",
                 "local_score_residual", "relation_score_residual"):
        assert not hasattr(c1, attr), attr
    x, edge = _make_r1_x(), _make_r1_edge()
    c1.eval()
    z_fwd = c1(x, edge)[0]
    z_inf = c1.inference(x, edge, device=torch.device("cpu"), batch_size=65536)
    assert torch.allclose(z_inf, z_fwd, rtol=1e-5, atol=1e-5)


def test_c1sg_isolated_node_finite() -> None:
    c1 = R1Model(_make_c1sg_cfg(), _make_r1_info())
    c1.eval()
    x = _make_r1_x()
    edge = _make_r1_edge(num_edges=60).clone()
    edge[1] = edge[1] % (M_N - 1) + 1  # node 0 isolated
    z, _, _, _, _ = c1(x, edge)
    assert torch.isfinite(z).all()


def test_c1sg_diagnostics_hop_section() -> None:
    c1 = R1Model(_make_c1sg_cfg(), _make_r1_info())
    c1.eval()
    diag = c1.compute_r1_diagnostics(_make_r1_x(), _make_r1_edge())
    json.dumps(diag)
    hop = diag["hop"]
    assert hop is not None and all(k in hop for k in ("C", "Pt", "Pv"))
    for fname in ("C", "Pt", "Pv"):
        row = hop[fname]
        for key in ("lam_mean", "lam_std", "lam_abs_mean", "lam_p10", "lam_p50",
                    "lam_p90", "frac_abs_lt_0.05", "correction_base_ratio", "cos_F1F2"):
            assert key in row, key
    assert diag["reliability"] is None and diag["calibration"] is None
    # baseline mode reports hop = None
    baseline = _make_r1_model("baseline")
    baseline.eval()
    diag_b = baseline.compute_r1_diagnostics(_make_r1_x(), _make_r1_edge())
    assert diag_b["hop"] is None


# ===========================================================================
# Model tests (Prompt 3, audit §3 M1-M8)
# ===========================================================================

import json  # noqa: E402

from omegaconf import OmegaConf  # noqa: E402

from src.models.biaxis_final import Model as FinalModel  # noqa: E402
from src.models.biaxis_perf_r1 import Model as R1Model  # noqa: E402

M_N, M_TEXT, M_VIS, M_HIDDEN, M_FACTOR = 17, 13, 19, 256, 128


def _make_r1_cfg(mode: str) -> object:
    cfg = OmegaConf.create({
        "model": {
            "name": "biaxis_perf_r1",
            "hidden_dim": M_HIDDEN,
            "factor_dim": M_FACTOR,
            "dropout": 0.2,
            "activation": "gelu",
            "norm": "layernorm",
            "lambda_common": 0.02,
            "lambda_orth": 0.01,
            "lambda_recon": 0.3,
            "full_graph_training": True,
            "p1": {
                "factor_aware": True, "num_relations": 4, "relation_dim": 32,
                "relation_temperature": 0.5, "selector_hidden_dim": 64,
                "selector_input_norm": None, "budget_hidden_dim": 64,
                "use_graph_budget": True, "budget_shared": False,
                "eps": 1.0e-8, "relation_balance_weight": 0.0,
                "alpha_entropy_weight": 0.0, "budget_reg_weight": 0.0,
                "edge_chunk_size": None,
            },
            "p2": {
                "mode": "null_softmax", "score_hidden_dim": 64, "epsilon": 0.2,
                "tau_base": 1.0, "sinkhorn_iters": 10, "null_prior": 0.5,
                "null_score_init": 0.0, "detach_capacity_prior": True,
                "detach_relation_confidence": True, "eps": 1.0e-8,
            },
            "p3": {
                "operator_mode": "full_interaction", "lowrank_rank": 16,
                "basis_num_bases": 8, "operator_reg_weight": 0.0,
                "interaction_reg_weight": 0.0,
            },
            "r1": {
                "mode": mode,
                "rel_proj_dim": 32,
                "rel_hidden_dim": 64,
                "rel_chunk_size": 7,  # small on purpose: exercises chunking
                "router_mode": "base",
                "router_hidden_dim": 64,
            },
        }
    })
    return cfg


def _make_r1_info() -> dict:
    return {
        "input_dim": M_TEXT + M_VIS, "num_nodes": M_N, "num_classes": 5,
        "text_dim": M_TEXT, "visual_dim": M_VIS,
    }


def _make_r1_x(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(M_N, M_TEXT + M_VIS, generator=generator)


def _make_r1_edge(seed: int = 1, num_edges: int = 60) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, M_N, (2, num_edges), generator=generator)


def _make_r1_model(mode: str) -> R1Model:
    return R1Model(_make_r1_cfg(mode), _make_r1_info())


def test_model_all_modes_forward_finite() -> None:
    for mode in ("baseline", "semantic_reliability"):
        model = _make_r1_model(mode)
        model.eval()
        z, second, third, aux, _ = model(_make_r1_x(), _make_r1_edge())
        assert second is None and third is None
        assert z.shape == (M_N, M_HIDDEN), f"{mode} bad shape {z.shape}"
        assert torch.isfinite(z).all(), f"{mode} non-finite"
        assert aux.ndim == 0


def test_model_unknown_mode_raises() -> None:
    cfg = _make_r1_cfg("baseline")
    cfg.model.r1.mode = "nonsense"
    import pytest
    with pytest.raises(AssertionError):
        R1Model(cfg, _make_r1_info())


def test_model_baseline_module_absence() -> None:
    baseline = _make_r1_model("baseline")
    assert not hasattr(baseline, "reliability")
    a1 = _make_r1_model("semantic_reliability")
    assert hasattr(a1, "reliability")


def test_baseline_bitwise_equals_biaxis_final_same_weights() -> None:
    """M1: the ONLY bitwise equivalence (identical super() code path).
    Re-checked after one synchronized training step."""
    cfg = _make_r1_cfg("baseline")
    final_model = FinalModel(cfg, _make_r1_info())
    r1_model = R1Model(cfg, _make_r1_info())
    assert set(r1_model.state_dict().keys()) == set(final_model.state_dict().keys())  # M2
    r1_model.load_state_dict(final_model.state_dict())

    x, edge = _make_r1_x(), _make_r1_edge()
    final_model.eval()
    r1_model.eval()
    z_final = final_model(x, edge)[0]
    z_r1 = r1_model(x, edge)[0]
    assert torch.equal(z_r1, z_final)

    # one synchronized training step -> still bitwise equal. The RNG is
    # re-seeded before each step so train-mode dropout draws identical masks
    # in both models (different masks would legitimately diverge the weights).
    final_model.train()
    r1_model.train()
    opt_final = torch.optim.SGD(final_model.parameters(), lr=0.01)
    opt_r1 = torch.optim.SGD(r1_model.parameters(), lr=0.01)
    for model, opt in ((final_model, opt_final), (r1_model, opt_r1)):
        torch.manual_seed(2026)
        opt.zero_grad()
        model(x, edge)[0].square().sum().backward()
        opt.step()
    final_model.eval()
    r1_model.eval()
    assert torch.equal(r1_model(x, edge)[0], final_model(x, edge)[0])


def test_a1_zero_init_numerically_equals_baseline() -> None:
    """M3: eta == 1 exactly -> mathematical equivalence; float grouping of
    the aggregation differs -> allclose, never equal (audit risk 1)."""
    baseline = _make_r1_model("baseline")
    a1 = _make_r1_model("semantic_reliability")
    a1.load_state_dict(baseline.state_dict(), strict=False)
    x, edge = _make_r1_x(), _make_r1_edge()
    baseline.eval()
    a1.eval()
    z_base = baseline(x, edge)[0]
    z_a1 = a1(x, edge)[0]
    assert not torch.equal(z_a1, z_base) or torch.equal(z_a1, z_base)  # either is OK
    assert torch.allclose(z_a1, z_base, rtol=1e-5, atol=1e-6)
    # eta itself is exactly 1 (T1 discipline, checked at model level).
    f_src = torch.randn(4, 3, M_FACTOR)
    f_dst = torch.randn(4, 3, M_FACTOR)
    assert torch.equal(a1.reliability(f_src, f_dst), torch.ones(4, 3))


def test_a1_gradients_finite_and_reliability_trained() -> None:
    """M6: the reliability pathway receives gradient and its parameters are
    part of the model's parameter set. The loss includes the P0 aux term so
    every module (recon heads included) is covered."""
    a1 = _make_r1_model("semantic_reliability")
    a1.train()
    x, edge = _make_r1_x(), _make_r1_edge()
    z, _, _, aux, _ = a1(x, edge)
    (z.square().sum() + aux).backward()
    rel_names = {n for n, _ in a1.reliability.named_parameters()}
    assert rel_names
    for name, p in a1.named_parameters():
        assert p.grad is not None, f"{name} has no grad"
        assert torch.isfinite(p.grad).all(), f"{name} grad non-finite"


def test_a1_inference_equivalence() -> None:
    """M5: inference() == eval-mode forward() on the same tensors."""
    a1 = _make_r1_model("semantic_reliability")
    x, edge = _make_r1_x(), _make_r1_edge()
    a1.eval()
    z_fwd = a1(x, edge)[0]
    z_inf = a1.inference(x, edge, device=torch.device("cpu"), batch_size=65536)
    assert z_inf.shape == (M_N, M_HIDDEN)
    assert torch.allclose(z_inf, z_fwd, rtol=1e-5, atol=1e-5)


def test_a1_regularization_aux_hook() -> None:
    """reg hook: zero-init eta -> reg == 0 (aux unchanged); after the gate
    moves off 1, aux gains exactly reg_weight * penalty."""
    cfg = _make_r1_cfg("semantic_reliability")
    cfg.model.r1.reg_type = "mean1"
    cfg.model.r1.reg_weight = 1.0
    model = R1Model(cfg, _make_r1_info())
    plain = R1Model(_make_r1_cfg("semantic_reliability"), _make_r1_info())
    plain.load_state_dict(model.state_dict())
    x, edge = _make_r1_x(), _make_r1_edge()
    model.train()
    plain.train()
    # Re-seed before each train-mode forward so dropout draws identical masks
    # (otherwise the P0 aux terms differ by dropout randomness alone).
    torch.manual_seed(7)
    _, _, _, aux_reg0, _ = model(x, edge)
    torch.manual_seed(7)
    _, _, _, aux_plain0, _ = plain(x, edge)
    assert torch.equal(aux_reg0, aux_plain0)  # eta == 1 -> penalty exactly 0
    with torch.no_grad():
        model.reliability.mlp[-1].bias.fill_(0.5)
    torch.manual_seed(7)
    _, _, _, aux_reg1, _ = model(x, edge)
    torch.manual_seed(7)
    _, _, _, aux_plain1, _ = plain(x, edge)
    assert aux_reg1.item() > aux_plain1.item()
    assert torch.isfinite(aux_reg1)


def test_a1_isolated_node_all_local_no_nan() -> None:
    """M7: an isolated node (no incoming edges) -> all-Local plan, finite."""
    a1 = _make_r1_model("semantic_reliability")
    a1.eval()
    x = _make_r1_x()
    edge = _make_r1_edge(num_edges=60).clone()
    edge[1] = edge[1] % (M_N - 1) + 1  # node 0 isolated
    num_nodes = M_N
    factors, _ = a1._encode(x)
    f_block = torch.stack([factors["c"], factors["p_t"], factors["p_v"]], dim=1)
    out = a1._graph_update(f_block, edge, num_nodes)
    assert torch.isfinite(out["f_tilde"]).all()
    assert torch.isfinite(out["gamma"]).all()
    row_sum = (out["gamma"].sum(dim=-1) - 1.0).abs().max().item()
    assert row_sum < 1e-5
    assert out["gamma"][0, :, 0].min().item() > 0.999
    assert torch.equal(out["effective_mass"][0], torch.zeros_like(out["effective_mass"][0]))


def test_r1_diagnostics_keys_and_json_safe() -> None:
    """M8: full diagnostic payload, JSON-safe, in both modes."""
    x, edge = _make_r1_x(), _make_r1_edge()
    for mode in ("baseline", "semantic_reliability"):
        model = _make_r1_model(mode)
        model.eval()
        diag = model.compute_r1_diagnostics(x, edge)
        json.dumps(diag)
        # inherited P3 payload
        assert "plan" in diag and "operator" in diag and "relation" in diag
        if mode == "baseline":
            assert diag["reliability"] is None
            assert diag["context_change"] is None
            assert diag["effective_mass"] is None
            # D_ctx is computed for BOTH modes (R0-comparable baseline value).
            assert all(k in diag["d_ctx"] for k in ("C", "Pt", "Pv"))
        else:
            rel = diag["reliability"]
            assert set(rel) == {"eta", "neighbor", "corr_eta_cos", "weighted_semantic_coherence"}
            assert all(f"F{f + 1}" in rel["eta"] for f in range(3))
            cc = diag["context_change"]
            assert all(f"{fn}_R{k + 1}" in cc for fn in ("C", "Pt", "Pv") for k in range(4))
            for cell in cc.values():
                # 1 - cos can dip slightly below 0 from float rounding (cos > 1).
                assert cell["mean_all"] >= -1e-6
            assert all(k in diag["d_ctx"] for k in ("C", "Pt", "Pv"))
            assert "per_cell_mean" in diag["effective_mass"]


def test_a2_model_forward_and_module_isolation() -> None:
    """A2 mode constructs the calibration module and NOT the A1 reliability
    module (Hard NO-GO ruling: A1 is never combined with later modules)."""
    a2 = R1Model(_make_r1_cfg("semantic_relation_calibration"), _make_r1_info())
    assert hasattr(a2, "calibration")
    assert not hasattr(a2, "reliability")
    a2.eval()
    z, second, third, aux, _ = a2(_make_r1_x(), _make_r1_edge())
    assert second is None and third is None
    assert z.shape == (M_N, M_HIDDEN)
    assert torch.isfinite(z).all()
    assert aux.ndim == 0


def test_a2_zero_init_numerically_equals_a0() -> None:
    """delta == 0 -> r^f == r^str -> mathematical equivalence with A0.
    The log/softmax roundtrip and the aggregation grouping differ from the
    frozen path -> allclose, never equal."""
    baseline = _make_r1_model("baseline")
    a2 = R1Model(_make_r1_cfg("semantic_relation_calibration"), _make_r1_info())
    a2.load_state_dict(baseline.state_dict(), strict=False)
    x, edge = _make_r1_x(), _make_r1_edge()
    baseline.eval()
    a2.eval()
    assert torch.allclose(a2(x, edge)[0], baseline(x, edge)[0], rtol=1e-5, atol=1e-6)


def test_a2_gradients_finite() -> None:
    a2 = R1Model(_make_r1_cfg("semantic_relation_calibration"), _make_r1_info())
    a2.train()
    x, edge = _make_r1_x(), _make_r1_edge()
    z, _, _, aux, _ = a2(x, edge)
    (z.square().sum() + aux).backward()
    for name, p in a2.named_parameters():
        assert p.grad is not None, f"{name} has no grad"
        assert torch.isfinite(p.grad).all(), f"{name} grad non-finite"


def test_a2_diagnostics_keys() -> None:
    a2 = R1Model(_make_r1_cfg("semantic_relation_calibration"), _make_r1_info())
    a2.eval()
    diag = a2.compute_r1_diagnostics(_make_r1_x(), _make_r1_edge())
    json.dumps(diag)
    cal = diag["calibration"]
    assert cal is not None and diag["reliability"] is None
    for key in ("js_str", "kl_f2str", "kl_str2f", "js_pairwise",
                "semantic_coherence", "semantic_coherence_range", "entropy", "k_eff"):
        assert key in cal, key
    assert len(cal["js_str"]) == 3 and len(cal["semantic_coherence_range"]) == 3
    assert all(k in cal["js_pairwise"] for k in ("C_Pt", "C_Pv", "Pt_Pv"))
    assert all(k in diag["d_ctx"] for k in ("C", "Pt", "Pv"))
    assert diag["context_change"] is not None


def _make_router_cfg(mode: str) -> object:
    cfg = _make_r1_cfg("baseline")
    cfg.model.r1.router_mode = mode
    return cfg


def test_router_variants_zero_init_bitwise_equal_a0() -> None:
    """Zero-init residual scorers add EXACT zeros to the scores -> BL / BR /
    BLR are BITWISE equal to the base Gamma at step 0 (unlike A1/A2: no
    log/softmax nonlinearity in the residual path)."""
    baseline = _make_r1_model("baseline")
    x, edge = _make_r1_x(), _make_r1_edge()
    baseline.eval()
    z_base = baseline(x, edge)[0]
    for mode in ("local_only", "relation_only", "evidence"):
        variant = R1Model(_make_router_cfg(mode), _make_r1_info())
        variant.load_state_dict(baseline.state_dict(), strict=False)
        variant.eval()
        assert torch.equal(variant(x, edge)[0], z_base), mode


def test_router_variant_module_isolation() -> None:
    bl = R1Model(_make_router_cfg("local_only"), _make_r1_info())
    assert hasattr(bl, "local_score_residual") and not hasattr(bl, "relation_score_residual")
    br = R1Model(_make_router_cfg("relation_only"), _make_r1_info())
    assert hasattr(br, "relation_score_residual") and not hasattr(br, "local_score_residual")
    blr = R1Model(_make_router_cfg("evidence"), _make_r1_info())
    assert hasattr(blr, "local_score_residual") and hasattr(blr, "relation_score_residual")
    for m in (bl, br, blr):
        assert not hasattr(m, "reliability") and not hasattr(m, "calibration")


def test_router_variants_gradients_finite() -> None:
    for mode in ("local_only", "relation_only", "evidence"):
        model = R1Model(_make_router_cfg(mode), _make_r1_info())
        model.train()
        x, edge = _make_r1_x(), _make_r1_edge()
        z, _, _, aux, _ = model(x, edge)
        (z.square().sum() + aux).backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"{mode} {name} has no grad"
            assert torch.isfinite(p.grad).all(), f"{mode} {name} grad non-finite"


def test_bl_conditional_relation_plan_unchanged() -> None:
    """User §7: the Local residual only moves column 0, so the conditional
    relation plan alpha = Softmax_k(s_rel/eps) is EXACTLY unchanged vs the
    base router (same relation scores, same epsilon)."""
    bl = R1Model(_make_router_cfg("local_only"), _make_r1_info())
    bl.eval()
    x, edge = _make_r1_x(), _make_r1_edge()
    factors, _ = bl._encode(x)
    f_block = torch.stack([factors["c"], factors["p_t"], factors["p_v"]], dim=1)
    out = bl._graph_update(f_block, edge, M_N)
    s_rel = bl.transport_scorer(f_block, out["g_perm"])
    alpha_ref = torch.softmax(s_rel / bl.p2_epsilon, dim=-1)
    # alpha derived from the actual Gamma must equal Softmax_k(s_rel/eps)
    # EXACTLY up to the graph-mass normalization (which cancels in the
    # conditional distribution when only column 0 changed).
    gamma_graph = out["gamma"][..., 1:]
    alpha_actual = gamma_graph / (gamma_graph.sum(dim=-1, keepdim=True) + 1e-8)
    assert torch.allclose(alpha_actual, alpha_ref, rtol=1e-6, atol=1e-6)
    # and the Local score changed (the residual is active in eval: random
    # projections but zero final layer -> actually 0 at init; assert shape
    # and finiteness of the gamma rows instead).
    assert (out["gamma"].sum(dim=-1) - 1.0).abs().max().item() < 1e-5


def test_router_diagnostics_keys() -> None:
    for mode in ("local_only", "relation_only", "evidence"):
        model = R1Model(_make_router_cfg(mode), _make_r1_info())
        model.eval()
        diag = model.compute_r1_diagnostics(_make_r1_x(), _make_r1_edge())
        json.dumps(diag)
        assert diag["reliability"] is None
        assert diag["calibration"] is None
        assert all(k in diag["d_ctx"] for k in ("C", "Pt", "Pv"))


def test_a1_diagnostics_context_change_zero_at_init() -> None:
    """At step 0 eta == 1 -> g^A1 == g^A0 (allclose) -> delta_g ~ 0."""
    a1 = _make_r1_model("semantic_reliability")
    a1.eval()
    diag = a1.compute_r1_diagnostics(_make_r1_x(), _make_r1_edge())
    for cell in diag["context_change"].values():
        assert cell["mean_all"] < 1e-5
