"""Unit tests for the R2-0B explicit structural-function basis (user §十五).

Covers the pre-run sanity list:
    A. Splus unchanged / topology-only (re-checks the r20 compute_splus);
    B. G1 == neighbor_mean(F);
    C. G2 == neighbor_mean(G1);
    D. w_sim / w_diff finite and w_sim + w_diff ~ 1 + 2eps;
    E. weighted aggregation chunk/full equivalent (channel level);
    F. strict dimension asserts for all probe blocks;
    G. no-Test guard (re-checks guard_no_test).

CPU-only, deterministic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.analysis.perf_r20_utils as r20  # noqa: E402
from src.models.biaxis_p1_components import neighbor_mean  # noqa: E402


def _tiny_graph() -> torch.Tensor:
    """Edges 0->1, 2->1, 1->3, 0->3 on 4 nodes (node 0/2 isolated as dst)."""
    return torch.tensor([[0, 2, 1, 0], [1, 1, 3, 3]], dtype=torch.long)


def _random_graph(num_nodes: int = 13, num_edges: int = 41, seed: int = 7) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, num_nodes, (2, num_edges), generator=gen)


# --- A. Splus unchanged / topology-only ------------------------------------


def test_splus_topology_only_and_deterministic():
    ei = _random_graph(seed=5)
    a = r20.compute_splus(ei, 13)
    b = r20.compute_splus(ei.clone(), 13)
    assert torch.equal(a, b)
    assert torch.isfinite(a).all()


# --- D. edge weights --------------------------------------------------------


def test_edge_weights_sum_to_one_plus_2eps():
    ei = _random_graph(seed=9)
    splus = r20.compute_splus(ei, 13)
    w_sim, w_diff = r20.structural_edge_weights(ei, splus)
    assert torch.isfinite(w_sim).all() and torch.isfinite(w_diff).all()
    assert w_sim.shape == (ei.size(1),) and w_diff.shape == (ei.size(1),)
    total = w_sim + w_diff
    expected = 1.0 + 2.0 * r20._EPS
    assert torch.allclose(total, torch.full_like(total, expected), atol=1e-6)


def test_edge_weights_range():
    ei = _random_graph(seed=11)
    splus = r20.compute_splus(ei, 13)
    w_sim, w_diff = r20.structural_edge_weights(ei, splus)
    # c in [-1, 1] (up to float rounding) -> weights in [eps, 1+eps]
    assert (w_sim >= 0).all() and (w_diff >= 0).all()
    assert (w_sim <= 1.0 + r20._EPS + 1e-6).all()
    assert (w_diff <= 1.0 + r20._EPS + 1e-6).all()


def test_edge_weights_identical_rows_never_negative():
    """Numerical-hardening regression: identical / near-identical Splus rows
    must not produce negative contrast weights (float32 cosine overshoot)."""
    # rows that are bitwise identical -> c must be clamped to exactly 1
    ei = torch.tensor([[0, 1, 0, 2], [2, 2, 3, 3]], dtype=torch.long)
    v = torch.randn(8)
    v = v / v.norm()
    splus = torch.stack([v, v, v, v], dim=0)  # all 4 rows identical, unit norm
    w_sim, w_diff = r20.structural_edge_weights(ei, splus)
    assert torch.isfinite(w_sim).all() and torch.isfinite(w_diff).all()
    assert (w_sim >= 0).all() and (w_diff >= 0).all()
    assert torch.allclose(w_diff, torch.full_like(w_diff, r20._EPS), atol=1e-9)
    # near-identical rows (float32 noise around 1) stay >= 0
    v2 = v + torch.randn(8) * 1e-7
    v2 = v2 / v2.norm()
    splus2 = torch.stack([v, v2, v, v2], dim=0)
    _, w_diff2 = r20.structural_edge_weights(ei, splus2)
    assert (w_diff2 >= 0).all()


def test_edge_weights_deterministic_same_input():
    ei = _random_graph(seed=13)
    splus = r20.compute_splus(ei, 13)
    a = r20.structural_edge_weights(ei, splus)
    b = r20.structural_edge_weights(ei, splus)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


# --- B/C/E. explicit channels ----------------------------------------------


def test_channels_g1_equals_neighbor_mean():
    ei = _random_graph()
    f = torch.randn(13, 5)
    splus = r20.compute_splus(ei, 13)
    ch = r20.explicit_channels(ei, f, splus, 13, edge_chunk_size=3)
    assert torch.equal(ch["G1"], neighbor_mean(ei, f, 13))


def test_channels_g2_equals_neighbor_mean_of_g1():
    ei = _random_graph()
    f = torch.randn(13, 5)
    splus = r20.compute_splus(ei, 13)
    ch = r20.explicit_channels(ei, f, splus, 13, edge_chunk_size=3)
    assert torch.equal(ch["G2"], neighbor_mean(ei, ch["G1"], 13))


def test_channels_chunk_full_equivalence():
    ei = _random_graph(seed=15)
    f = torch.randn(13, 5)
    splus = r20.compute_splus(ei, 13)
    full = r20.explicit_channels(ei, f, splus, 13)
    chunked = r20.explicit_channels(ei, f, splus, 13, edge_chunk_size=3)
    for name in ("G1", "G2", "Gsim", "Gdiff"):
        assert torch.equal(full[name], chunked[name]), name


def test_channels_isolated_finite_zero():
    ei = _tiny_graph()
    f = torch.randn(4, 3)
    splus = r20.compute_splus(ei, 4)
    ch = r20.explicit_channels(ei, f, splus, 4)
    for name in ("G1", "G2", "Gsim", "Gdiff"):
        assert torch.isfinite(ch[name]).all(), name
        assert torch.all(ch[name][0] == 0), name  # node 0 has no incoming edges
        assert torch.all(ch[name][2] == 0), name


def test_channels_no_gradient_dependency_on_f_splus_shape():
    ei = _random_graph()
    f = torch.randn(13, 5)
    splus = r20.compute_splus(ei, 13)
    ch = r20.explicit_channels(ei, f, splus, 13)
    for name, tensor in ch.items():
        assert tensor.shape == (13, 5), name
        assert tensor.dtype == torch.float32, name


# --- F. dimension asserts ---------------------------------------------------


def test_assert_feature_dim_ok_and_raises():
    x = torch.randn(10, 128)
    r20.assert_feature_dim(x, 10, 128, "t")  # no raise
    with pytest.raises(AssertionError):
        r20.assert_feature_dim(x, 10, 129, "t")
    with pytest.raises(AssertionError):
        r20.assert_feature_dim(torch.full((10, 128), float("nan")), 10, 128, "t")


def test_probe_block_shapes():
    """Plan §十五 F: per-factor current/explicit = 5d; joint = 15d;
    final current/explicit = hidden + 12d (fake tensors, d=8, K=2, h=16)."""
    n, d, k, h = 10, 8, 2, 16
    f = {name: torch.randn(n, d) for name in ("C", "Pt", "Pv")}
    z = torch.randn(n, h)
    g_perm = torch.randn(n, 3, k, d)  # [N, F, K, d]
    ch = {name: r20.explicit_channels(torch.randint(0, n, (2, 40)), t, torch.randn(n, 8), n)
          for name, t in f.items()}

    L = r20.context_concat([f[name] for name in ("C", "Pt", "Pv")])
    r20.assert_feature_dim(L, n, 3 * d, "L")
    # per-factor current [F | g_R1..g_RK] and explicit [F | G1|G2|Gsim|Gdiff]
    for i, name in enumerate(("C", "Pt", "Pv")):
        current = r20.context_concat([f[name], g_perm[:, i].reshape(n, -1)])
        r20.assert_feature_dim(current, n, (1 + k) * d, f"per-factor current {name}")
        explicit = r20.context_concat(
            [f[name], ch[name]["G1"], ch[name]["G2"], ch[name]["Gsim"], ch[name]["Gdiff"]]
        )
        r20.assert_feature_dim(explicit, n, 5 * d, f"per-factor explicit {name}")
    # joint current = (3 + 3K)d (production K=4 -> 15d);
    # joint explicit ALWAYS = 15d (4 channels per factor, independent of K).
    joint_current = r20.context_concat([L] + [g_perm[:, i].reshape(n, -1) for i in range(3)])
    r20.assert_feature_dim(joint_current, n, (3 + 3 * k) * d, "joint current")
    joint_explicit = r20.context_concat(
        [L]
        + [
            r20.context_concat([ch[name][c] for c in ("G1", "G2", "Gsim", "Gdiff")])
            for name in ("C", "Pt", "Pv")
        ]
    )
    r20.assert_feature_dim(joint_explicit, n, 15 * d, "joint explicit")
    # final current = hidden + 3Kd (production K=4 -> hidden + 12d);
    # final explicit ALWAYS = hidden + 12d.
    b_current = r20.context_concat([g_perm[:, i].reshape(n, -1) for i in range(3)])
    b_explicit = r20.context_concat(
        [r20.context_concat([ch[name][c] for c in ("G1", "G2", "Gsim", "Gdiff")]) for name in ("C", "Pt", "Pv")]
    )
    r20.assert_feature_dim(r20.context_concat([z, b_current]), n, h + 3 * k * d, "final current")
    r20.assert_feature_dim(r20.context_concat([z, b_explicit]), n, h + 12 * d, "final explicit")


# --- G. no-Test guard -------------------------------------------------------


def test_no_test_guard_still_passes():
    from src.data.types import MAGData

    data = MAGData(
        name="toy", source="x", task="nc", x=torch.zeros(4, 8),
        edge_index=torch.empty(2, 0, dtype=torch.long), num_nodes=4,
        y=torch.arange(4, dtype=torch.long), train_idx=torch.tensor([0, 1]),
        val_idx=torch.tensor([2]), test_idx=torch.tensor([3]), num_classes=4,
    )
    r20.guard_no_test(data)
    assert data.test_idx is None
    assert torch.equal(data.y[data.train_idx], torch.tensor([0, 1]))
    assert torch.equal(data.y[data.val_idx], torch.tensor([2]))
    assert data.y[3].item() == -1
