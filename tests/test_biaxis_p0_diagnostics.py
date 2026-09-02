from __future__ import annotations

import math

import torch

from src.utils.biaxis_p0_diagnostics import (
    aggregate_fixed_gcn,
    compute_conflict_statistics,
    compute_edge_factor_statistics,
    compute_factor_sanity,
    propagate_fixed_gcn,
)


def _make_factors(num_nodes: int = 50, factor_dim: int = 8, seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    c = torch.randn(num_nodes, factor_dim, generator=generator)
    return {
        "c": c,
        "c_t": c,
        "c_v": c,
        "p_t": c.clone(),
        "p_v": c.clone(),
        "z_local": torch.randn(num_nodes, 16, generator=generator),
    }


# ---------------------------------------------------------------------------
# Fixed GCN propagation
# ---------------------------------------------------------------------------


def test_aggregate_fixed_gcn_matches_hand_computation() -> None:
    # Undirected graph: 0-1, 1-2, 2-0, 3-4 (both directions given).
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 0, 3, 4], [1, 0, 2, 1, 0, 2, 4, 3]], dtype=torch.long
    )
    h = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    got = aggregate_fixed_gcn(h, edge_index, num_nodes=5)

    deg_plus_one = torch.tensor([3.0, 3.0, 3.0, 2.0, 2.0])  # A+I row sums
    inv_sqrt = 1.0 / deg_plus_one.sqrt()
    expected = torch.zeros_like(h)
    for v in range(5):
        neighbors = torch.tensor(
            [[1, 2], [0, 2], [0, 1], [4], [3]][v], dtype=torch.long
        )
        acc = inv_sqrt[v] * inv_sqrt[v] * h[v]  # self-loop weight: 1/D_v
        for u in neighbors:
            acc = acc + inv_sqrt[v] * inv_sqrt[u] * h[u]
        expected[v] = acc
    assert torch.allclose(got, expected, atol=1e-5), f"got {got} expected {expected}"


def test_aggregate_fixed_gcn_isolated_node_keeps_own_feature() -> None:
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    h = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    got = aggregate_fixed_gcn(h, edge_index, num_nodes=3)
    assert torch.allclose(got[2], h[2], atol=1e-6)  # D=1, only self-loop
    # LN is scale-invariant up to eps: LN(2h) == LN(h) for an isolated node.
    prop = propagate_fixed_gcn(h, edge_index, num_nodes=3)
    assert torch.allclose(prop[2], torch.nn.functional.layer_norm(h[2], (2,)), atol=1e-4)


# ---------------------------------------------------------------------------
# Factor sanity
# ---------------------------------------------------------------------------


def test_factor_sanity_identical_factors() -> None:
    factors = _make_factors()
    stats = compute_factor_sanity(factors, max_nodes=100)
    assert abs(stats["common_sim"] - 1.0) < 1e-5
    assert abs(stats["private_sim"] - 1.0) < 1e-5
    assert stats["cp_overlap_t"] >= 0
    assert stats["num_nodes_sampled"] == 50


def _factors_from_matrix(matrix: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "c": matrix,
        "c_t": matrix,
        "c_v": matrix,
        "p_t": matrix.clone(),
        "p_v": matrix.clone(),
        "z_local": torch.randn(matrix.size(0), 16),
    }


def test_effective_rank_rank_one_and_full_rank() -> None:
    generator = torch.Generator().manual_seed(0)
    rank_one = torch.randn(200, 8, generator=generator)[:, :1] @ torch.randn(1, 8, generator=generator)
    stats = compute_factor_sanity(_factors_from_matrix(rank_one), max_nodes=100)
    assert stats["effrank_c"] < 1.5

    full_rank = torch.randn(1000, 32, generator=generator)
    stats = compute_factor_sanity(_factors_from_matrix(full_rank), max_nodes=2000)
    assert stats["effrank_pt"] > 20


# ---------------------------------------------------------------------------
# Edge statistics
# ---------------------------------------------------------------------------


def test_edge_statistics_identical_factors() -> None:
    num_nodes = 30
    factors = _make_factors(num_nodes=num_nodes)
    generator = torch.Generator().manual_seed(1)
    edge_index = torch.randint(0, num_nodes, (2, 100), generator=generator)
    stats = compute_edge_factor_statistics(factors, edge_index, max_edges=1000)
    for pair in ("C_Pt", "C_Pv", "Pt_Pv"):
        assert abs(stats[f"rho_{pair}"] - 1.0) < 1e-5
        assert abs(stats[f"mean_abs_gap_{pair}"]) < 1e-6
        assert stats[f"frac_gap_gt_0.25_{pair}"] == 0.0
        assert abs(stats[f"jaccard_top10_{pair}"] - 1.0) < 1e-6
        assert abs(stats[f"jaccard_top20_{pair}"] - 1.0) < 1e-6


def test_edge_statistics_canonicalizes_undirected_edges() -> None:
    num_nodes = 10
    factors = _make_factors(num_nodes=num_nodes)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4], [1, 0, 0, 2, 3, 2, 4, 3, 4]], dtype=torch.long
    )
    # canonical undirected edges: (0,1), (2,3), (3,4) -> 3
    # (directions deduped; self-loops (2,2) and (4,4) dropped)
    stats = compute_edge_factor_statistics(factors, edge_index, max_edges=1000)
    assert stats["num_edges_sampled"] == 3


def test_edge_statistics_sampling_is_seeded_and_capped() -> None:
    num_nodes = 200
    factors = _make_factors(num_nodes=num_nodes)
    generator = torch.Generator().manual_seed(2)
    edge_index = torch.randint(0, num_nodes, (2, 5000), generator=generator)
    stats_a = compute_edge_factor_statistics(factors, edge_index, max_edges=1000, seed=42)
    stats_b = compute_edge_factor_statistics(factors, edge_index, max_edges=1000, seed=42)
    assert stats_a["num_edges_sampled"] == 1000
    assert stats_a == stats_b


def test_spearman_constant_input_is_nan() -> None:
    from src.utils.biaxis_p0_diagnostics import _spearman

    const = torch.ones(10)
    assert math.isnan(_spearman(const, const))


# ---------------------------------------------------------------------------
# Conflict statistics
# ---------------------------------------------------------------------------


def test_conflict_statistics_known_patterns() -> None:
    delta_c = torch.tensor([1.0, 1.0, -1.0, -1.0])
    delta_t = torch.tensor([1.0, 1.0, -1.0, -1.0])
    delta_v = torch.tensor([1.0, -1.0, 1.0, -1.0])
    stats = compute_conflict_statistics({"C": delta_c, "Pt": delta_t, "Pv": delta_v})
    assert stats["conflict_C_Pt"] == 0.0
    assert abs(stats["corr_spearman_delta_C_Pt"] - 1.0) < 1e-6
    assert stats["conflict_C_Pv"] == 0.5
    assert stats["pattern_all_help"] == 0.25
    assert stats["pattern_all_hurt"] == 0.25
    assert stats["frac_positive"]["C"] == 0.5
