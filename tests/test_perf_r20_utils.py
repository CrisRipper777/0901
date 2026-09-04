"""Unit tests for R2-0 common utilities (plan §9 Prompt 1).

Covers the plan's required checks:
    - Splus is strictly topology-only (features / labels never change it);
    - weighted_neighbor_mean with all-one weights == neighbor_mean;
    - chunk / full equivalence of weighted_neighbor_mean;
    - isolated nodes -> finite zero contexts;
    - the setup wrapper neither reads nor exposes the test split;
    - the fixed Ridge protocol is R0's ridge_probe reused unchanged.

All tests are CPU-only and deterministic (no GPU atomics).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.analysis.perf_r0_utils as r0  # noqa: E402
import src.analysis.perf_r20_utils as r20  # noqa: E402
from src.data.types import MAGData  # noqa: E402
from src.models.biaxis_p1_components import neighbor_mean  # noqa: E402

_R20_SOURCE = (PROJECT_ROOT / "src" / "analysis" / "perf_r20_utils.py").read_text(encoding="utf-8")


def _tiny_graph() -> torch.Tensor:
    """Hand-computable graph: edges 0->1, 2->1, 1->3, 0->3 on 4 nodes."""
    return torch.tensor([[0, 2, 1, 0], [1, 1, 3, 3]], dtype=torch.long)


def _random_graph(num_nodes: int = 13, num_edges: int = 41, seed: int = 7) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, num_nodes, (2, num_edges), generator=gen)


def _strip_docstrings(text: str) -> str:
    """Remove triple-quoted strings so source scans ignore docstrings."""
    out, i = [], 0
    while i < len(text):
        t = text.find('"""', i)
        if t == -1:
            out.append(text[i:])
            break
        out.append(text[i:t])
        e = text.find('"""', t + 3)
        assert e != -1, "unterminated docstring"
        i = e + 3
    return "".join(out)


def _function_body(name: str) -> str:
    lines = _strip_docstrings(_R20_SOURCE).splitlines()
    start = next(j for j, ln in enumerate(lines) if ln.startswith(f"def {name}("))
    body: list[str] = []
    for ln in lines[start + 1 :]:
        if ln.startswith("def ") or (ln and not ln[0].isspace() and not ln.startswith("#")):
            break
        body.append(ln)
    return "\n".join(body)


def _fake_data(**kwargs) -> MAGData:
    defaults = dict(
        name="toy",
        source="x",
        task="nc",
        x=torch.zeros(4, 8),
        edge_index=torch.empty(2, 0, dtype=torch.long),
        num_nodes=4,
        y=torch.zeros(4, dtype=torch.long),
        train_idx=torch.tensor([0, 1]),
        val_idx=torch.tensor([2]),
        test_idx=torch.tensor([3]),
        num_classes=2,
    )
    defaults.update(kwargs)
    return MAGData(**defaults)


# ---------------------------------------------------------------------------
# No-test guard + protocol reuse (plan §9 items 1/7, unit list items 5/6)
# ---------------------------------------------------------------------------


def test_guard_cuts_test_idx_only():
    data = _fake_data()
    out = r20.guard_no_test(data)
    assert out is data
    assert data.test_idx is None
    assert data.train_idx is not None and data.val_idx is not None


def test_guard_requires_train_val():
    with pytest.raises(AssertionError):
        r20.guard_no_test(_fake_data(val_idx=None))
    with pytest.raises(AssertionError):
        r20.guard_no_test(_fake_data(train_idx=None))


def test_guard_requires_loaded_test_before_cut():
    with pytest.raises(AssertionError):
        r20.guard_no_test(_fake_data(test_idx=None))


def test_test_idx_only_referenced_inside_guard():
    """The only code in perf_r20_utils that touches test_idx is guard_no_test."""
    lines = _strip_docstrings(_R20_SOURCE).splitlines()
    body = _function_body("guard_no_test")
    hits = [ln.strip() for ln in lines if "test_idx" in ln]
    assert hits, "guard must contain the test_idx cut"
    for ln in hits:
        assert ln in [b.strip() for b in body.splitlines()], (
            f"test_idx referenced outside guard_no_test: {ln!r}"
        )


def test_ridge_probe_and_extract_forward_reused_unchanged():
    """Fixed Ridge protocol is R0's ridge_probe (StandardScaler +
    RidgeClassifier(alpha=1.0), TRAIN fit / VAL eval) — not rewritten."""
    assert r20.ridge_probe is r0.ridge_probe
    assert r20.extract_forward is r0.extract_forward
    assert r20._r0_load_setup is r0.load_setup
    assert r20.load_setup is not r0.load_setup  # wrapper adds the guard


# ---------------------------------------------------------------------------
# weighted_neighbor_mean (plan §9 item 3)
# ---------------------------------------------------------------------------


def test_weighted_mean_all_one_equals_neighbor_mean():
    ei = _random_graph()
    feats = torch.randn(13, 5)
    ref = neighbor_mean(ei, feats, 13)
    got = r20.weighted_neighbor_mean(ei, torch.ones(ei.size(1)), feats, 13, edge_chunk_size=3)
    assert torch.equal(got, ref)


def test_weighted_mean_chunk_full_equivalence():
    ei = _random_graph(seed=11)
    feats = torch.randn(13, 5)
    w = torch.rand(ei.size(1)) + 0.5
    full = r20.weighted_neighbor_mean(ei, w, feats, 13)
    chunked = r20.weighted_neighbor_mean(ei, w, feats, 13, edge_chunk_size=3)
    assert torch.equal(full, chunked)


def test_weighted_mean_src_dst_direction_hand_computed():
    ei = _tiny_graph()
    # F_j = (j+1) constant across feature dims -> exact float64 arithmetic
    feats = torch.arange(1.0, 5.0, dtype=torch.float64).unsqueeze(-1).expand(4, 2)
    w = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    got = r20.weighted_neighbor_mean(ei, w, feats, 4)
    eps = r20._EPS  # denominator guard: g = sum wF / (sum w + eps)
    g1 = (1.0 * 1.0 + 2.0 * 3.0) / (3.0 + eps)  # edges 0->1 (w1, F0), 2->1 (w2, F2)
    g3 = (3.0 * 2.0 + 4.0 * 1.0) / (7.0 + eps)  # edges 1->3 (w3, F1), 0->3 (w4, F0)
    assert torch.equal(got[0], torch.zeros(2, dtype=torch.float64))
    assert torch.equal(got[1], torch.full((2,), g1, dtype=torch.float64))
    assert torch.equal(got[2], torch.zeros(2, dtype=torch.float64))
    assert torch.equal(got[3], torch.full((2,), g3, dtype=torch.float64))


def test_weighted_mean_isolated_and_zero_sum_finite_zero():
    ei = _tiny_graph()
    feats = torch.randn(4, 3)
    got = r20.weighted_neighbor_mean(ei, torch.ones(4), feats, 4)
    assert torch.all(got[0] == 0) and torch.isfinite(got[0]).all()
    # zero total weight on an edge -> finite zero as well
    got0 = r20.weighted_neighbor_mean(ei, torch.zeros(4), feats, 4)
    assert torch.all(got0 == 0) and torch.isfinite(got0).all()


def test_weighted_mean_empty_graph_zeros():
    ei = torch.empty(2, 0, dtype=torch.long)
    got = r20.weighted_neighbor_mean(ei, torch.empty(0), torch.randn(5, 3), 5)
    assert torch.all(got == 0) and torch.isfinite(got).all()


def test_weighted_mean_rejects_weight_mismatch():
    with pytest.raises(ValueError):
        r20.weighted_neighbor_mean(_random_graph(), torch.ones(5), torch.randn(13, 3), 13)


# ---------------------------------------------------------------------------
# Splus (plan §9 item 4)
# ---------------------------------------------------------------------------


def test_raw_splus_hand_computed():
    ln3 = float(torch.log(torch.tensor(3.0)))
    raw = r20.raw_splus(_tiny_graph(), 4)
    assert raw.shape == (4, 8)
    exp = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [ln3, 0.0, 0.0, 0.0, 0.0, 0.0, ln3, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [ln3, ln3 / 2.0, 0.0, 0.0, 1.0, 1.0, ln3 / 2.0, ln3 / 2.0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(raw, exp, atol=1e-6)


def test_splus_finite_unit_rows_and_deterministic():
    ei = _random_graph(17, 60, seed=3)
    a = r20.compute_splus(ei, 17)
    b = r20.compute_splus(ei, 17)
    assert torch.equal(a, b)
    assert torch.isfinite(a).all()
    norms = a.norm(dim=-1)
    nonzero = norms > 0
    assert torch.allclose(norms[nonzero], torch.ones_like(norms[nonzero]), atol=1e-6)


def test_splus_isolated_node_finite():
    ei = _tiny_graph()
    s = r20.compute_splus(ei, 4)
    assert torch.isfinite(s).all()
    # node 0 and 2 are isolated: zero raw rows -> finite constant z-scored rows
    assert torch.isfinite(s[0]).all() and torch.isfinite(s[2]).all()


def test_splus_strictly_topology_only():
    """Source-level check: raw_splus / compute_splus never read features,
    factors, labels or logits — their inputs are edge_index / num_nodes."""
    for name in ("raw_splus", "compute_splus"):
        body = _function_body(name)
        for forbidden in ("features", "labels", "factors", "logits", "y[", ".y"):
            assert forbidden not in body, f"{name} body references {forbidden!r}"


# ---------------------------------------------------------------------------
# Factor aliases (plan §9 item 2)
# ---------------------------------------------------------------------------


def test_factor_tensor_aliases():
    c, pt, pv = torch.randn(5, 4), torch.randn(5, 4), torch.randn(5, 4)
    fex = {"factors": {"c": c, "p_t": pt, "p_v": pv}}
    assert r20.factor_tensor(fex, "C") is c
    assert r20.factor_tensor(fex, "Pt") is pt
    assert r20.factor_tensor(fex, "Pv") is pv
    with pytest.raises(KeyError):
        r20.factor_tensor(fex, "q")


def test_factor_block_passthrough():
    fb = torch.randn(6, 3, 4)
    assert r20.factor_block({"f_block": fb}) is fb


def test_context_concat():
    a, b = torch.randn(4, 2), torch.randn(4, 3)
    out = r20.context_concat([a, b])
    assert out.shape == (4, 5)
    assert torch.equal(out[:, :2], a) and torch.equal(out[:, 2:], b)


def test_round1_datasets_and_seeds():
    assert r20.DATASETS == ["Movies", "Toys", "Grocery"]
    assert r20.SEEDS == [42, 43, 44]
    assert r20.FACTOR_NAMES == ["C", "Pt", "Pv"]
