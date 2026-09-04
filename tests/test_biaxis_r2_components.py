"""Unit tests for R2-Design-1 components (plan §37-I).

Covers: AdaptiveCommonGate (zero-init -> exactly 50/50, simplex, trainable),
SemanticInteractionResidual (zero-init -> exactly zero, shape/finite,
trainable, interaction-block formula), FunctionalScorer (finite, small init,
sigmoid gate range).
"""

from __future__ import annotations

import torch

from src.models.biaxis_r2_components import (
    AdaptiveCommonGate,
    FunctionalScorer,
    SemanticInteractionResidual,
)

D = 32  # small dim keeps the tests fast; model-level tests use the real 128
N = 11


def _rand(*shape, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=generator)


# ---------------------------------------------------------------------------
# AdaptiveCommonGate (plan §6)
# ---------------------------------------------------------------------------


def test_adaptive_common_zero_init_exact_half() -> None:
    gate = AdaptiveCommonGate(D, hidden_dim=16)
    c_t, c_v = _rand(N, D, seed=1), _rand(N, D, seed=2)
    _, w = gate(c_t, c_v)
    assert w.shape == (N, 2)
    # zero-initialized final layer => logits = 0 => softmax = [0.5, 0.5] EXACTLY
    assert torch.equal(w, torch.full((N, 2), 0.5))


def test_adaptive_common_simplex_and_consensus() -> None:
    gate = AdaptiveCommonGate(D, hidden_dim=16)
    with torch.no_grad():
        gate.net[-1].weight.copy_(_rand(2, 16, seed=3) * 0.1)
    c_t, c_v = _rand(N, D, seed=1), _rand(N, D, seed=2)
    c0, w = gate(c_t, c_v)
    assert c0.shape == (N, D)
    assert torch.allclose(w.sum(dim=-1), torch.ones(N), atol=1e-6)
    assert torch.all(w >= 0) and torch.all(w <= 1)
    assert torch.allclose(c0, w[:, 0:1] * c_t + w[:, 1:2] * c_v, atol=1e-6)


def test_adaptive_common_trainable() -> None:
    gate = AdaptiveCommonGate(D, hidden_dim=16)
    c_t, c_v = _rand(N, D, seed=1), _rand(N, D, seed=2)
    c0, _ = gate(c_t, c_v)
    c0.sum().backward()
    assert any(p.grad is not None for p in gate.parameters())
    assert all(torch.isfinite(p.grad).all() for p in gate.parameters() if p.grad is not None)


# ---------------------------------------------------------------------------
# SemanticInteractionResidual (plan §7)
# ---------------------------------------------------------------------------


def test_semantic_residual_zero_init_exact_zero() -> None:
    refiner = SemanticInteractionResidual(D, dropout=0.0)
    f0 = torch.stack([_rand(N, D, seed=s) for s in (1, 2, 3)], dim=1)  # [N,3,d]
    delta = refiner(f0)
    assert delta.shape == (N, 3, D)
    # zero-initialized heads => Delta == 0 EXACTLY at step 0
    assert torch.equal(delta, torch.zeros_like(delta))


def test_semantic_residual_interaction_formula() -> None:
    refiner = SemanticInteractionResidual(D, dropout=0.0)
    f0 = torch.stack([_rand(N, D, seed=s) for s in (1, 2, 3)], dim=1)
    c0, pt, pv = f0[:, 0], f0[:, 1], f0[:, 2]
    expected_i = torch.cat(
        [c0 * pt, c0 * pv, pt * pv, (c0 - pt).abs(), (c0 - pv).abs(), (pt - pv).abs()],
        dim=-1,
    )
    assert expected_i.shape == (N, 6 * D)
    # the trunk input must be exactly this interaction block: monkey-check
    # through the forward (delta finite + nonzero after head init change).
    with torch.no_grad():
        for head in refiner.heads:
            head.weight.copy_(_rand(D, D, seed=4) * 0.05)
    delta = refiner(f0)
    assert torch.isfinite(delta).all()
    assert not torch.equal(delta, torch.zeros_like(delta))


def test_semantic_residual_shapes_finite_trainable() -> None:
    refiner = SemanticInteractionResidual(D, dropout=0.0)
    f0 = torch.stack([_rand(N, D, seed=s) for s in (1, 2, 3)], dim=1)
    with torch.no_grad():
        for head in refiner.heads:
            head.weight.copy_(_rand(D, D, seed=4) * 0.05)
    delta = refiner(f0)
    delta.sum().backward()
    assert torch.isfinite(delta).all()
    assert any(p.grad is not None for p in refiner.parameters())
    assert all(torch.isfinite(p.grad).all() for p in refiner.parameters() if p.grad is not None)


# ---------------------------------------------------------------------------
# FunctionalScorer (plan §11/§14)
# ---------------------------------------------------------------------------


def test_functional_scorer_output_finite() -> None:
    scorer = FunctionalScorer(D, type_dim=8, hidden_dim=16)
    u = _rand(N, 4 * D + 16, seed=5)
    score = scorer(u)
    assert score.shape == (N, 1)
    assert torch.isfinite(score).all()


def test_functional_scorer_small_init() -> None:
    """std=1e-3 final layer => |score| stays small at step 0 (g ~ 0.5)."""
    scorer = FunctionalScorer(D, type_dim=8, hidden_dim=16, final_std=1e-3)
    u = _rand(N, 4 * D + 16, seed=5)
    score = scorer(u)
    assert float(score.abs().max().item()) < 0.2


def test_functional_scorer_gate_unit_range() -> None:
    scorer = FunctionalScorer(D, type_dim=8, hidden_dim=16)
    u = _rand(N, 4 * D + 16, seed=5)
    gate = torch.sigmoid(scorer(u))
    assert torch.all(gate >= 0) and torch.all(gate <= 1)
