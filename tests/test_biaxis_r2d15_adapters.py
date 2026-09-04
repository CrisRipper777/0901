"""Unit tests for the R2-Design-1.5 frozen-B0 adapters (plan §33).

Covers: exact-zero step-0 output for every adapter (fresh init degenerates
to the frozen B0 factors), D2/D3 parameter-count matching, FiLM zero-init,
shapes/finiteness, and determinism under a fixed node permutation.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.models.biaxis_r2d15_adapters import (
    ConcatVectorAdapter,
    FiLMVectorAdapter,
    ProdDiffVectorAdapter,
    ScalarAdapter,
    build_adapter,
)

D = 32
N = 17
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _rand_f(n: int = N, d: int = D, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n, 3, d, generator=generator)


def test_all_adapters_zero_init_exact_zero() -> None:
    """Plan §33: HEAD/D2/D3/D4 (and D1) at step 0 must output EXACTLY zero,
    so a fresh adapter degenerates to the frozen B0 factor output."""
    f_pre, n_block = _rand_f(seed=1), _rand_f(seed=2)
    for name, adapter in (
        ("D1", ScalarAdapter(D)),
        ("D2", ConcatVectorAdapter(D)),
        ("D3", ProdDiffVectorAdapter(D)),
        ("D4", FiLMVectorAdapter(D)),
    ):
        delta = adapter(f_pre, n_block)
        assert delta.shape == (N, 3, D), name
        assert torch.equal(delta, torch.zeros_like(delta)), f"{name} not exactly zero at init"


def test_d2_d3_parameter_count_matched() -> None:
    d2 = ConcatVectorAdapter(D)
    d3 = ProdDiffVectorAdapter(D)
    p2 = sum(p.numel() for p in d2.parameters())
    p3 = sum(p.numel() for p in d3.parameters())
    assert p2 == p3, f"D2={p2} vs D3={p3} not matched"
    # structural match too: same named shapes
    for (n2, t2), (n3, t3) in zip(d2.named_parameters(), d3.named_parameters()):
        assert n2 == n3 and t2.shape == t3.shape


def test_film_delta_gamma_beta_zero_init() -> None:
    adapter = FiLMVectorAdapter(D)
    final = adapter.net[-1]
    assert torch.equal(final.weight, torch.zeros_like(final.weight))
    assert torch.equal(final.bias, torch.zeros_like(final.bias))
    # and through the forward: delta == 0 exactly (covered above), so here
    # verify the raw [delta_gamma, beta] head is zero by checking the net
    # output on the interaction input
    f_pre, n_block = _rand_f(seed=1), _rand_f(seed=2)
    num_nodes = N
    src_t = adapter.src_type_emb.weight
    tgt_t = adapter.tgt_type_emb.weight
    u = torch.cat(
        [
            f_pre[:, 0], n_block[:, 0], f_pre[:, 0] * n_block[:, 0],
            (f_pre[:, 0] - n_block[:, 0]).abs(),
            src_t[0].unsqueeze(0).expand(num_nodes, -1),
            tgt_t[0].unsqueeze(0).expand(num_nodes, -1),
        ],
        dim=-1,
    )
    assert torch.equal(adapter.net(u), torch.zeros_like(adapter.net(u)))


def test_scalar_adapter_alpha_zero_init() -> None:
    adapter = ScalarAdapter(D)
    assert torch.equal(adapter.alpha, torch.zeros(3))
    assert sum(p.numel() for p in adapter.parameters()) > 0  # but has params


def test_adapter_forward_finite_and_deterministic() -> None:
    f_pre, n_block = _rand_f(seed=3), _rand_f(seed=4)
    for name in ("D1", "D2", "D3", "D4"):
        adapter = build_adapter(name, D)
        with torch.no_grad():
            # perturb off zero-init so the forward is nontrivial
            for p in adapter.parameters():
                p.add_(0.01 * torch.randn_like(p))
        out1 = adapter(f_pre, n_block)
        out2 = adapter(f_pre, n_block)
        assert torch.isfinite(out1).all(), name
        assert torch.equal(out1, out2), f"{name} not deterministic"


def test_adapter_permuted_context_deterministic() -> None:
    """Permuting the context rows along the fixed node permutation is
    deterministic (same input twice -> same output) and changes the output
    relative to the real (unpermuted) contexts — the mismatch-control
    building block."""
    adapter = ConcatVectorAdapter(D)
    with torch.no_grad():
        for p in adapter.parameters():
            p.add_(0.05 * torch.randn_like(p))
    f_pre, n_block = _rand_f(seed=5), _rand_f(seed=6)
    generator = torch.Generator().manual_seed(20260904)
    perm = torch.randperm(N, generator=generator)
    out_real = adapter(f_pre, n_block)
    out_perm_a = adapter(f_pre, n_block[perm])
    out_perm_b = adapter(f_pre, n_block[perm])
    assert torch.equal(out_perm_a, out_perm_b)  # deterministic
    assert torch.isfinite(out_perm_a).all()
    assert not torch.equal(out_real, out_perm_a)  # permutation actually matters


def test_build_adapter_names() -> None:
    for name in ("D1", "D2", "D3", "D4"):
        assert build_adapter(name, D) is not None
    try:
        build_adapter("HEAD", D)
        raise AssertionError("HEAD must not be an adapter")
    except ValueError:
        pass


def test_no_test_logic_in_adapter_sources() -> None:
    path = PROJECT_ROOT / "src" / "models" / "biaxis_r2d15_adapters.py"
    text = path.read_text(encoding="utf-8")
    for token in ("test_idx", "test_acc", "data.y[", "test_mask"):
        assert token not in text, f"forbidden token {token!r}"
