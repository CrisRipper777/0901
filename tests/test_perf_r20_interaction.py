"""Unit tests for the R2-0C semantic-interaction blocks (user §十三 1-9).

Covers: finiteness, real/mismatch/shuffle shape identity, fixed-permutation
determinism, permutation bijectivity (train/val index safety), all block
dimensions (2d / 4d / 3d / 4d / 6d / hidden+2d), no-Test guard.

CPU-only, deterministic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.analysis.perf_r20_utils as r20  # noqa: E402

N, D, H = 10, 128, 256
SHUFFLE_SEED = 20260904


def _factors() -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(3)
    return {f: torch.randn(N, D, generator=gen) for f in ("C", "Pt", "Pv")}


def _fixed_perm(n: int = N) -> torch.Tensor:
    return torch.randperm(n, generator=torch.Generator().manual_seed(SHUFFLE_SEED))


# --- 1. finiteness ----------------------------------------------------------


def test_all_blocks_finite():
    F = _factors()
    h_t, h_v = torch.randn(N, H), torch.randn(N, H)
    assert torch.isfinite(r20.factor_interaction_block(F)).all()
    assert torch.isfinite(r20.modal_interaction_block(h_t, h_v)).all()
    assert torch.isfinite(r20.cond_interaction_block(F["C"], F["Pt"])).all()


# --- 2. real / mismatch / shuffle shape identity -----------------------------


def test_real_and_permuted_shapes_identical():
    F = _factors()
    perm = _fixed_perm()
    I = r20.factor_interaction_block(F)
    I_shuf = I[perm]
    assert I_shuf.shape == I.shape
    I_cond = r20.cond_interaction_block(F["C"], F["Pt"])
    I_cond_mismatch = r20.cond_interaction_block(F["C"], F["Pt"][perm])
    assert I_cond_mismatch.shape == I_cond.shape


# --- 3. fixed permutation deterministic --------------------------------------


def test_fixed_perm_deterministic():
    a = _fixed_perm()
    b = _fixed_perm()
    assert torch.equal(a, b)


def test_fixed_perm_is_bijection():
    perm = _fixed_perm()
    assert sorted(perm.tolist()) == list(range(N))
    # row-shuffle of features is a bijection on rows -> all rows still present
    F = _factors()["C"]
    shuf = F[perm]
    assert set(torch.argwhere((shuf.unsqueeze(0) == F.unsqueeze(1)).all(-1)).T[1].tolist()) == set(range(N))


# --- 4. perm never touches train/val indices ----------------------------------


def test_perm_leaves_split_indices_untouched():
    from src.data.types import MAGData

    data = MAGData(
        name="toy", source="x", task="nc", x=torch.zeros(N, 8),
        edge_index=torch.empty(2, 0, dtype=torch.long), num_nodes=N,
        y=torch.arange(N, dtype=torch.long), train_idx=torch.arange(3),
        val_idx=torch.arange(3, 5), test_idx=torch.arange(5, N), num_classes=N,
    )
    r20.guard_no_test(data)
    train_before = data.train_idx.clone()
    val_before = data.val_idx.clone()
    F = _factors()["C"]
    _ = F[_fixed_perm()]  # the shuffle happens only on feature rows
    assert torch.equal(data.train_idx, train_before)
    assert torch.equal(data.val_idx, val_before)


# --- 5/6/7/8. dimensions ------------------------------------------------------


def test_factor_cond_shapes():
    F = _factors()
    r20.assert_feature_dim(r20.factor_interaction_block(F), N, 6 * D, "I_factor")
    r20.assert_feature_dim(F["C"], N, D, "F")
    r20.assert_feature_dim(F["Pt"], N, D, "N")
    r20.assert_feature_dim(r20.cond_interaction_block(F["C"], F["Pt"]), N, 2 * D, "I_ab")


def test_linear_and_inter_dims():
    F = _factors()
    N_ctx = F["Pt"]  # stand-in neighbor context [N, d]
    linear = r20.context_concat([F["C"], N_ctx])
    inter = r20.context_concat([F["C"], N_ctx, r20.cond_interaction_block(F["C"], N_ctx)])
    r20.assert_feature_dim(linear, N, 2 * D, "X_linear")
    r20.assert_feature_dim(inter, N, 4 * D, "X_inter")


def test_local_complete_dims():
    F = _factors()
    L = r20.context_concat([F[f] for f in ("C", "Pt", "Pv")])
    N_ctx = F["C"]
    r20.assert_feature_dim(L, N, 3 * D, "L")
    r20.assert_feature_dim(r20.context_concat([L, N_ctx]), N, 4 * D, "L+N")
    block = r20.context_concat([L, N_ctx, r20.cond_interaction_block(F["Pt"], N_ctx)])
    r20.assert_feature_dim(block, N, 6 * D, "L+N+I")


def test_final_residual_dims():
    z = torch.randn(N, H)
    F = _factors()
    block = r20.context_concat([z, r20.cond_interaction_block(F["C"], F["Pt"])])
    r20.assert_feature_dim(block, N, H + 2 * D, "Z+I_ab")


def test_modal_block_dims():
    h_t, h_v = torch.randn(N, H), torch.randn(N, H)
    r20.assert_feature_dim(r20.modal_interaction_block(h_t, h_v), N, 2 * H, "I_modal")
    r20.assert_feature_dim(r20.context_concat([h_t, h_v]), N, 2 * H, "modal base")
    r20.assert_feature_dim(
        r20.context_concat([h_t, h_v, r20.modal_interaction_block(h_t, h_v)]), N, 4 * H, "modal inter"
    )


# --- 9. no-Test guard ----------------------------------------------------------


def test_no_test_guard():
    from src.data.types import MAGData

    data = MAGData(
        name="toy", source="x", task="nc", x=torch.zeros(4, 8),
        edge_index=torch.empty(2, 0, dtype=torch.long), num_nodes=4,
        y=torch.arange(4, dtype=torch.long), train_idx=torch.tensor([0, 1]),
        val_idx=torch.tensor([2]), test_idx=torch.tensor([3]), num_classes=4,
    )
    r20.guard_no_test(data)
    assert data.test_idx is None
    assert data.y[3].item() == -1
    assert torch.equal(data.y[data.train_idx], torch.tensor([0, 1]))
