"""Unit tests for the R3 Ownership-Structured Semantic Transition Network
(docs/R3_Ownership_Structured_Transition_阶段推进计划.md, §19 R3-0B).

Covers the plan's T1-T7: shapes, diagonal-only degeneration (T2), exact
identity with frozen-zero layer scale (T3), edge-order invariance of the
mean aggregation (T4), gradient audit (T5), off-diagonal init ratio (T6),
forward/inference parity (T7), plus eval-mode aux silence and the aux-info
whitelist passthrough for the r3_* transition stats.
"""

from __future__ import annotations

import torch
import pytest

from src.models.biaxis_r3_components import OwnershipTransitionLayer
from src.tasks.common import update_aux_info_stats

N = 32
D = 16
HIDDEN = 24
TEXT_DIM = 8
VISUAL_DIM = 16
EPS = 1e-8


class _CfgNode(dict):
    def __getattr__(self, name: str):
        return self[name]


def _transition_cfg(**overrides) -> _CfgNode:
    t = dict(
        num_transition_layers=2,
        relation_dim=16,
        factor_id_dim=8,
        context_dim=8,
        transition_mode="basis",
        cross_factor=True,
        use_dual_space=True,
        use_same_node_context=True,
        preserve_source_channels=True,
        num_bases=4,
        basis_rank=8,
        router_hidden_dim=16,
        offdiag_init_scale=0.1,
        layer_scale_init=0.1,
        neighbor_aggregation="mean",
        multi_scale="concat",
        fusion="concat_mlp",
        use_exposure=False,
        log_transition_stats=True,
        log_basis_stats=True,
        log_grad_stats=False,
        edge_chunk_size=7,
    )
    t.update(overrides)
    return _CfgNode(t)


def _model_cfg(**transition_overrides) -> _CfgNode:
    return _CfgNode(
        model=_CfgNode(
            hidden_dim=HIDDEN,
            factor_dim=D,
            dropout=0.0,
            activation="gelu",
            norm="layernorm",
            lambda_common=0.02,
            lambda_orth=0.01,
            lambda_recon=0.3,
            orth_fallback_batch=16,
            full_graph_training=True,
            transition=_transition_cfg(**transition_overrides),
        ),
    )


def _info() -> dict:
    return {
        "input_dim": TEXT_DIM + VISUAL_DIM,
        "num_nodes": N,
        "num_classes": 4,
        "text_dim": TEXT_DIM,
        "visual_dim": VISUAL_DIM,
    }


def _graph(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Tiny random graph (~100 edges) with some isolated nodes."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(N, TEXT_DIM + VISUAL_DIM, generator=generator)
    src = torch.randint(0, N, (N * 4,), generator=generator)
    dst = torch.randint(0, N, (N * 4,), generator=generator)
    keep = src != dst
    edge_index = torch.stack([src[keep], dst[keep]], dim=0)
    return x, edge_index


def _layer(**overrides) -> OwnershipTransitionLayer:
    t = dict(
        factor_dim=D,
        relation_dim=16,
        factor_id_dim=8,
        context_dim=8,
        transition_mode="basis",
        cross_factor=True,
        use_dual_space=True,
        use_same_node_context=True,
        preserve_source_channels=True,
        num_bases=4,
        basis_rank=8,
        router_hidden_dim=16,
        offdiag_init_scale=0.1,
        layer_scale_init=0.1,
        edge_chunk_size=7,
        dropout=0.0,
        activation="gelu",
        norm="layernorm",
    )
    t.update(overrides)
    torch.manual_seed(0)
    return OwnershipTransitionLayer(**t)


def _build_model(**transition_overrides):
    from src.models.biaxis_r3 import Model

    torch.manual_seed(0)
    return Model(_model_cfg(**transition_overrides), _info())


# ---------------------------------------------------------------------------
# T1: shapes
# ---------------------------------------------------------------------------


def test_layer_shapes() -> None:
    layer = _layer()
    H = torch.randn(N, 3, D)
    _, edge_index = _graph(1)
    H_out, stats = layer(H, edge_index, N)
    assert H_out.shape == (N, 3, D)
    # transition + basis stats present (plan §18.1-§18.5)
    assert "diag_norm_c" in stats["transition"]
    assert "offdiag_norm_pv" in stats["transition"]
    assert "ch_c_pt" in stats["transition"]
    assert "cos_c_pt" in stats["transition"]
    assert "basis_entropy" in stats["basis"]
    assert torch.isfinite(H_out).all()


def test_model_shapes_and_out_dim() -> None:
    model = _build_model()
    x, edge_index = _graph(1)
    model.eval()
    z, _, _, _, _ = model(x, edge_index)
    assert z.shape == (N, HIDDEN)
    assert model.out_dim == HIDDEN


def test_no_edge_safety() -> None:
    layer = _layer()
    H = torch.randn(N, 3, D)
    empty = torch.empty(2, 0, dtype=torch.long)
    H_out, stats = layer(H, empty, N)
    assert torch.isfinite(H_out).all()
    assert H_out.shape == (N, 3, D)


# ---------------------------------------------------------------------------
# T2: diagonal-only degeneration
# ---------------------------------------------------------------------------


def test_cross_factor_false_builds_no_offdiag_modules() -> None:
    model = _build_model(cross_factor=False, transition_mode="diagonal")
    for key, _ in model.named_parameters():
        assert not any(
            token in key
            for token in ("cross_static", "target_decode", "router", "basis_down", "basis_up")
        ), f"off-diagonal module leaked into diagonal-only build: {key}"


def test_frozen_zero_offdiag_scale_contributes_exactly_nothing() -> None:
    """offdiag_init_scale=0.0 (test mode): the off-diagonal path must have
    EXACTLY zero contribution — perturbing its weights cannot change the
    output, and its parameters receive no gradient."""
    layer = _layer(transition_mode="static", offdiag_init_scale=0.0)
    H = torch.randn(N, 3, D)
    _, edge_index = _graph(2)
    layer.eval()
    H_out0, _ = layer(H, edge_index, N)
    with torch.no_grad():
        for p in layer.cross_static.parameters():
            p.add_(1.0)
        for p in layer.target_decode.parameters():
            p.add_(1.0)
    H_out1, _ = layer(H, edge_index, N)
    assert torch.equal(H_out0, H_out1)
    # gradient audit: the skipped path carries no gradient
    loss = H_out1.sum()
    loss.backward()
    for p in layer.cross_static.parameters():
        assert p.grad is None
    for p in layer.target_decode.parameters():
        assert p.grad is None
    # the diagonal path still learns
    assert all(p.grad is not None and float(p.grad.norm()) > 0 for p in layer.diag.parameters())


# ---------------------------------------------------------------------------
# T3: exact identity with frozen-zero layer scale (plan §16.7)
# ---------------------------------------------------------------------------


def test_layer_scale_zero_exact_identity() -> None:
    layer = _layer(layer_scale_init=0.0)
    H = torch.randn(N, 3, D)
    _, edge_index = _graph(3)
    H_out, stats = layer(H, edge_index, N)
    assert torch.equal(H_out, H)
    assert stats == {}


def test_model_layer_scale_zero_preserves_state_through_stack() -> None:
    """L layers with frozen-zero scale: H^(L) == H^(0) bitwise, so with
    multi_scale=last the final z must equal the P0-fusion of the raw
    factorizer output computed on the same factors."""
    model = _build_model(layer_scale_init=0.0, multi_scale="last")
    x, edge_index = _graph(4)
    model.eval()
    z, _, _, _, _ = model(x, edge_index)
    # reference: fusion of the raw factors (same model, bypass graph stack)
    x_t, x_v = model._split_modalities(x)
    factors = model.factorizer(x_t, x_v)
    z_ref = model.fusion(
        torch.cat([factors["c"], factors["p_t"], factors["p_v"]], dim=-1)
    )
    assert torch.equal(z, z_ref)


# ---------------------------------------------------------------------------
# T4: edge-order invariance of the mean aggregation
# ---------------------------------------------------------------------------


def test_edge_order_invariance() -> None:
    layer = _layer()
    H = torch.randn(N, 3, D)
    _, edge_index = _graph(5)
    perm = torch.randperm(edge_index.size(1))
    shuffled = edge_index[:, perm]
    layer.eval()
    H_out_a, _ = layer(H, edge_index, N)
    H_out_b, _ = layer(H, shuffled, N)
    # index_add accumulation order differs -> allclose (repo convention 1e-5)
    assert torch.allclose(H_out_a, H_out_b, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# T5: gradient audit — no zero-gradient module (plan §16.6)
# ---------------------------------------------------------------------------


def test_gradient_audit_all_modules_learn() -> None:
    model = _build_model()  # basis + dual space + context + preserve + concat
    x, edge_index = _graph(6)
    model.train()
    z, _, _, aux_loss, _ = model(x, edge_index)
    loss = (z ** 2).sum() + aux_loss
    loss.backward()
    groups = {
        "factorizer": model.factorizer,
        "diag": model.transition_layers[0].diag,
        "basis_down": model.transition_layers[0].basis_down,
        "basis_up": model.transition_layers[0].basis_up,
        "router": model.transition_layers[0].router,
        "target_decode": model.transition_layers[0].target_decode,
        "update": model.transition_layers[0].update,
        "ms_proj": model.ms_proj,
        "fusion": model.fusion,
    }
    for name, module in groups.items():
        norm = sum(
            float(p.grad.norm().item() ** 2)
            for p in module.parameters()
            if p.grad is not None
        ) ** 0.5
        assert norm > 0, f"zero gradient on {name}"
    # no trainable param left with grad None
    for key, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"param {key} has no gradient"


# ---------------------------------------------------------------------------
# T6: off-diagonal init ratio (plan §4.4: small but nonzero)
# ---------------------------------------------------------------------------


def test_offdiag_init_ratio_small_positive() -> None:
    layer = _layer()
    H = torch.randn(N, 3, D)
    _, edge_index = _graph(7)
    layer.eval()
    _, stats = layer(H, edge_index, N)
    ratio = sum(
        stats["transition"][f"offdiag_diag_ratio_{n}"] for n in ("c", "pt", "pv")
    ) / 3.0
    assert 0.0 < ratio < 1.0, f"offdiag/diag init ratio {ratio:.4f} outside (0, 1)"


# ---------------------------------------------------------------------------
# T7: forward / inference parity (plan §16.5)
# ---------------------------------------------------------------------------


def test_forward_inference_parity() -> None:
    model = _build_model()
    x, edge_index = _graph(8)
    model.eval()
    z, _, _, _, _ = model(x, edge_index)
    z_inf = model.inference(x, edge_index)
    assert z_inf.device.type == "cpu"
    assert torch.allclose(z, z_inf, atol=1e-6, rtol=1e-6)


def test_eval_forward_emits_no_aux_stats() -> None:
    model = _build_model()
    x, edge_index = _graph(9)
    model.eval()
    _, _, _, aux_loss, aux_info = model(x, edge_index)
    assert float(aux_loss.item()) == 0.0
    assert aux_info == {}


def test_training_aux_includes_r3_stats() -> None:
    model = _build_model()
    x, edge_index = _graph(10)
    model.train()
    _, _, _, _, aux_info = model(x, edge_index)
    r3_keys = [k for k in aux_info if k.startswith("r3_")]
    assert any(k.startswith("r3_l1_diag_norm") for k in r3_keys)
    assert any(k.startswith("r3_l1_basis_entropy") for k in r3_keys)
    assert any(k.startswith("r3_l2_ch_") for k in r3_keys)
    for k in r3_keys:
        assert torch.isfinite(torch.as_tensor(float(aux_info[k])))


# ---------------------------------------------------------------------------
# T8 (support): aux-info whitelist passthrough for r3_* keys
# ---------------------------------------------------------------------------


def test_aux_info_whitelist_passthrough() -> None:
    sums: dict[str, float] = {}
    counts: dict[str, float] = {}
    update_aux_info_stats(
        sums, counts, {"p0_common_sim": 0.5, "r3_l1_diag_norm_c": 0.25}, weight=2.0
    )
    assert sums["p0_common_sim"] == pytest.approx(1.0)
    assert sums["r3_l1_diag_norm_c"] == pytest.approx(0.5)
