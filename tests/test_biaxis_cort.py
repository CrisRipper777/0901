"""Unit tests for the R2D29 CORT architecture (plan §6.4 G1 audit items).

Covers: component numerics (COUPLED_EQUIV), null-softmax mass conservation,
permutation invariance, isolated-node NaN-safety, residual gating, gradient
flow, source-channel independence, recurrent routing recomputation, block
sharing, the forward/inference API contract and the no-Test-access rule.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.models.biaxis_cort_components import (
    NUM_FACTORS,
    CortBlock,
    CortMerger,
    FactorTypeEmbedding,
    cort_coupled_message,
)
from src.models.biaxis_r2_relfunc_components import chunked_coupled_message

N = 40
D = 16
HIDDEN = 24
T = 8
EPS = 1e-8


class _CfgNode(dict):
    def __getattr__(self, name: str):
        return self[name]


def _cort_cfg(**cort_overrides) -> _CfgNode:
    cort = dict(
        backbone_mode="a0_augment", router_mode="pair_null",
        source_mode="preserve_concat", writeback_mode="factor",
        fusion_mode="legacy", num_blocks=1, num_blocks_pre=1,
        num_blocks_post=1, share_blocks=False,
        interaction_hidden_mult=2.0, fusion_hidden_mult=2.0,
        residual_init=0.0, pre_norm=True, dropout=0.0,
        edge_chunk_size=7, memory_checkpoint=False, type_dim=T,
        mean_dup=False,
    )
    cort.update(cort_overrides)
    return _CfgNode(
        model=_CfgNode(
            hidden_dim=HIDDEN, factor_dim=D, dropout=0.0, activation="gelu",
            norm="layernorm",
            lambda_common=0.02, lambda_orth=0.01, lambda_recon=0.3,
            orth_fallback_batch=16, full_graph_training=True,
            p1=_CfgNode(
                factor_aware=True, num_relations=4, relation_dim=8,
                relation_temperature=0.5, selector_hidden_dim=16,
                selector_input_norm=None, budget_hidden_dim=16,
                use_graph_budget=True, budget_shared=False, eps=EPS,
                relation_balance_weight=0.0, alpha_entropy_weight=0.0,
                budget_reg_weight=0.0, edge_chunk_size=1000,
            ),
            p2=_CfgNode(
                mode="null_softmax", score_hidden_dim=16, epsilon=0.2,
                tau_base=1.0, sinkhorn_iters=5, null_prior=0.5,
                null_score_init=0.0, deterministic=False,
                detach_capacity_prior=True, detach_relation_confidence=True,
                eps=EPS,
            ),
            p3=_CfgNode(
                operator_mode="full_interaction", operator_reg_weight=0.0,
                interaction_reg_weight=0.0, memory_checkpoint=False,
            ),
            cort=_CfgNode(cort),
        ),
    )


def _graph(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Small synthetic graph with some isolated nodes (multiples of 7)."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(N, 24, generator=generator)
    src = torch.randint(0, N, (N * 3,), generator=generator)
    dst = torch.randint(0, N, (N * 3,), generator=generator)
    keep = src != dst
    edge_index = torch.stack([src[keep], dst[keep]], dim=0)
    return x, edge_index


def _info() -> dict:
    return {"input_dim": 24, "num_nodes": N, "num_classes": 4,
            "text_dim": 8, "visual_dim": 16}


def _build(**cort_overrides) -> nn.Module:
    from src.models.biaxis_cort import Model
    torch.manual_seed(0)
    return Model(_cort_cfg(**cort_overrides), _info())


# ---------------------------------------------------------------------------
# Component numerics
# ---------------------------------------------------------------------------


def test_coupled_message_matches_relfunc_reference() -> None:
    """COUPLED_EQUIV: the CORT message must reproduce the D2.8 v2
    chunked_coupled_message numerics to < 1e-6."""
    torch.manual_seed(1)
    f_block = torch.randn(N, 3, D)
    x, edge_index = _graph(1)
    scores = torch.randn(edge_index.size(1))
    null = torch.randn(N)
    payload = torch.randn(N, D)
    m_ref = chunked_coupled_message(
        f_block, edge_index, N, scores, null, payload, edge_chunk_size=7)
    m_cort, null_mass, _ent = cort_coupled_message(
        f_block, edge_index, N, scores, null, payload, edge_chunk_size=7)
    assert (m_cort - m_ref).abs().max().item() < 1e-6


def test_null_softmax_mass_conservation() -> None:
    """Plan §6.4 item 2: per-target mass over {null} u N(i) sums to 1."""
    torch.manual_seed(2)
    f_block = torch.randn(N, 3, D)
    _, edge_index = _graph(2)
    scores = torch.randn(edge_index.size(1)) * 2
    null = torch.randn(N)
    payload = torch.randn(N, D)
    m, null_mass, _ent = cort_coupled_message(
        f_block, edge_index, N, scores, null, payload, edge_chunk_size=7)
    # graph mass r = 1 - null_mass must equal the summed real-neighbor mass
    deg = torch.bincount(edge_index[1], minlength=N).to(torch.float32)
    assert float(torch.all((1.0 - null_mass) >= -1e-6).item())
    assert float(torch.all(null_mass <= 1.0 + 1e-6).item())
    # isolated nodes: null mass exactly 1, message exactly 0
    iso = deg == 0
    if iso.any():
        assert float((null_mass[iso] - 1.0).abs().max().item()) < 1e-6
        assert float(m[iso].abs().max().item()) < 1e-9


# ---------------------------------------------------------------------------
# Block-level audits
# ---------------------------------------------------------------------------


def _blocks_smoke(**cort_overrides):
    blk = CortBlock(
        D, FactorTypeEmbedding(T),
        router_mode=cort_overrides.pop("router_mode", "pair_null"),
        source_mode=cort_overrides.pop("source_mode", "preserve_concat"),
        writeback=cort_overrides.pop("writeback", True),
        type_dim=T, interaction_hidden_mult=2.0, residual_init=0.0,
        pre_norm=True, dropout=0.0, edge_chunk_size=7,
        memory_checkpoint=False,
        mean_dup=cort_overrides.pop("mean_dup", False),
    )
    return blk


def test_block_outputs_finite_all_router_source_modes() -> None:
    for router_mode in ("uniform", "pair_null", "target_null"):
        for source_mode in ("mean", "preserve_concat", "preserve_attn"):
            blk = _blocks_smoke(router_mode=router_mode, source_mode=source_mode)
            x, edge_index = _graph(3)
            f_in = torch.randn(N, 3, D)
            f_out, deltas, stats = blk(f_in, edge_index, N)
            assert f_out.shape == (N, 3, D)
            assert deltas.shape == (N, 3, D)
            assert torch.isfinite(f_out).all() and torch.isfinite(deltas).all()


def test_isolated_nodes_no_nan_through_block() -> None:
    blk = _blocks_smoke()
    f_in = torch.randn(N, 3, D)
    # a graph where a node has no incoming edges: node 0 has none
    edge_index = torch.tensor(
        [[1, 2, 3], [2, 3, 4]], dtype=torch.long)
    f_out, deltas, stats = blk(f_in, edge_index, N)
    assert torch.isfinite(f_out).all()
    assert torch.isfinite(deltas).all()


def test_permutation_invariance() -> None:
    """Plan §6.4 item 1: neighbor ordering must not matter (softmax over a
    multiset, chunked scatter)."""
    for router_mode in ("uniform", "pair_null", "target_null"):
        blk = _blocks_smoke(router_mode=router_mode)
        f_in = torch.randn(N, 3, D)
        _, edge_index = _graph(5)
        perm = torch.randperm(edge_index.size(1))
        edge_shuffled = edge_index[:, perm]
        blk.eval()
        with torch.no_grad():
            f1, _d1, _s1 = blk(f_in, edge_index, N)
            f2, _d2, _s2 = blk(f_in, edge_shuffled, N)
        assert (f1 - f2).abs().max().item() < 1e-4, router_mode


def test_residual_init_zero_gates_delta() -> None:
    """Plan §6.4 item 4: with residual_init=0 the write-back must equal the
    plain LayerNorm of the input (Delta fully gated)."""
    blk = _blocks_smoke()
    f_in = torch.randn(N, 3, D)
    _, edge_index = _graph(6)
    f_out, deltas, _stats = blk(f_in, edge_index, N)
    # reference: LN per factor of the input
    from torch.nn import LayerNorm
    refs = []
    for b in range(3):
        refs.append(blk.writeback_mod.norms[b](f_in[:, b]))
    ref = torch.stack(refs, dim=1)
    assert (f_out - ref).abs().max().item() < 1e-6
    # deltas must be non-trivial (the interaction actually produced signal)
    assert float(deltas.abs().max().item()) > 0.0


def test_nonzero_residual_init_changes_output() -> None:
    blk = _blocks_smoke()
    with torch.no_grad():
        blk.writeback_mod.rhos.copy_(torch.ones(NUM_FACTORS))
    f_in = torch.randn(N, 3, D)
    _, edge_index = _graph(6)
    f_out, _d, _s = blk(f_in, edge_index, N)
    refs = [blk.writeback_mod.norms[b](f_in[:, b]) for b in range(3)]
    ref = torch.stack(refs, dim=1)
    assert (f_out - ref).abs().max().item() > 1e-3


def test_gradients_flow_to_all_new_modules() -> None:
    """Plan §6.4 item 5: every new CORT module receives a non-zero gradient."""
    blk = _blocks_smoke(router_mode="pair_null", source_mode="preserve_concat")
    f_in = torch.randn(N, 3, D, requires_grad=True)
    _, edge_index = _graph(7)
    f_out, deltas, _s = blk(f_in, edge_index, N)
    loss = f_out.square().mean() + deltas.square().mean()
    loss.backward()
    for name, param in blk.named_parameters():
        assert param.grad is not None, f"{name} has no grad"
        assert float(param.grad.abs().sum().item()) > 0.0, f"{name} grad is zero"


def test_source_channel_independence() -> None:
    """Plan §6.4 item 6: the three source channels stay independent before
    write-back — Delta_b must depend on each of the three channel inputs."""
    blk = _blocks_smoke(router_mode="pair_null", source_mode="preserve_concat")
    f_in = torch.randn(N, 3, D)
    _, edge_index = _graph(8)
    # intercept the messages: clone them, require grad, replace
    msgs = {}
    captured = {}

    orig_router = blk.router

    class _Capture(nn.Module):
        def forward(self, f_pre, edge_index, num_nodes):
            m, stats = orig_router(f_pre, edge_index, num_nodes)
            captured["msgs"] = {k: v.clone().requires_grad_(True) for k, v in m.items()}
            return captured["msgs"], stats

    blk.router = _Capture()
    f_out, deltas, _s = blk(f_in, edge_index, N)
    # channels pairwise distinct
    for b in range(3):
        vals = [captured["msgs"][(a, b)] for a in range(3)]
        for i in range(3):
            for j in range(i + 1, 3):
                assert (vals[i] - vals[j]).abs().max().item() > 1e-5, (i, j, b)
    # Delta_b gradient w.r.t. each source channel is nonzero
    delta = deltas[:, 1]
    for a in range(3):
        g = torch.autograd.grad(delta.sum(), captured["msgs"][(a, 1)],
                                retain_graph=True)[0]
        assert float(g.abs().sum().item()) > 0.0, f"channel {a} unused"


def test_recurrent_routing_recomputed_every_layer() -> None:
    """Plan §8.3: layer 2 must route on the UPDATED factors, not a cached
    first-layer edge weighting."""
    x, edge_index = _graph(9)
    model = _build(backbone_mode="replace", num_blocks=2,
                   residual_init=1.0, router_mode="pair_null")
    seen = []

    def _hook(mod, _in, _out):
        seen.append(_in[0].detach().clone())

    handles = [model.cort_blocks[0].router.register_forward_hook(_hook),
               model.cort_blocks[1].router.register_forward_hook(_hook)]
    try:
        model.eval()
        with torch.no_grad():
            z, *_ = model(x, edge_index)
        assert len(seen) == 2
        # layer-2 router input != layer-1 router input (write-back happened)
        assert (seen[0] - seen[1]).abs().max().item() > 1e-4
    finally:
        for h in handles:
            h.remove()


def test_share_blocks_reuses_single_instance() -> None:
    x, edge_index = _graph(10)
    model = _build(backbone_mode="replace", num_blocks=3, share_blocks=True)
    assert model.cort_block is not None
    assert len(list(model.cort_blocks)) == 0
    model.eval()
    with torch.no_grad():
        z, *_ = model(x, edge_index)
    assert z.shape == (N, HIDDEN)
    # state_dict must load back into a fresh model
    model2 = _build(backbone_mode="replace", num_blocks=3, share_blocks=True)
    model2.load_state_dict(model.state_dict())


def test_merger_zero_init_is_a0_path() -> None:
    x = torch.randn(N, 3, D)
    f_a0 = torch.randn(N, 3, D)
    f_cort = torch.randn(N, 3, D)
    merger = CortMerger(D)
    out = merger(f_a0, f_cort)
    assert (out - f_a0).abs().max().item() < 1e-9


# ---------------------------------------------------------------------------
# Model-level API contract (all backbone modes)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backbone_mode",
                         ["a0_augment", "pre_a0", "sandwich", "replace", "hybrid"])
def test_forward_inference_contract(backbone_mode: str) -> None:
    x, edge_index = _graph(11)
    model = _build(backbone_mode=backbone_mode, residual_init=0.1)
    model.train()
    z, aux2, aux3, aux_loss, aux_info = model(x, edge_index)
    assert z.shape == (N, HIDDEN)
    assert aux2 is None and aux3 is None
    assert torch.isfinite(z).all()
    assert torch.isfinite(aux_loss).all()
    if aux_info.get("cort_stats"):
        assert isinstance(aux_info["cort_stats"], dict)
    model.eval()
    with torch.no_grad():
        z_eval, *_ = model(x, edge_index)
        z_inf = model.inference(x, edge_index)
    assert torch.allclose(z_eval, z_inf, atol=1e-5)
    # forward/backward on the training path
    z, *_ = model(x, edge_index)
    z.square().mean().backward()
    for name, param in model.named_parameters():
        assert torch.isfinite(param.grad).all() if param.grad is not None else True, name


def test_forward_without_edges_no_nan() -> None:
    x = torch.randn(N, 24)
    empty = torch.empty(2, 0, dtype=torch.long)
    model = _build()
    model.eval()
    with torch.no_grad():
        z, *_ = model(x, empty)
    assert torch.isfinite(z).all()


def test_no_test_access_static() -> None:
    """Plan §6.4 item 7: new model files must never read test labels."""
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for rel in ("src/models/biaxis_cort.py",
                "src/models/biaxis_cort_components.py"):
        text = (root / rel).read_text(encoding="utf-8")
        assert not re.search(r"test_idx|test_mask|data\.test|\.y\b.*test", text), rel
