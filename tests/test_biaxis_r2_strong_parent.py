"""Unit tests for the R2-Design-2.6 strong-parent model (plan Prompt 1, 12 items).

    frozen A0 full path reproduction; side-off exact z_base;
    H1 control never accesses H0/H2 (neighbor_mean call count);
    HOP/H1 parameter parity per readout; H2->H1 only changes the intended
    token; shuffle deterministic; factor-specific ablation correct;
    aux heads absent from inference output; no Test access;
    diagnostics finite; classifier init replay; A0 weights unchanged in
    frozen mode.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.analysis.perf_r2d26_utils import (
    READOUT_TYPES,
    assert_no_test_access,
    load_or_make_head_init,
    scheduled_lr,
    train_strong_parent,
)
from src.models.biaxis_r2_strong_parent import CAUSAL_OVERRIDES, Model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXT_DIM = 13
VISUAL_DIM = 19
NUM_NODES = 17
FACTOR_DIM = 32
HIDDEN_DIM = 64


def _make_parent() -> nn.Module:
    """A real tiny A0 parent (biaxis_perf_r1 baseline, tiny dims)."""
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


def _make_cfg(readout: str, source: str, deep_sup: bool = True,
              dsup_lambda: float = 0.1) -> OmegaConf:
    return OmegaConf.create({
        "model": {
            "name": "biaxis_r2_strong_parent", "readout_type": readout,
            "token_source": source, "hidden_dim": HIDDEN_DIM,
            "factor_dim": FACTOR_DIM, "dropout": 0.0,
            "activation": "gelu", "norm": "layernorm",
            "deep_supervision": {"enabled": deep_sup, "lambda": dsup_lambda},
        }
    })


def _make_model(readout: str, source: str, parent: nn.Module | None = None,
                deep_sup: bool = True) -> Model:
    info = {"input_dim": TEXT_DIM + VISUAL_DIM, "num_nodes": NUM_NODES,
            "num_classes": 5, "text_dim": TEXT_DIM, "visual_dim": VISUAL_DIM}
    return Model(_make_cfg(readout, source, deep_sup), info,
                 parent if parent is not None else _make_parent()).eval()


def _make_x(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(NUM_NODES, TEXT_DIM + VISUAL_DIM, generator=generator)


def _make_edges(seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, NUM_NODES, (2, 40), generator=generator)


def _finite(d) -> bool:
    if torch.is_tensor(d):
        return bool(torch.isfinite(d).all().item()) if d.is_floating_point() else True
    if isinstance(d, dict):
        return all(_finite(v) for v in d.values())
    if isinstance(d, (list, tuple)):
        return all(_finite(v) for v in d)
    if isinstance(d, (str, type(None))):
        return True
    return math.isfinite(float(d))


# ---------------------------------------------------------------------------
# 1/2. frozen A0 full path + side-off exact z_base
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("readout", READOUT_TYPES)
def test_side_off_reproduces_parent_bitwise(readout) -> None:
    torch.manual_seed(7)
    parent = _make_parent()
    m = _make_model(readout, "hop", parent)
    x, ei = _make_x(), _make_edges()
    with torch.no_grad():
        z_parent = parent(x, ei)[0]
        z_off, _, _, _, _ = m(x, ei, causal="side_off")
    assert torch.equal(z_off, z_parent)


# ---------------------------------------------------------------------------
# 3. H1 control never accesses H0/H2
# ---------------------------------------------------------------------------


def test_h1_control_never_computes_h2(monkeypatch) -> None:
    import src.models.biaxis_r2_strong_parent as mod

    real = mod.neighbor_mean
    calls = {"n": 0}

    def wrapped(edge_index, features, num_nodes, edge_chunk_size=None):
        calls["n"] += 1
        if calls["n"] > 1:
            raise AssertionError("H2 accessed by an H1-only control")
        return real(edge_index, features, num_nodes, edge_chunk_size=edge_chunk_size)

    monkeypatch.setattr(mod, "neighbor_mean", wrapped)
    m = _make_model("factor_hop_concat", "h1")
    x, ei = _make_x(), _make_edges()
    with torch.no_grad():
        m(x, ei)
    assert calls["n"] == 1


def test_hop_mode_computes_two_hop_contexts(monkeypatch) -> None:
    import src.models.biaxis_r2_strong_parent as mod

    real = mod.neighbor_mean
    calls = {"n": 0}

    def wrapped(edge_index, features, num_nodes, edge_chunk_size=None):
        calls["n"] += 1
        return real(edge_index, features, num_nodes, edge_chunk_size=edge_chunk_size)

    monkeypatch.setattr(mod, "neighbor_mean", wrapped)
    m = _make_model("factor_hop_concat", "hop")
    x, ei = _make_x(), _make_edges()
    with torch.no_grad():
        m(x, ei)
    assert calls["n"] == 2


# ---------------------------------------------------------------------------
# 4. HOP/H1 parameter parity per readout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("readout", ["no_compression_concat", "factor_hop_concat",
                                     "residual_side_fusion",
                                     "base_anchored_hier_attention"])
def test_hop_h1_parameter_parity(readout) -> None:
    m_hop = _make_model(readout, "hop")
    m_h1 = _make_model(readout, "h1")
    assert m_hop.side_parameter_count == m_h1.side_parameter_count
    assert m_hop.out_dim == m_h1.out_dim
    # parent params excluded from the side count
    assert m_hop.side_parameter_count == sum(p.numel() for p in m_hop.parameters())


def test_readout_only_param_matching_reported() -> None:
    m = _make_model("readout_only_control", "hop")
    match = m._readout_match
    assert abs(m.readout_mlp.parameter_count() - match["target_hier_side_params"]) \
        <= 0.05 * match["target_hier_side_params"]


def test_out_dim_per_readout() -> None:
    assert _make_model("no_compression_concat", "hop").out_dim == HIDDEN_DIM + 9 * FACTOR_DIM
    assert _make_model("factor_hop_concat", "hop").out_dim == HIDDEN_DIM + 3 * FACTOR_DIM
    assert _make_model("residual_side_fusion", "hop").out_dim == HIDDEN_DIM
    assert _make_model("base_anchored_hier_attention", "hop").out_dim == HIDDEN_DIM
    assert _make_model("readout_only_control", "hop").out_dim == HIDDEN_DIM


# ---------------------------------------------------------------------------
# 5. H2->H1 only changes the intended token
# ---------------------------------------------------------------------------


def test_h2_to_h1_only_changes_e2_token() -> None:
    torch.manual_seed(11)
    m = _make_model("no_compression_concat", "hop")
    x, ei = _make_x(), _make_edges()
    with torch.no_grad():
        _, tokens_full, _, _ = m.forward_with_experts(x, ei, causal="full")
        _, tokens_cf, _, _ = m.forward_with_experts(x, ei, causal="h2_to_h1")
    for key in ("e0", "e1"):
        assert torch.equal(tokens_full[key], tokens_cf[key]), key
    assert not torch.equal(tokens_full["e2"], tokens_cf["e2"])


# ---------------------------------------------------------------------------
# 6. shuffle deterministic
# ---------------------------------------------------------------------------


def test_h2_shuffle_deterministic() -> None:
    torch.manual_seed(13)
    m = _make_model("factor_hop_concat", "hop")
    x, ei = _make_x(), _make_edges()
    with torch.no_grad():
        z1, _, _, _, _ = m(x, ei, causal="h2_shuffle")
        z2, _, _, _, _ = m(x, ei, causal="h2_shuffle")
    assert torch.equal(z1, z2)
    with torch.no_grad():
        z_full, _, _, _, _ = m(x, ei, causal="full")
    assert not torch.equal(z1, z_full)


# ---------------------------------------------------------------------------
# 7. factor-specific ablation correct (only the named factor's H2 is replaced)
# ---------------------------------------------------------------------------


def test_factor_specific_h2_ablation() -> None:
    torch.manual_seed(17)
    m = _make_model("no_compression_concat", "hop")
    x, ei = _make_x(), _make_edges()
    with torch.no_grad():
        _, full, _, _ = m.forward_with_experts(x, ei, causal="full")
        _, cf, _, _ = m.forward_with_experts(x, ei, causal="pt_h2_off")
    # Pt column (1) differs; C/Pv columns (0, 2) untouched
    assert torch.equal(full["e2"][:, 0], cf["e2"][:, 0])
    assert torch.equal(full["e2"][:, 2], cf["e2"][:, 2])
    assert not torch.equal(full["e2"][:, 1], cf["e2"][:, 1])
    # e0/e1 never touched
    for key in ("e0", "e1"):
        assert torch.equal(full[key], cf[key])


# ---------------------------------------------------------------------------
# 8. aux heads absent from inference output
# ---------------------------------------------------------------------------


def test_aux_heads_do_not_affect_forward() -> None:
    torch.manual_seed(19)
    m = _make_model("factor_hop_concat", "hop", deep_sup=True)
    assert len(m.aux_heads) == 3
    x, ei = _make_x(), _make_edges()
    with torch.no_grad():
        z_before, _, _, _, _ = m(x, ei)
    with torch.no_grad():
        for key in m.aux_heads:
            for h in m.aux_heads[key]:
                h.weight.zero_()
                h.bias.zero_()
    with torch.no_grad():
        z_after, _, _, _, _ = m(x, ei)
    assert torch.equal(z_before, z_after)
    # disabled model has no aux heads
    m2 = _make_model("factor_hop_concat", "hop", deep_sup=False)
    assert len(m2.aux_heads) == 0


def test_deep_supervision_loss_finite() -> None:
    m = _make_model("factor_hop_concat", "hop", deep_sup=True)
    x, ei = _make_x(), _make_edges()
    train_idx = torch.arange(0, NUM_NODES, 2)
    y = torch.randint(0, 5, (train_idx.size(0),))
    with torch.enable_grad():
        _, tokens, _, _ = m.forward_with_experts(x, ei)
        loss = m.deep_supervision_loss(tokens, train_idx, y)
    assert math.isfinite(float(loss.item()))


# ---------------------------------------------------------------------------
# 9. no Test access (guarded labels) in the frozen training loop
# ---------------------------------------------------------------------------


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


def _fake_data(with_guard: bool) -> SimpleNamespace:
    torch.manual_seed(23)
    n = 40
    data = SimpleNamespace(
        x=torch.randn(n, TEXT_DIM + VISUAL_DIM),
        edge_index=torch.randint(0, n, (2, 120)),
        train_idx=torch.arange(0, 20), val_idx=torch.arange(20, 28),
        test_idx=torch.arange(28, 40), y=torch.randint(0, 5, (n,)),
        num_classes=5,
    )
    if with_guard:
        data.y = _GuardY(data.y, data.test_idx)
    return data


def test_no_test_access_in_training_loop() -> None:
    torch.manual_seed(29)
    parent = _make_parent()
    m = _make_model("factor_hop_concat", "hop", parent)
    head = nn.Linear(m.out_dim, 5)
    data = _fake_data(with_guard=True)
    res = train_strong_parent(
        data, m, head, torch.device("cpu"), total_epochs=2,
        deep_sup_lambda=0.1,
    )
    assert 0.0 <= res["best_val_acc"] <= 1.0
    assert math.isfinite(res["history"][0]["train_ce"])


# ---------------------------------------------------------------------------
# 10. all diagnostics finite
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("readout", READOUT_TYPES)
def test_diagnostics_finite(readout) -> None:
    torch.manual_seed(31)
    m = _make_model(readout, "hop")
    x, ei = _make_x(), _make_edges()
    with torch.no_grad():
        diag = m.compute_diagnostics(x, ei)
        sens = m.gradient_sensitivity(x, ei)
    assert _finite(diag), readout
    assert _finite(sens), readout
    # CPU is deterministic: bitwise holds; GPU atomics admit ~2e-6 noise
    # (R15-0), so the tolerance check is the cross-device criterion.
    assert diag["base_preservation"]["side_off_bitwise_equal_base"] is True
    assert diag["base_preservation"]["side_off_reproduces_base"] is True


# ---------------------------------------------------------------------------
# 11. classifier init replay
# ---------------------------------------------------------------------------


def test_classifier_init_replay(tmp_path) -> None:
    h1 = load_or_make_head_init(tmp_path / "head.pt", HIDDEN_DIM, 5, torch.device("cpu"))
    h2 = load_or_make_head_init(tmp_path / "head.pt", HIDDEN_DIM, 5, torch.device("cpu"))
    for (ka, va), (kb, vb) in zip(h1.state_dict().items(), h2.state_dict().items()):
        assert torch.equal(va, vb)


# ---------------------------------------------------------------------------
# 12. A0 weights unchanged in frozen mode
# ---------------------------------------------------------------------------


def test_a0_weights_unchanged_in_frozen_training() -> None:
    torch.manual_seed(37)
    parent = _make_parent()
    theta0 = {k: v.detach().clone() for k, v in parent.state_dict().items()}
    m = _make_model("factor_hop_concat", "hop", parent)
    head = nn.Linear(m.out_dim, 5)
    data = _fake_data(with_guard=False)
    train_strong_parent(data, m, head, torch.device("cpu"),
                        total_epochs=2, deep_sup_lambda=0.1)
    for k, v in parent.state_dict().items():
        assert torch.equal(v, theta0[k]), k


# ---------------------------------------------------------------------------
# misc: causal validation + LR schedule sanity
# ---------------------------------------------------------------------------


def test_h1_rejects_h2_causal() -> None:
    m = _make_model("factor_hop_concat", "h1")
    x, ei = _make_x(), _make_edges()
    with pytest.raises(AssertionError):
        m(x, ei, causal="h2_to_h1")


def test_scheduled_lr_endpoints() -> None:
    assert scheduled_lr(1, 300, 1e-3) == pytest.approx(1e-4, rel=1e-6)
    assert scheduled_lr(300, 300, 1e-3) == pytest.approx(1e-5, rel=1e-6)


def test_causal_names_registered() -> None:
    assert "h2_shuffle" in CAUSAL_OVERRIDES
    assert "side_off" in CAUSAL_OVERRIDES
    assert "pv_h2_off" in CAUSAL_OVERRIDES
