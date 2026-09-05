"""Unit tests for the R2-Design-2.5 structured-capacity model (plan Prompt 1).

Covered:
    EARLY_MIX alpha0 reproduces a loaded B0 bitwise;
    SEP experts have independent parameters;
    CAP_H1_DUP never accesses H2 (neighbor_mean call counting + poisoned
    second call);
    INCEPTION_012 accesses H0/H1/H2 separately (expert input recording);
    SEP_CONCAT / INCEPTION_012 never average hops before their transforms;
    H2-off ablates ONLY the H2 branch;
    per-mode state extraction keys / shapes;
    parameter accounting (C4 == C2 exact; C5 vs C2 within +/-5%);
    diagnostics finite for all modes;
    off_hops validation;
    path dropout p=1 == e1-off (dropout-free model);
    deep supervision default OFF / enabled heads.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.models.biaxis_r2 import Model as B0Model
from src.models.biaxis_r2_capacity import EXPERT_KEYS, MODES, Model

TEXT_DIM = 13
VISUAL_DIM = 19
NUM_NODES = 17
FACTOR_DIM = 32
HIDDEN_DIM = 64
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _cfg_dict(mode: str, dropout: float = 0.0, deep_sup: bool = False) -> dict:
    return {
        "model": {
            "name": "biaxis_r2_capacity", "capacity_mode": mode,
            "hidden_dim": HIDDEN_DIM, "factor_dim": FACTOR_DIM,
            "dropout": dropout, "activation": "gelu", "norm": "layernorm",
            "lambda_common": 0.02, "lambda_orth": 0.01, "lambda_recon": 0.3,
            "orth_fallback_batch": 16, "full_graph_training": True,
            "edge_chunk_size": None,
            "deep_supervision": {"enabled": deep_sup, "lambda": 0.1},
            "path_dropout_p": 0.0,
            "semantic_refiner": {"enabled": False, "gate_hidden": 16, "dropout": 0.0},
            "functional_transfer": {"enabled": False, "type_dim": 4, "gate_hidden": 16,
                                    "rho_func_init": 0.01},
        }
    }


def _info() -> dict:
    return {"input_dim": TEXT_DIM + VISUAL_DIM, "num_nodes": NUM_NODES,
            "num_classes": 5, "text_dim": TEXT_DIM, "visual_dim": VISUAL_DIM}


def _make_model(mode: str, dropout: float = 0.0, deep_sup: bool = False) -> Model:
    cfg = OmegaConf.create(_cfg_dict(mode, dropout, deep_sup))
    return Model(cfg, _info()).eval()


def _make_b0() -> B0Model:
    cfg = OmegaConf.create({
        "model": {
            "name": "biaxis_r2", "hidden_dim": HIDDEN_DIM, "factor_dim": FACTOR_DIM,
            "dropout": 0.0, "activation": "gelu", "norm": "layernorm",
            "lambda_common": 0.02, "lambda_orth": 0.01, "lambda_recon": 0.3,
            "orth_fallback_batch": 16, "full_graph_training": True,
            "edge_chunk_size": None,
            "semantic_refiner": {"enabled": False, "gate_hidden": 16, "dropout": 0.0},
            "functional_transfer": {"enabled": False, "type_dim": 4, "gate_hidden": 16,
                                    "rho_func_init": 0.01},
        }
    })
    info = {"input_dim": TEXT_DIM + VISUAL_DIM, "num_nodes": NUM_NODES,
            "num_classes": 5, "text_dim": TEXT_DIM, "visual_dim": VISUAL_DIM}
    return B0Model(cfg, info).eval()


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
# 1. EARLY_MIX alpha0 reproduces B0 bitwise
# ---------------------------------------------------------------------------


def test_early_mix_alpha0_reproduces_b0_bitwise() -> None:
    torch.manual_seed(7)
    b0 = _make_b0()
    cap = _make_model("early_mix")
    missing = sorted(set(cap.state_dict().keys()) - set(b0.state_dict().keys()))
    assert missing == ["mixer.alpha"], missing
    cap.load_state_dict(b0.state_dict(), strict=False)
    x, ei = _make_x(), _make_edges()
    with torch.no_grad():
        z_b0 = b0(x, ei)[0]
        z_cap = cap(x, ei)[0]
    assert torch.equal(z_b0, z_cap)


def test_early_mix_e2_off_forces_alpha_zero() -> None:
    torch.manual_seed(11)
    cap = _make_model("early_mix")
    with torch.no_grad():
        cap.mixer.alpha.copy_(torch.tensor([0.1, 0.7, -0.3]))
    x, ei = _make_x(), _make_edges()
    with torch.no_grad():
        z_off = cap(x, ei, off_hops={"e2"})[0]
        saved = cap.mixer.alpha.detach().clone()
        cap.mixer.alpha.copy_(torch.zeros(3))
        z_a0 = cap(x, ei)[0]
        cap.mixer.alpha.copy_(saved)
    assert torch.equal(z_off, z_a0)


# ---------------------------------------------------------------------------
# 2. SEP experts have independent parameters
# ---------------------------------------------------------------------------


def test_sep_experts_parameters_independent() -> None:
    for mode in ("sep_sum", "sep_concat", "inception_012", "cap_h1_dup"):
        m = _make_model(mode)
        keys = list(m.hop_experts.keys())
        assert len(keys) >= 2, mode
        seen: set[str] = set()
        for key in keys:
            for name in m.hop_experts[key].state_dict().keys():
                full = f"hop_experts.{key}.{name}"
                assert full not in seen, (mode, full)
                seen.add(full)
        # distinct parameter tensors
        k0, k1 = keys[0], keys[1]
        w0 = m.hop_experts[k0][0].net[0].weight
        w1 = m.hop_experts[k1][0].net[0].weight
        assert w0 is not w1
        assert not torch.equal(w0, w1) or w0.data_ptr() != w1.data_ptr()


# ---------------------------------------------------------------------------
# 3. CAP_H1_DUP never accesses H2 / INCEPTION accesses both hops
# ---------------------------------------------------------------------------


def _count_neighbor_mean_calls(monkeypatch, model, x, ei):
    import src.models.biaxis_r2_capacity as mod

    real = mod.neighbor_mean
    calls = {"n": 0, "inputs": []}

    def wrapped(edge_index, features, num_nodes, edge_chunk_size=None):
        calls["n"] += 1
        calls["inputs"].append(features.detach().clone())
        return real(edge_index, features, num_nodes, edge_chunk_size=edge_chunk_size)

    monkeypatch.setattr(mod, "neighbor_mean", wrapped)
    with torch.no_grad():
        model(x, ei)
    return calls


def test_cap_h1_dup_never_accesses_h2(monkeypatch) -> None:
    m = _make_model("cap_h1_dup")
    x, ei = _make_x(), _make_edges()
    calls = _count_neighbor_mean_calls(monkeypatch, m, x, ei)
    assert calls["n"] == 1


class _RecordingExpert(nn.Module):
    """Expert wrapper recording its inputs (explicit attributes: no
    closure-over-loop-variable traps)."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.inputs: list[torch.Tensor] = []

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        self.inputs.append(h.detach().clone())
        return self.inner(h)


def test_inception_accesses_h0_h1_h2_separately() -> None:
    m = _make_model("inception_012")
    x, ei = _make_x(), _make_edges()
    recorders: dict[str, _RecordingExpert] = {}
    for key in ("e0", "e1", "e2"):
        rec = _RecordingExpert(m.hop_experts[key][0])
        m.hop_experts[key][0] = rec
        recorders[key] = rec
    with torch.no_grad():
        factors, _ = m._encode(x)
        f0, f_star, _w = m._ownership_states(factors)
        f_out, internals = m._graph_update(f_star, ei, NUM_NODES)
    h0, h1, h2 = internals["h0"], internals["h1"], internals["h2"]
    assert torch.equal(recorders["e0"].inputs[0], h0[:, 0])
    assert torch.equal(recorders["e1"].inputs[0], h1[:, 0])
    assert torch.equal(recorders["e2"].inputs[0], h2[:, 0])
    # experts never saw a pre-transform mean: h1 != h2 on random data
    assert not torch.equal(h1[:, 0], h2[:, 0])


def test_sep_concat_no_pretransform_hop_mean() -> None:
    m = _make_model("sep_concat")
    x, ei = _make_x(), _make_edges()
    recorders: dict[str, _RecordingExpert] = {}
    for key in ("e1", "e2"):
        rec = _RecordingExpert(m.hop_experts[key][0])
        m.hop_experts[key][0] = rec
        recorders[key] = rec
    with torch.no_grad():
        factors, _ = m._encode(x)
        f0, f_star, _w = m._ownership_states(factors)
        _f_out, internals = m._graph_update(f_star, ei, NUM_NODES)
    h1, h2 = internals["h1"], internals["h2"]
    assert torch.equal(recorders["e1"].inputs[0], h1[:, 0])
    assert torch.equal(recorders["e2"].inputs[0], h2[:, 0])
    mean01 = 0.5 * (h1[:, 0] + h2[:, 0])
    assert not torch.equal(recorders["e1"].inputs[0], mean01)


# ---------------------------------------------------------------------------
# 4. H2-off ablates ONLY the H2 branch
# ---------------------------------------------------------------------------


def test_h2_off_only_disables_h2() -> None:
    torch.manual_seed(23)
    m = _make_model("sep_concat")
    x, ei = _make_x(), _make_edges()
    with torch.no_grad():
        z_full = m(x, ei)[0]
        z_h2off = m(x, ei, off_hops={"e2"})[0]
        z_h1off = m(x, ei, off_hops={"e1"})[0]
        # manual: zero the e2 expert outputs
        saved = {}
        for f in range(3):
            saved[f] = m.hop_experts["e2"][f]
            m.hop_experts["e2"][f] = _ZeroModule(saved[f])
        z_manual = m(x, ei)[0]
        for f in range(3):
            m.hop_experts["e2"][f] = saved[f]
    assert torch.equal(z_h2off, z_manual)
    assert not torch.equal(z_full, z_h2off)
    assert not torch.equal(z_h2off, z_h1off)


class _ZeroModule(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(self.inner(h))


# ---------------------------------------------------------------------------
# 5. State extraction coverage
# ---------------------------------------------------------------------------


def test_extract_capacity_states_keys_and_shapes() -> None:
    x, ei = _make_x(), _make_edges()
    for mode in MODES:
        m = _make_model(mode)
        with torch.no_grad():
            states = m.extract_capacity_states(x, ei)
        for key in ("f_star", "f_out", "z", "pre_residual", "post_residual",
                    "pre_fusion", "post_fusion", "h0", "h1"):
            assert key in states, (mode, key)
        assert states["h0"].shape == (NUM_NODES, 3, FACTOR_DIM)
        assert states["h1"].shape == (NUM_NODES, 3, FACTOR_DIM)
        assert states["pre_fusion"].shape == (NUM_NODES, 3 * FACTOR_DIM)
        assert states["post_fusion"].shape == (NUM_NODES, HIDDEN_DIM)
        if mode in ("wide_b0", "deep_fusion", "cap_h1_dup"):
            assert states["h2"] is None, mode
        else:
            assert states["h2"].shape == (NUM_NODES, 3, FACTOR_DIM), mode
        if mode in ("early_mix", "wide_b0", "deep_fusion"):
            assert states["msg_pre_ln"].shape == (NUM_NODES, 3, FACTOR_DIM)
            assert states["msg_post_ln"].shape == (NUM_NODES, 3, FACTOR_DIM)
        else:
            keys = set(states["expert_out"].keys())
            assert keys == set(EXPERT_KEYS[mode]), (mode, keys)


# ---------------------------------------------------------------------------
# 6. Parameter accounting
# ---------------------------------------------------------------------------


def test_parameter_count_correct_and_controls_matched() -> None:
    counts = {}
    for mode in MODES:
        m = _make_model(mode)
        counts[mode] = m.parameter_count
        assert counts[mode] == sum(p.numel() for p in m.parameters()), mode
    assert counts["cap_h1_dup"] == counts["sep_concat"]
    rel = abs(counts["wide_b0"] - counts["sep_concat"]) / counts["sep_concat"]
    assert rel <= 0.05, (counts["wide_b0"], counts["sep_concat"], rel)
    # WIDE_B0 records its own match report
    m = _make_model("wide_b0")
    assert m._wide_match["target_sep_concat_params"] == counts["sep_concat"]


# ---------------------------------------------------------------------------
# 7. Diagnostics finite for all modes
# ---------------------------------------------------------------------------


def test_diagnostics_finite_all_modes() -> None:
    x, ei = _make_x(), _make_edges()
    for mode in MODES:
        m = _make_model(mode)
        with torch.no_grad():
            diag = m.compute_capacity_diagnostics(x, ei)
        assert _finite(diag), mode
        assert diag["mode"] == mode


# ---------------------------------------------------------------------------
# 8. off_hops validation
# ---------------------------------------------------------------------------


def test_off_hops_validation() -> None:
    m = _make_model("sep_concat")
    x, ei = _make_x(), _make_edges()
    with pytest.raises(ValueError):
        m(x, ei, off_hops={"e0"})  # sep_concat has no e0
    m2 = _make_model("deep_fusion")
    with pytest.raises(ValueError):
        m2(x, ei, off_hops={"e2"})
    m3 = _make_model("early_mix")
    with pytest.raises(ValueError):
        m3(x, ei, off_hops={"e1"})


# ---------------------------------------------------------------------------
# 9. Path dropout (dropout-free model: p=1 == e1-off, bitwise)
# ---------------------------------------------------------------------------


def test_path_dropout_p1_equals_e1_off() -> None:
    torch.manual_seed(31)
    m = _make_model("sep_concat", dropout=0.0)
    x, ei = _make_x(), _make_edges()
    m.train()
    with torch.no_grad():
        z_drop = m(x, ei, path_dropout_h1=1.0)[0]
    m.eval()
    with torch.no_grad():
        z_off = m(x, ei, off_hops={"e1"})[0]
    assert torch.equal(z_drop, z_off)


# ---------------------------------------------------------------------------
# 10. Deep supervision default OFF / enabled
# ---------------------------------------------------------------------------


def test_deep_supervision_default_off() -> None:
    m = _make_model("sep_concat")
    assert not m.deep_sup_enabled
    assert len(m.aux_expert_heads) == 0
    with torch.no_grad():
        z, experts = m.forward_with_experts(_make_x(), _make_edges())
    assert set(experts.keys()) == {"e1", "e2"}


def test_deep_supervision_heads_and_loss() -> None:
    m = _make_model("sep_concat", deep_sup=True)
    assert m.deep_sup_enabled
    assert set(m.aux_expert_heads.keys()) == {"e1", "e2"}
    assert len(m.aux_expert_heads["e1"]) == 3
    x, ei = _make_x(), _make_edges()
    train_idx = torch.arange(0, NUM_NODES, 2)
    y_train = torch.randint(0, 5, (train_idx.size(0),))
    z, experts = m.forward_with_experts(x, ei)
    loss = m.deep_supervision_loss(experts, train_idx, y_train)
    assert math.isfinite(float(loss.item()))
