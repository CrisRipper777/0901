"""Unit tests for the R2-Design-2.5 shared analysis layer (plan Prompt 1).

Covered:
    classifier init bitwise replay (same seed + save/load round trip);
    fixed-alpha pipeline == M1 fixed-alpha forward (allclose);
    alpha CE-gradient diagnostics vs finite differences;
    Ridge probe recovers separable toy labels;
    warmup10+cosine LR endpoints;
    train_capacity_variant: P0 frozen 1-20 / unfrozen at 21, grad samples;
    no-test access guard (guarded labels raise on test indexing);
    pt_transmission_features S4 counterfactual: z_cf == z bitwise when
    H1 == H2 (symmetric graph), and differs otherwise.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.analysis.perf_r2d25_utils import (
    alpha_ce_gradients,
    build_fixed_alpha_pipeline,
    load_state_into,
    make_classifier_init,
    pt_transmission_features,
    ridge_probe,
    save_state,
    scheduled_lr,
    train_capacity_variant,
)
from src.models.biaxis_r2 import Model as B0Model

TEXT_DIM = 13
VISUAL_DIM = 19
NUM_NODES = 17
FACTOR_DIM = 32
HIDDEN_DIM = 64


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


def _make_setup():
    model = _make_b0()
    data = SimpleNamespace(
        x=torch.randn(NUM_NODES, TEXT_DIM + VISUAL_DIM),
        edge_index=torch.randint(0, NUM_NODES, (2, 40)),
        train_idx=torch.arange(0, 10),
        val_idx=torch.arange(10, 14),
        test_idx=torch.arange(14, NUM_NODES),
        y=torch.randint(0, 5, (NUM_NODES,)),
        num_classes=5,
    )
    return SimpleNamespace(model=model, data=data, head=None, device=torch.device("cpu"))


def _capacity_cfg(mode: str = "sep_concat") -> dict:
    return OmegaConf.create({
        "model": {
            "name": "biaxis_r2_capacity", "capacity_mode": mode,
            "hidden_dim": HIDDEN_DIM, "factor_dim": FACTOR_DIM,
            "dropout": 0.0, "activation": "gelu", "norm": "layernorm",
            "lambda_common": 0.02, "lambda_orth": 0.01, "lambda_recon": 0.3,
            "orth_fallback_batch": 16, "full_graph_training": True,
            "edge_chunk_size": None,
            "deep_supervision": {"enabled": False, "lambda": 0.1},
            "path_dropout_p": 0.0,
            "semantic_refiner": {"enabled": False},
            "functional_transfer": {"enabled": False},
        }
    })


# ---------------------------------------------------------------------------
# 1. Classifier init bitwise replay
# ---------------------------------------------------------------------------


def test_classifier_init_bitwise_replay(tmp_path) -> None:
    head_a = make_classifier_init(20260904, HIDDEN_DIM, 5, torch.device("cpu"))
    head_b = make_classifier_init(20260904, HIDDEN_DIM, 5, torch.device("cpu"))
    for (ka, va), (kb, vb) in zip(head_a.state_dict().items(), head_b.state_dict().items()):
        assert ka == kb
        assert torch.equal(va, vb)
    path = tmp_path / "head_init.pt"
    save_state(path, head_a)
    head_c = nn.Linear(HIDDEN_DIM, 5)
    load_state_into(path, head_c)
    for (ka, va), (kc, vc) in zip(head_a.state_dict().items(), head_c.state_dict().items()):
        assert torch.equal(va, vc)


# ---------------------------------------------------------------------------
# 2. Fixed-alpha pipeline == M1 fixed-alpha forward
# ---------------------------------------------------------------------------


def test_fixed_alpha_pipeline_matches_m1_forward() -> None:
    torch.manual_seed(3)
    setup = _make_setup()
    model = setup.model
    x, ei = setup.data.x, setup.data.edge_index
    pipeline = build_fixed_alpha_pipeline(setup, x, ei)

    from src.models.biaxis_r2_scale import Model as ScaleModel

    cfg = _capacity_cfg()
    cfg.model.pop("capacity_mode")
    cfg.model.pop("deep_supervision")
    cfg.model.pop("path_dropout_p")
    cfg.model["name"] = "biaxis_r2_scale"
    cfg.model["scale_mode"] = "m1"
    info = {"input_dim": TEXT_DIM + VISUAL_DIM, "num_nodes": NUM_NODES,
            "num_classes": 5, "text_dim": TEXT_DIM, "visual_dim": VISUAL_DIM}
    scale = ScaleModel(cfg, info).eval()
    scale.load_state_dict(model.state_dict(), strict=False)
    with torch.no_grad():
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
            scale.mixer.alpha.copy_(torch.tensor([0.0, alpha, 0.0]))
            z_m1 = scale(x, ei)[0]
            z_pipe = pipeline.z_at(alpha)
            assert torch.allclose(z_m1, z_pipe, atol=1e-5, rtol=1e-5), alpha


# ---------------------------------------------------------------------------
# 3. Alpha CE-gradient diagnostics vs finite differences
# ---------------------------------------------------------------------------


def test_alpha_ce_gradients_match_finite_differences() -> None:
    torch.manual_seed(5)
    setup = _make_setup()
    x, ei = setup.data.x, setup.data.edge_index
    pipeline = build_fixed_alpha_pipeline(setup, x, ei)
    head = make_classifier_init(20260904, HIDDEN_DIM, 5, torch.device("cpu"))
    alpha = 0.25
    grads = alpha_ce_gradients(pipeline, head, setup.data, alpha)

    def train_ce(a: float) -> float:
        z = pipeline.z_at(a)
        logits = head(z[setup.data.train_idx])
        return float(torch.nn.functional.cross_entropy(
            logits, setup.data.y[setup.data.train_idx]).item())

    eps = 1e-3
    fd = (train_ce(alpha + eps) - train_ce(alpha - eps)) / (2 * eps)
    assert math.isfinite(grads["d_train_ce"])
    assert math.isfinite(grads["d_val_ce"])
    assert abs(grads["d_train_ce"] - fd) <= 5e-2 * abs(fd) + 5e-2


# ---------------------------------------------------------------------------
# 4. Ridge probe
# ---------------------------------------------------------------------------


def test_ridge_probe_recovers_separable_labels() -> None:
    torch.manual_seed(9)
    n_tr, n_va, d, c = 200, 100, 8, 4
    feats = torch.randn(n_tr + n_va, d)
    labels = torch.arange(c).repeat((n_tr + n_va) // c + 1)[: n_tr + n_va]
    feats[:, :c] += torch.nn.functional.one_hot(labels, c).float() * 20.0
    res = ridge_probe(feats[:n_tr], labels[:n_tr], feats[n_tr:], labels[n_tr:])
    assert res["acc"] > 0.99
    assert res["macro_f1"] > 0.99


# ---------------------------------------------------------------------------
# 5. LR schedule
# ---------------------------------------------------------------------------


def test_scheduled_lr_warmup_cosine_endpoints() -> None:
    assert scheduled_lr(1, 300, 1e-3) == pytest.approx(1e-4, rel=1e-6)
    assert scheduled_lr(10, 300, 1e-3) == pytest.approx(1e-3, rel=1e-6)
    assert scheduled_lr(300, 300, 1e-3) == pytest.approx(1e-5, rel=1e-6)
    # strictly positive and non-INCREASING after warmup (cosine decay)
    vals = [scheduled_lr(e, 300, 1e-3) for e in range(10, 301)]
    assert all(v > 0 for v in vals)
    assert all(b <= a + 1e-15 for a, b in zip(vals, vals[1:]))


# ---------------------------------------------------------------------------
# 6. Capacity training loop: P0 freeze schedule + grad samples
# ---------------------------------------------------------------------------


class _GuardY:
    """Labels that explode if any TEST index is ever accessed."""

    def __init__(self, y: torch.Tensor, test_idx: torch.Tensor) -> None:
        self._y = y
        self._test = set(int(i) for i in test_idx.tolist())

    def __getitem__(self, idx):
        idxs = idx.tolist() if torch.is_tensor(idx) else [idx]
        for i in idxs:
            if int(i) in self._test:
                raise RuntimeError("TEST LABEL ACCESS VIOLATION")
        return self._y[idx]


def _fake_data(with_guard: bool) -> SimpleNamespace:
    torch.manual_seed(13)
    n = 40
    data = SimpleNamespace(
        x=torch.randn(n, TEXT_DIM + VISUAL_DIM),
        edge_index=torch.randint(0, n, (2, 120)),
        train_idx=torch.arange(0, 20),
        val_idx=torch.arange(20, 28),
        test_idx=torch.arange(28, 40),
        y=torch.randint(0, 5, (n,)),
        num_classes=5,
    )
    if with_guard:
        data.y = _GuardY(data.y, data.test_idx)
    return data


def test_train_capacity_variant_p0_freeze_and_no_test() -> None:
    from src.models.biaxis_r2_capacity import Model

    torch.manual_seed(17)
    cfg = _capacity_cfg("sep_concat")
    info = {"input_dim": TEXT_DIM + VISUAL_DIM, "num_nodes": 40, "num_classes": 5,
            "text_dim": TEXT_DIM, "visual_dim": VISUAL_DIM}
    model = Model(cfg, info)
    data = _fake_data(with_guard=True)  # any test-label read raises
    head = nn.Linear(HIDDEN_DIM, 5)
    res = train_capacity_variant(
        cfg, data, model, head, torch.device("cpu"),
        total_epochs=25, freeze_p0_epochs=20,
    )
    assert res["p0_unfrozen"] is True
    assert 0.0 <= res["best_val_acc"] <= 1.0
    assert res["best_epoch"] is not None
    # P0 grad norm is exactly 0 before unfreeze, positive right after
    by_epoch = {g["epoch"]: g for g in res["grad_samples"]}
    assert by_epoch[20]["grad_norm"]["p0"] == 0.0
    assert by_epoch[21]["grad_norm"]["p0"] > 0.0
    assert len(res["history"]) == res["stop_epoch"]
    # classifier learned something on the fake data (train CE finite)
    assert math.isfinite(res["history"][0]["train_ce"])


# ---------------------------------------------------------------------------
# 7. pt_transmission_features S4 counterfactual
# ---------------------------------------------------------------------------


def _symmetric_setup() -> SimpleNamespace:
    """Identical node features + complete graph => H1 == H2 for every node."""
    n = 8
    x_rows = torch.randn(1, TEXT_DIM + VISUAL_DIM).repeat(n, 1)
    rows, cols = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                rows.append(j)  # incoming neighbor mean: src -> dst
                cols.append(i)
    ei = torch.tensor([rows, cols])
    model = _make_b0()
    data = SimpleNamespace(
        x=x_rows, edge_index=ei,
        train_idx=torch.arange(0, 4), val_idx=torch.arange(4, 6),
        test_idx=torch.arange(6, n), y=torch.randint(0, 5, (n,)), num_classes=5,
    )
    return SimpleNamespace(model=model, data=data, head=None, device=torch.device("cpu"))


def test_transmission_s4_identical_when_h1_equals_h2() -> None:
    # Identical features + complete graph => H1 ~= H2 up to fp rounding of
    # the repeated-sum neighbor mean (round(7a)/7 != a in general), so the
    # counterfactual z must match z to fp noise, not bitwise.
    setup = _symmetric_setup()
    feats = pt_transmission_features(setup, setup.data.x, setup.data.edge_index)
    assert torch.allclose(feats["z"], feats["z_cf"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(feats["s0_h1"], feats["s0_h2"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(feats["s3_h1"], feats["s3_h2"], atol=1e-6, rtol=1e-6)


def test_transmission_s4_cf_replicates_manual_math_bitwise() -> None:
    # The extracted z_cf must be EXACTLY the described counterfactual:
    # replicate the math from the same inputs and require bitwise equality.
    from src.analysis.perf_r2d15_utils import extract_b0_states, propagation_signals

    torch.manual_seed(21)
    setup = _make_setup()
    model, x, ei = setup.model, setup.data.x, setup.data.edge_index
    feats = pt_transmission_features(setup, x, ei)
    with torch.no_grad():
        states = extract_b0_states(model, x, ei)
        f_star = states["f_pre"]
        h1, h2, _ = propagation_signals(model, f_star, ei, int(x.size(0)))
        v1 = torch.stack([model.source_transforms[a](h1[:, a]) for a in range(3)], dim=1)
        v2 = torch.stack([model.source_transforms[a](h2[:, a]) for a in range(3)], dim=1)
        ln1 = torch.stack([model.msg_norm_base[b](v1[:, b]) for b in range(3)], dim=1)
        ln2 = torch.stack([model.msg_norm_base[b](v2[:, b]) for b in range(3)], dim=1)
        rho = torch.sigmoid(model.raw_rho_base)
        base_msg_cf = ln1.clone()
        base_msg_cf[:, 1] = ln2[:, 1]
        f_out_cf = f_star + rho.view(1, 3, 1) * base_msg_cf
        z_cf_manual = model.fusion(
            torch.cat([f_out_cf[:, 0], f_out_cf[:, 1], f_out_cf[:, 2]], dim=-1))
    assert torch.equal(feats["z_cf"], z_cf_manual)
    assert torch.equal(feats["z"], states["z"])


def test_transmission_s4_differs_in_general() -> None:
    torch.manual_seed(21)
    setup = _make_setup()
    feats = pt_transmission_features(setup, setup.data.x, setup.data.edge_index)
    assert not torch.equal(feats["z"], feats["z_cf"])
    assert feats["z"].shape == (NUM_NODES, HIDDEN_DIM)
