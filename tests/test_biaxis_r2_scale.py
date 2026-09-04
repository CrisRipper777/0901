"""Unit tests for the R2-Design-2.0 scale model (plan §36, 10 items).

M0 == B0 bitwise; M1 alpha=0 exact M0; M2 init numerically near M0 (max
diff reported); isolated-node correctness; sequential neighbor_mean H2;
factor order; forbidden-mechanism source scan; no test access; scale
diagnostics.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from src.models.biaxis_r2 import Model as B0Model
from src.models.biaxis_r2_scale import Model as ScaleModel, load_b0_checkpoint_into

TEXT_DIM = 13
VISUAL_DIM = 19
NUM_NODES = 17
FACTOR_DIM = 32
HIDDEN_DIM = 64
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _make_b0() -> B0Model:
    cfg = OmegaConf.create({
        "model": {
            "name": "biaxis_r2", "hidden_dim": HIDDEN_DIM, "factor_dim": FACTOR_DIM,
            "dropout": 0.2, "activation": "gelu", "norm": "layernorm",
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


def _make_scale(mode: str) -> ScaleModel:
    cfg = OmegaConf.create({
        "model": {
            "name": "biaxis_r2_scale", "scale_mode": mode,
            "hidden_dim": HIDDEN_DIM, "factor_dim": FACTOR_DIM,
            "dropout": 0.2, "activation": "gelu", "norm": "layernorm",
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
    return ScaleModel(cfg, info).eval()


def _make_x(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(NUM_NODES, TEXT_DIM + VISUAL_DIM, generator=generator)


def _make_edges(seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, NUM_NODES, (2, 40), generator=generator)


def _scale_with_b0_weights(mode: str, b0: B0Model) -> ScaleModel:
    model = _make_scale(mode)
    report = load_b0_checkpoint_into(model, None) if False else None  # placeholder
    state = b0.state_dict()
    model_keys = set(model.state_dict().keys())
    admissible = {"mixer.alpha", "mixer.hop_logits"}
    missing = sorted(model_keys - set(state.keys()))
    bad = [k for k in missing if k not in admissible]
    assert not bad and not (set(state.keys()) - model_keys)
    model.load_state_dict(state, strict=False)
    return model


# ---------------------------------------------------------------------------
# (1) M0 == B0 bitwise
# ---------------------------------------------------------------------------


def test_m0_matches_b0_bitwise() -> None:
    b0 = _make_b0()
    m0 = _make_scale("m0")
    m0.load_state_dict(b0.state_dict())  # strict: key sets identical
    x, ei = _make_x(), _make_edges()
    z_b0, _, _, _, _ = b0(x, ei)
    z_m0, _, _, _, _ = m0(x, ei)
    assert torch.equal(z_b0, z_m0)


# ---------------------------------------------------------------------------
# (2) M1 alpha=0 -> exact M0
# ---------------------------------------------------------------------------


def test_m1_zero_alpha_matches_m0_bitwise() -> None:
    b0 = _make_b0()
    m0 = _make_scale("m0")
    m0.load_state_dict(b0.state_dict())
    m1 = _scale_with_b0_weights("m1", b0)
    assert torch.equal(m1.mixer.alpha, torch.zeros(3))
    x, ei = _make_x(), _make_edges()
    z_m0, _, _, _, _ = m0(x, ei)
    z_m1, _, _, _, _ = m1(x, ei)
    assert torch.equal(z_m0, z_m1)


# ---------------------------------------------------------------------------
# (3) M2 init numerically near M0, max diff reported
# ---------------------------------------------------------------------------


def test_m2_init_near_m0_with_reported_diff() -> None:
    b0 = _make_b0()
    m0 = _make_scale("m0")
    m0.load_state_dict(b0.state_dict())
    m2 = _scale_with_b0_weights("m2", b0)
    x, ei = _make_x(), _make_edges()
    z_m0, _, _, _, _ = m0(x, ei)
    z_m2, _, _, _, _ = m2(x, ei)
    max_diff = float((z_m0 - z_m2).abs().max().item())
    # plan §36: "numerically near M0, report the max diff" — assert the
    # perturbation stays small RELATIVE to the embedding magnitude.
    z_scale = float(z_m0.norm(dim=-1).mean().item())
    assert max_diff < 0.05 * z_scale, f"max diff {max_diff} vs z scale {z_scale}"
    gamma = torch.softmax(m2.mixer.hop_logits, dim=-1)
    assert gamma[0, 1].item() > 0.999  # gamma1 ≈ 0.9993
    print(f"[info] M2-init max diff vs M0: {max_diff:.6f} (z scale {z_scale:.3f})")


# ---------------------------------------------------------------------------
# (4) isolated nodes correct
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["m0", "m1", "m2"])
def test_isolated_nodes(mode: str) -> None:
    b0 = _make_b0()
    model = _make_scale("m0") if mode == "m0" else _scale_with_b0_weights(mode, b0)
    x, _ = _make_x(), _make_edges()
    empty = torch.empty(2, 0, dtype=torch.long)
    x_t, x_v = model._split_modalities(x)
    factors = model.factorizer(x_t, x_v)
    f0, f_star, _w = model._ownership_states(factors)
    f_out, h1, base_msg, func_msg = model._graph_update(f_star, empty, NUM_NODES)
    if mode in ("m0", "m1"):
        # H1=H2=0 -> Hmix=0 -> V(0)=0 -> LN(0)=0 -> EXACT identity
        assert torch.equal(f_out, f_star)
        assert torch.equal(base_msg, torch.zeros_like(base_msg))
    else:
        # M2 isolated: Hmix = gamma0 * H0 (the intentional 0-hop ego term)
        gamma = torch.softmax(model.mixer.hop_logits, dim=-1)
        assert torch.equal(h1, torch.zeros_like(h1))
        assert torch.allclose(
            f_out - f_star,
            torch.sigmoid(model.raw_rho_base).view(1, 3, 1) * base_msg,
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# (5) H2 = sequential normalized neighbor_mean
# ---------------------------------------------------------------------------


def test_h2_is_sequential_neighbor_mean() -> None:
    from src.models.biaxis_p1_components import neighbor_mean

    b0 = _make_b0()
    m1 = _scale_with_b0_weights("m1", b0)
    x, ei = _make_x(), _make_edges()
    x_t, x_v = m1._split_modalities(x)
    factors = m1.factorizer(x_t, x_v)
    _f0, f_star, _w = m1._ownership_states(factors)
    _out, h1_model, _b, _f = m1._graph_update(f_star, ei, NUM_NODES)
    d = m1.factor_dim
    h2_ref = neighbor_mean(
        ei, h1_model.reshape(NUM_NODES, 3 * d), NUM_NODES,
        edge_chunk_size=m1.edge_chunk_size,
    ).reshape(NUM_NODES, 3, d)
    # the mixer consumed h2 internally; recompute through the same path and
    # verify equality of the full forward against a manual reconstruction
    h0_cat = f_star.reshape(NUM_NODES, 3 * d)
    h1_ref = neighbor_mean(ei, h0_cat, NUM_NODES).reshape(NUM_NODES, 3, d)
    assert torch.equal(h1_ref, h1_model)
    hmix_ref = h1_ref + m1.mixer.alpha.view(1, 3, 1) * (h2_ref - h1_ref)
    v = torch.stack([m1.source_transforms[a](hmix_ref[:, a]) for a in range(3)], dim=1)
    base = torch.stack([m1.msg_norm_base[b](v[:, b]) for b in range(3)], dim=1)
    f_ref = f_star + torch.sigmoid(m1.raw_rho_base).view(1, 3, 1) * base
    z_ref = m1.fusion(torch.cat([f_ref[:, 0], f_ref[:, 1], f_ref[:, 2]], dim=-1))
    z_model, _, _, _, _ = m1(x, ei)
    assert torch.equal(z_ref, z_model)


# ---------------------------------------------------------------------------
# (6) factor order [C, Pt, Pv]
# ---------------------------------------------------------------------------


def test_factor_order_cptv() -> None:
    b0 = _make_b0()
    m1 = _scale_with_b0_weights("m1", b0)
    x = _make_x()
    x_t, x_v = m1._split_modalities(x)
    factors = m1.factorizer(x_t, x_v)
    f0, f_star, _w = m1._ownership_states(factors)
    assert torch.equal(f_star[:, 0], factors["c"])
    assert torch.equal(f_star[:, 1], factors["p_t"])
    assert torch.equal(f_star[:, 2], factors["p_v"])
    assert m1.mixer.alpha.numel() == 3  # [C, Pt, Pv]


# ---------------------------------------------------------------------------
# (7)/(8) forbidden mechanisms absent
# ---------------------------------------------------------------------------


def test_no_forbidden_mechanisms_in_scale_sources() -> None:
    files = [
        PROJECT_ROOT / "src" / "models" / "biaxis_r2_scale.py",
        PROJECT_ROOT / "src" / "models" / "biaxis_r2_scale_components.py",
    ]
    for path in files:
        text = path.read_text(encoding="utf-8")
        # code lines only (docstrings/comments may name the mechanisms)
        code_lines = [
            line for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith(("#", '"""'))
        ]
        code = "\n".join(code_lines)
        for token in ("highpass", "high_pass", "relation_weighted_mean",
                      "null_augmented", "prototype"):
            assert token not in code, f"{path.name} code contains {token!r}"
    # the M2 softmax is the ONLY normalization, on the HOP axis only
    # (plan §6.2: M1 alphas never pass softmax/sigmoid/clamp)
    comp = (PROJECT_ROOT / "src" / "models" / "biaxis_r2_scale_components.py").read_text()
    # softmax exists ONLY in the M2 branches (forward + diagnostics), hop axis
    assert comp.count("torch.softmax") == 2
    m1 = _make_scale("m1")
    assert hasattr(m1.mixer, "alpha") and not hasattr(m1.mixer, "hop_logits")
    # the M1 branch is a plain interpolation (plan §6.2: no gate, no clamp)
    import inspect
    mixer_src = inspect.getsource(type(m1.mixer).forward)
    m1_branch = mixer_src.split("if self.mode == self.M1:")[1].split("return")[0]
    assert "softmax" not in m1_branch and "sigmoid" not in m1_branch and "clamp" not in m1_branch


# ---------------------------------------------------------------------------
# (9) no Test access
# ---------------------------------------------------------------------------


def test_no_test_logic_in_scale_sources() -> None:
    for rel in ("src/models/biaxis_r2_scale.py", "src/models/biaxis_r2_scale_components.py"):
        text = (PROJECT_ROOT / rel).read_text(encoding="utf-8")
        for token in ("test_idx", "test_accuracy", "data.y[", "test_mask", "data.test"):
            assert token not in text, f"{rel} contains {token!r}"


# ---------------------------------------------------------------------------
# (10) scale diagnostics
# ---------------------------------------------------------------------------


def test_scale_diagnostics() -> None:
    b0 = _make_b0()
    m1 = _scale_with_b0_weights("m1", b0)
    with torch.no_grad():
        m1.mixer.alpha.copy_(torch.tensor([0.1, 0.4, -0.2]))
    x, ei = _make_x(), _make_edges()
    diag = m1.compute_scale_diagnostics(x, ei)
    scale = diag["scale"]
    assert scale["mode"] == "m1"
    assert scale["alpha"][1] == pytest.approx(0.4, abs=1e-6)
    for name in ("C", "Pt", "Pv"):
        s = diag["smoothing"][name]
        assert 0.0 <= s["sim_h0_h1"] <= 1.0
        assert 0.0 <= s["sim_h0_h2"] <= 1.0
        assert s["rel_h2_h1_gap"] >= 0.0

    m2 = _scale_with_b0_weights("m2", b0)
    diag2 = m2.compute_scale_diagnostics(x, ei)
    scale2 = diag2["scale"]
    assert scale2["mode"] == "m2"
    assert len(scale2["gamma"]) == 3 and len(scale2["gamma"][0]) == 3
    assert abs(sum(scale2["gamma"][0]) - 1.0) < 1e-6
    assert scale2["effective_depth"][0] == pytest.approx(1.0, abs=1e-2)
