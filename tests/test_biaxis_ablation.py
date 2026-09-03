"""Unit tests for the paper-facing ablation model (plan §22 Prompt 6)."""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from src.models.biaxis_ablation import Model as AblationModel
from src.models.biaxis_p3 import Model as P3Model

N = 17
TEXT_DIM = 13
VISUAL_DIM = 19
FACTOR_DIM = 128
HIDDEN = 256
MODES = ["full_reference", "no_factor_axis", "no_relation_axis",
         "no_adaptive_allocation", "shared_operator", "no_cell_correction"]


def _make_cfg(mode: str) -> object:
    return OmegaConf.create({
        "model": {
            "name": "biaxis_ablation",
            "ablation": {"mode": mode},
            "hidden_dim": HIDDEN,
            "factor_dim": FACTOR_DIM,
            "dropout": 0.2,
            "activation": "gelu",
            "norm": "layernorm",
            "lambda_common": 0.02,
            "lambda_orth": 0.01,
            "lambda_recon": 0.3,
            "full_graph_training": True,
            "p1": {
                "factor_aware": True, "num_relations": 4, "relation_dim": 32,
                "relation_temperature": 0.5, "selector_hidden_dim": 64,
                "selector_input_norm": None, "budget_hidden_dim": 64,
                "use_graph_budget": True, "budget_shared": False,
                "eps": 1.0e-8, "relation_balance_weight": 0.0,
                "alpha_entropy_weight": 0.0, "budget_reg_weight": 0.0,
                "edge_chunk_size": None,
            },
            "p2": {
                "mode": "null_softmax", "score_hidden_dim": 64, "epsilon": 0.2,
                "tau_base": 1.0, "sinkhorn_iters": 10, "null_prior": 0.5,
                "null_score_init": 0.0, "detach_capacity_prior": True,
                "detach_relation_confidence": True, "eps": 1.0e-8,
            },
            "p3": {
                "operator_mode": "full_interaction", "lowrank_rank": 16,
                "basis_num_bases": 8, "operator_reg_weight": 0.0,
                "interaction_reg_weight": 0.0,
            },
        }
    })


def _make_info() -> dict:
    return {
        "input_dim": TEXT_DIM + VISUAL_DIM, "num_nodes": N, "num_classes": 5,
        "text_dim": TEXT_DIM, "visual_dim": VISUAL_DIM,
    }


def _make_x(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(N, TEXT_DIM + VISUAL_DIM, generator=generator)


def _make_edge(seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, N, (2, 60), generator=generator)


def _make_model(mode: str) -> AblationModel:
    return AblationModel(_make_cfg(mode), _make_info())


def test_all_modes_forward_finite() -> None:
    for mode in MODES:
        model = _make_model(mode)
        model.eval()
        z, second, third, aux, _ = model(_make_x(), _make_edge())
        assert second is None and third is None
        assert z.shape == (N, HIDDEN), f"{mode} bad shape {z.shape}"
        assert torch.isfinite(z).all(), f"{mode} non-finite"
        assert aux.ndim == 0


def test_full_reference_equals_biaxis_final_same_weights() -> None:
    """full_reference constructs the exact biaxis_p3 full_interaction path:
    same weights -> bitwise-identical outputs."""
    final_cfg = _make_cfg("full_reference")
    final_cfg.model.name = "biaxis_final"
    del final_cfg.model["ablation"]
    final_model = P3Model(final_cfg, _make_info())
    ablation = _make_model("full_reference")
    ablation.load_state_dict(final_model.state_dict())
    x, edge_index = _make_x(), _make_edge()
    for model in (final_model, ablation):
        model.eval()
    z_final, _, _, _, _ = final_model(x, edge_index)
    z_abl, _, _, _, _ = ablation(x, edge_index)
    assert torch.equal(z_final, z_abl)


def test_passthrough_modes_match_p3_operator_modes() -> None:
    """shared_operator == P3 shared; no_cell_correction == P3 additive
    (same weights -> identical outputs)."""
    mapping = {"shared_operator": "shared", "no_cell_correction": "additive"}
    for abl_mode, p3_mode in mapping.items():
        p3_cfg = _make_cfg(abl_mode)
        p3_cfg.model.name = "biaxis_p3"
        del p3_cfg.model["ablation"]
        p3_cfg.model.p3.operator_mode = p3_mode
        p3_model = P3Model(p3_cfg, _make_info())
        ablation = _make_model(abl_mode)
        ablation.load_state_dict(p3_model.state_dict())
        x, edge_index = _make_x(), _make_edge()
        for model in (p3_model, ablation):
            model.eval()
        z_p3, _, _, _, _ = p3_model(x, edge_index)
        z_abl, _, _, _, _ = ablation(x, edge_index)
        assert torch.equal(z_p3, z_abl), f"{abl_mode} != P3 {p3_mode}"


def test_no_factor_axis_graph_side_factor_blind() -> None:
    model = _make_model("no_factor_axis")
    model.eval()
    edge_index = _make_edge()
    q = torch.randn(N, FACTOR_DIM)
    out = model._graph_update(q.unsqueeze(1), edge_index, N)
    gamma = out["gamma"]
    assert gamma.shape == (N, 1, 5)  # F=1, K+1=5
    assert torch.allclose(gamma.sum(dim=-1), torch.ones(N, 1), atol=1e-5)
    # operator is relation-only: T = W0 + B_k, F=1
    assert model.operator.mode == "relation"
    assert model.operator.num_factors == 1
    assert model.null_score.numel() == 1
    # aux objective still trains (factorizer kept)
    model.train()
    _, _, _, aux, aux_info = model(_make_x(), edge_index)
    assert aux.item() > 0
    for key in ("p0_common_loss", "p0_orth_loss", "p0_recon_loss"):
        assert key in aux_info


def test_no_relation_axis_strict_neighbor_context() -> None:
    model = _make_model("no_relation_axis")
    model.eval()
    edge_index = _make_edge()
    f_block = torch.randn(N, 3, FACTOR_DIM)
    out = model._graph_update(f_block, edge_index, N)
    r = out["r"]
    assert r.shape == (edge_index.size(1), 1)  # K=1 strict
    assert torch.equal(r, torch.ones_like(r))
    assert out["gamma"].shape == (N, 3, 2)  # Local + Graph
    assert torch.allclose(out["gamma"].sum(dim=-1), torch.ones(N, 3), atol=1e-5)
    assert model.operator.mode == "factor" and model.operator.num_relations == 1
    # M2 relation modules dropped (no dead params)
    assert not hasattr(model, "struct_signature_mlp")
    assert not hasattr(model, "relation_prototypes")


def test_no_adaptive_allocation_fixed_gamma() -> None:
    model = _make_model("no_adaptive_allocation")
    model.eval()
    edge_index = _make_edge()
    f_block = torch.randn(N, 3, FACTOR_DIM)
    out = model._graph_update(f_block, edge_index, N)
    gamma = out["gamma"]
    assert torch.equal(gamma[..., 0], torch.zeros(N, 3))  # Local mass = 0
    # Gamma_ifk = a_ik: factor-independent (identical across factors)
    a = out["availability"]  # [N, K]
    for f in range(3):
        assert torch.allclose(gamma[:, f, 1:], a, atol=1e-6)
    # scorer deleted: no learned allocation influence
    assert not hasattr(model, "transport_scorer")
    # full hierarchical operator kept
    assert model.operator.mode == "full_interaction"
    # gradient still reaches the relation prototypes through a_ik
    model.train()
    z, _, _, aux, _ = model(_make_x(), edge_index)
    (z.square().mean() + aux).backward()
    assert model.relation_prototypes.prototypes.grad is not None
    assert torch.isfinite(model.relation_prototypes.prototypes.grad).all()


def test_ablation_gradients_all_modes() -> None:
    for mode in MODES:
        model = _make_model(mode)
        model.train()
        z, _, _, aux, _ = model(_make_x(), _make_edge())
        (z.square().mean() + aux).backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert all(g is not None and torch.isfinite(g).all() for g in grads), f"{mode} bad grad"
        assert any(g.norm() > 1e-9 for g in grads), f"{mode} all-zero gradients"


def test_ablation_inference_equivalence() -> None:
    for mode in MODES:
        model = _make_model(mode)
        model.eval()
        x = _make_x()
        edge_index = _make_edge()
        z_fwd, _, _, _, _ = model(x, edge_index)
        z_inf = model.inference(x, edge_index, device=torch.device("cpu"))
        assert z_inf.shape == (N, HIDDEN)
        assert torch.allclose(z_fwd, z_inf, atol=1e-5), f"{mode} inference mismatch"
