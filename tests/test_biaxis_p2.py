"""Unit tests for the P2 transport math layer (plan §31/§43, Prompt 2 scope).

Covers: FactorRelationScore, null scores, augmented scores, reference
capacity, relation confidence, NullSoftmax, semi-relaxed transport
(fixed + adaptive), numerical stability, and stop-gradient discipline.
"""

from __future__ import annotations

import torch

from src.models.biaxis_p2_components import (
    FactorRelationScore,
    build_augmented_scores,
    build_reference_capacity,
    compute_node_relation_confidence,
    null_augmented_softmax,
    semi_relaxed_transport,
)

N, F, K, D = 7, 3, 4, 16
EPS = 1e-8


def _rand_scores(n=N, f=F, k=K, seed=0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n, f, k, generator=generator)


def _rand_g(n=N, f=F, k=K, d=D, seed=1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n, f, k, d, generator=generator)


def _rand_availability(n=N, k=K, seed=2) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    a = torch.rand(n, k, generator=generator)
    return a / a.sum(dim=-1, keepdim=True)


def _rand_r(num_edges=30, k=K, seed=3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    r = torch.rand(num_edges, k, generator=generator)
    return r / r.sum(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# Score components
# ---------------------------------------------------------------------------


def test_score_shapes_and_shared_network() -> None:
    scorer = FactorRelationScore(D, hidden_dim=32)
    s = scorer(_rand_g()[:, :, 0], _rand_g())  # f: [N,F,d]
    assert s.shape == (N, F, K)
    # one shared network: exactly 4 param tensors (2 Linear w/b)
    assert len(list(scorer.parameters())) == 4


def test_augmented_scores_shape_and_null_column() -> None:
    s_rel = _rand_scores()
    z = torch.zeros(F)
    s_aug = build_augmented_scores(s_rel, z)
    assert s_aug.shape == (N, F, K + 1)
    assert torch.equal(s_aug[..., 1:], s_rel)
    assert torch.equal(s_aug[..., 0], torch.zeros(N, F))


def test_score_gradient_finite_nonzero() -> None:
    scorer = FactorRelationScore(D, hidden_dim=32)
    s = scorer(_rand_g()[:, :, 0], _rand_g())
    # NOT s.mean(): scorer output has no simplex constraint, but use a slice
    # anyway to be explicit; assert effective nonzero gradient (review §20).
    s[:, :, 0].mean().backward()
    for p in scorer.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
        assert p.grad.norm() > 1e-9


# ---------------------------------------------------------------------------
# Reference capacity & relation confidence
# ---------------------------------------------------------------------------


def test_reference_capacity_sums_to_F() -> None:
    a = _rand_availability()
    nu = build_reference_capacity(a, num_factors=F, null_prior=0.5)
    assert nu.shape == (N, K + 1)
    assert torch.allclose(nu.sum(dim=-1), torch.full((N,), float(F)), atol=1e-5)
    assert torch.allclose(nu[:, 0], torch.full((N,), F * 0.5), atol=1e-6)
    assert torch.allclose(nu[:, 1:], F * 0.5 * a, atol=1e-5)


def test_reference_capacity_isolated_nodes_pure_local() -> None:
    a = _rand_availability()
    deg = torch.arange(N, dtype=torch.float32)  # node 0 isolated
    nu = build_reference_capacity(a, num_factors=F, null_prior=0.5, degree=deg)
    assert torch.allclose(nu[0, 0], torch.tensor(float(F)), atol=1e-6)
    assert torch.equal(nu[0, 1:], torch.zeros(K))


def test_capacity_reference_stop_gradient() -> None:
    a = _rand_availability().requires_grad_(True)
    nu = build_reference_capacity(a, num_factors=F)
    assert not nu.requires_grad  # plan §10


def test_capacity_reference_detach_switch() -> None:
    a = _rand_availability().requires_grad_(True)
    nu_no_detach = build_reference_capacity(a, num_factors=F, detach=False)
    assert nu_no_detach.requires_grad  # review §17b: switch is real now


def test_relation_confidence_uniform_r_near_zero() -> None:
    r = torch.full((30, K), 1.0 / K)
    edge_index = torch.randint(0, N, (2, 30))
    q = compute_node_relation_confidence(r, edge_index, N)
    assert q.shape == (N,)
    assert (q >= 0).all() and (q <= 1).all()
    assert q.max() < 1e-4  # uniform r -> q ≈ 0


def test_relation_confidence_onehot_r_near_one() -> None:
    r = torch.zeros(30, K)
    r[:, 0] = 1.0
    edge_index = torch.randint(0, N, (2, 30))
    q = compute_node_relation_confidence(r, edge_index, N)
    assert q.max() > 0.99
    assert q.min() >= 0


def test_relation_confidence_isolated_nodes_zero() -> None:
    r = torch.zeros(1, K)
    r[:, 0] = 1.0
    edge_index = torch.tensor([[0], [1]])  # only node 1 receives; node 2 isolated
    q = compute_node_relation_confidence(r, edge_index, 3)
    assert q[2] == 0.0
    assert q[0] == 0.0  # node 0 has no incoming edges either


def test_relation_confidence_stop_gradient() -> None:
    r = _rand_r().requires_grad_(True)
    edge_index = torch.randint(0, N, (2, 30))
    q = compute_node_relation_confidence(r, edge_index, N)
    assert not q.requires_grad  # plan §17


def test_relation_confidence_detach_switch() -> None:
    r = _rand_r().requires_grad_(True)
    edge_index = torch.randint(0, N, (2, 30))
    q = compute_node_relation_confidence(r, edge_index, N, detach=False)
    assert q.requires_grad  # review §17b


def test_relation_uot_local_column_unconstrained() -> None:
    """Per-column theta (review §19): with Local theta=0 the Local column is
    never updated (v_0 stays 1), so the plan is EXACTLY invariant to the
    Local reference nu[:,0]; fixed_uot responds to nu[:,0]. (Mechanics test;
    composition_uot builds on this with the NS-mass graph reference.)"""
    s_aug = build_augmented_scores(_rand_scores(), torch.zeros(F))
    a = _rand_availability()
    nu1 = build_reference_capacity(a, num_factors=F, null_prior=0.5)
    nu2 = nu1.clone()
    nu2[:, 0] = 0.3  # different Local capacity target
    theta_col = torch.full((N, K + 1), 1.0 / (1.0 + 0.2))
    theta_col[:, 0] = 0.0
    g_rel_1 = semi_relaxed_transport(s_aug, nu1, 0.2, 1.0, 50, theta_col)
    g_rel_2 = semi_relaxed_transport(s_aug, nu2, 0.2, 1.0, 50, theta_col)
    g_fix_1 = semi_relaxed_transport(s_aug, nu1, 0.2, 1.0, 50)
    g_fix_2 = semi_relaxed_transport(s_aug, nu2, 0.2, 1.0, 50)
    assert torch.allclose(g_rel_1, g_rel_2, atol=1e-6)  # invariant to nu_0
    assert not torch.allclose(g_fix_1, g_fix_2, atol=1e-4)  # fixed mode responds
    assert (g_fix_1[..., 0].sum(dim=-1) > g_fix_2[..., 0].sum(dim=-1)).all()  # larger nu_0 -> more Local


# ---------------------------------------------------------------------------
# NullSoftmax
# ---------------------------------------------------------------------------


def test_null_softmax_matches_direct_softmax() -> None:
    s_rel = _rand_scores()
    z = torch.zeros(F)
    s_aug = build_augmented_scores(s_rel, z)
    gamma = null_augmented_softmax(s_aug, epsilon=0.2)
    expected = torch.softmax(s_aug / 0.2, dim=-1)
    assert torch.allclose(gamma, expected, atol=1e-6)
    assert torch.allclose(gamma.sum(dim=-1), torch.ones(N, F), atol=1e-5)  # row marginal


def test_null_score_monotonicity() -> None:
    s_rel = _rand_scores()
    z_high = torch.full((F,), 2.0)
    z_low = torch.full((F,), -2.0)
    gamma_high = null_augmented_softmax(build_augmented_scores(s_rel, z_high), epsilon=0.2)
    gamma_low = null_augmented_softmax(build_augmented_scores(s_rel, z_low), epsilon=0.2)
    assert (gamma_high[..., 0] > gamma_low[..., 0]).all()  # larger null score -> more null mass


def test_null_score_gradient() -> None:
    s_rel = _rand_scores()
    z = torch.zeros(F, requires_grad=True)
    gamma = null_augmented_softmax(build_augmented_scores(s_rel, z), epsilon=0.2)
    # NOT gamma.sum(): each row sums to 1, so the total is a constant with
    # zero gradient (review §17c). Use the null column slice instead.
    gamma[..., 0].mean().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert z.grad.norm() > 1e-9


# ---------------------------------------------------------------------------
# Semi-relaxed transport
# ---------------------------------------------------------------------------


def test_semi_relaxed_row_marginal() -> None:
    s_rel = _rand_scores()
    z = torch.zeros(F)
    s_aug = build_augmented_scores(s_rel, z)
    nu = build_reference_capacity(_rand_availability(), num_factors=F)
    gamma = semi_relaxed_transport(s_aug, nu, epsilon=0.2, tau_base=1.0, sinkhorn_iters=10)
    assert gamma.shape == (N, F, K + 1)
    assert (gamma >= 0).all()
    assert torch.allclose(gamma.sum(dim=-1), torch.ones(N, F), atol=1e-5)


def test_tau_zero_equals_null_softmax() -> None:
    s_rel = _rand_scores()
    z = torch.zeros(F)
    s_aug = build_augmented_scores(s_rel, z)
    nu = build_reference_capacity(_rand_availability(), num_factors=F)
    gamma_uot = semi_relaxed_transport(s_aug, nu, epsilon=0.2, tau_base=0.0, sinkhorn_iters=5)
    gamma_ns = null_augmented_softmax(s_aug, 0.2)
    assert torch.allclose(gamma_uot, gamma_ns, atol=1e-5)


def test_large_tau_moves_columns_closer_to_reference() -> None:
    s_rel = _rand_scores()
    z = torch.zeros(F)
    s_aug = build_augmented_scores(s_rel, z)
    a = _rand_availability()
    nu = build_reference_capacity(a, num_factors=F)
    gamma_weak = semi_relaxed_transport(s_aug, nu, epsilon=0.2, tau_base=1e-3, sinkhorn_iters=50)
    gamma_strong = semi_relaxed_transport(s_aug, nu, epsilon=0.2, tau_base=1e3, sinkhorn_iters=50)
    col_weak = gamma_weak.sum(dim=1)  # [N, K+1]
    col_strong = gamma_strong.sum(dim=1)
    dev_weak = (col_weak - nu).abs().sum(dim=-1).mean()
    dev_strong = (col_strong - nu).abs().sum(dim=-1).mean()
    assert dev_strong < dev_weak  # stronger constraint -> closer to nu


def test_adaptive_theta_shapes_and_per_node_variation() -> None:
    s_rel = _rand_scores()
    z = torch.zeros(F)
    s_aug = build_augmented_scores(s_rel, z)
    nu = build_reference_capacity(_rand_availability(), num_factors=F)
    q = torch.linspace(0.0, 1.0, N).unsqueeze(-1)  # per-node confidence
    theta = 1.0 * q / (1.0 * q + 0.2)  # tau_i / (tau_i + eps)
    gamma = semi_relaxed_transport(
        s_aug, nu, epsilon=0.2, tau_base=1.0, sinkhorn_iters=10, theta_override=theta
    )
    assert gamma.shape == (N, F, K + 1)
    assert torch.allclose(gamma.sum(dim=-1), torch.ones(N, F), atol=1e-5)
    # q=0 rows must degenerate to NullSoftmax exactly.
    gamma_ns = null_augmented_softmax(s_aug, 0.2)
    assert torch.allclose(gamma[0], gamma_ns[0], atol=1e-5)


# ---------------------------------------------------------------------------
# Numerical stability / no NaN
# ---------------------------------------------------------------------------


def test_numerical_stability_large_scores() -> None:
    s_rel = torch.tensor([[[-1000.0, 0.0, 1000.0, -500.0]]]).expand(N, F, K)
    z = torch.tensor([-500.0, 0.0, 500.0])
    s_aug = build_augmented_scores(s_rel, z)
    nu = build_reference_capacity(_rand_availability(), num_factors=F)
    gamma_ns = null_augmented_softmax(s_aug, 0.2)
    gamma_uot = semi_relaxed_transport(s_aug, nu, epsilon=0.2, tau_base=1.0, sinkhorn_iters=10)
    for gamma in (gamma_ns, gamma_uot):
        assert torch.isfinite(gamma).all()
        assert (gamma >= 0).all()
        assert torch.allclose(gamma.sum(dim=-1), torch.ones(N, F), atol=1e-5)


def test_no_nan_random_inputs() -> None:
    for seed in range(5):
        s_rel = _rand_scores(seed=seed)
        z = torch.zeros(F)
        s_aug = build_augmented_scores(s_rel, z)
        nu = build_reference_capacity(_rand_availability(seed=seed), num_factors=F)
        gamma = semi_relaxed_transport(s_aug, nu, epsilon=0.2, tau_base=1.0, sinkhorn_iters=10)
        assert torch.isfinite(gamma).all(), f"NaN at seed={seed}"
        assert (gamma >= 0).all()


def test_transport_degenerates_to_local_when_null_score_dominates() -> None:
    s_rel = _rand_scores()
    z = torch.full((F,), 10.0)  # null score dominates
    s_aug = build_augmented_scores(s_rel, z)
    gamma = null_augmented_softmax(s_aug, 0.2)
    assert (gamma[..., 0] > 0.99).all()


# ---------------------------------------------------------------------------
# biaxis_p2 model (Prompt 3 scope: plan §44)
# ---------------------------------------------------------------------------

import pytest
from omegaconf import OmegaConf

from src.models.biaxis_p2 import Model as P2Model

P2_TEXT_DIM = 13
P2_VISUAL_DIM = 19
P2_NUM_NODES = 17
P2_FACTOR_DIM = 128
P2_HIDDEN_DIM = 256


def _make_p2_cfg(mode: str = "adaptive_uot", **p2_overrides) -> object:
    model_cfg = {
        "name": "biaxis_p2",
        "hidden_dim": P2_HIDDEN_DIM,
        "factor_dim": P2_FACTOR_DIM,
        "dropout": 0.2,
        "activation": "gelu",
        "norm": "layernorm",
        "lambda_common": 0.02,
        "lambda_orth": 0.01,
        "lambda_recon": 0.3,
        "full_graph_training": True,
        "p1": {
            "factor_aware": True,
            "num_relations": 4,
            "relation_dim": 32,
            "relation_temperature": 0.5,
            "selector_hidden_dim": 64,
            "selector_input_norm": None,
            "budget_hidden_dim": 64,
            "use_graph_budget": True,
            "budget_shared": False,
            "eps": 1.0e-8,
            "relation_balance_weight": 0.0,
            "alpha_entropy_weight": 0.0,
            "budget_reg_weight": 0.0,
            "edge_chunk_size": None,
        },
        "p2": {
            "mode": mode,
            "score_hidden_dim": 64,
            "epsilon": 0.2,
            "tau_base": 1.0,
            "sinkhorn_iters": 10,
            "null_prior": 0.5,
            "null_score_init": 0.0,
            "detach_capacity_prior": True,
            "detach_relation_confidence": True,
            "eps": 1.0e-8,
        },
    }
    model_cfg["p2"].update(p2_overrides)
    return OmegaConf.create({"model": model_cfg})


def _make_p2_data_info(**overrides) -> dict:
    info = {
        "input_dim": P2_TEXT_DIM + P2_VISUAL_DIM,
        "num_nodes": P2_NUM_NODES,
        "num_classes": 5,
        "text_dim": P2_TEXT_DIM,
        "visual_dim": P2_VISUAL_DIM,
    }
    info.update(overrides)
    return info


def _make_p2_x(num_nodes: int = P2_NUM_NODES, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(num_nodes, P2_TEXT_DIM + P2_VISUAL_DIM, generator=generator)


def _make_p2_edge_index(num_edges: int = 60, num_nodes: int = P2_NUM_NODES, seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, num_nodes, (2, num_edges), generator=generator)


def _make_p2_model(mode: str = "adaptive_uot", **p2_overrides) -> P2Model:
    return P2Model(_make_p2_cfg(mode, **p2_overrides), _make_p2_data_info())


def test_p2_three_modes_forward() -> None:
    for mode in ("null_softmax", "fixed_uot", "adaptive_uot"):
        model = _make_p2_model(mode)
        model.eval()
        x = _make_p2_x()
        edge_index = _make_p2_edge_index()
        z, second, third, aux, _ = model(x, edge_index)
        assert second is None and third is None
        assert z.shape == (P2_NUM_NODES, P2_HIDDEN_DIM)
        assert torch.isfinite(z).all()
        assert aux.ndim == 0


def test_p2_gamma_shape_and_row_sum() -> None:
    model = _make_p2_model("adaptive_uot")
    model.eval()
    edge_index = _make_p2_edge_index()
    f_block = torch.randn(P2_NUM_NODES, 3, P2_FACTOR_DIM)
    out = model._graph_update(f_block, edge_index, P2_NUM_NODES)
    gamma = out["gamma"]
    assert gamma.shape == (P2_NUM_NODES, 3, 5)
    assert (gamma >= 0).all()
    assert torch.allclose(gamma.sum(dim=-1), torch.ones(P2_NUM_NODES, 3), atol=1e-5)


def test_p2_beta_alpha_derived_from_gamma() -> None:
    model = _make_p2_model("fixed_uot")
    model.eval()
    edge_index = _make_p2_edge_index()
    f_block = torch.randn(P2_NUM_NODES, 3, P2_FACTOR_DIM)
    out = model._graph_update(f_block, edge_index, P2_NUM_NODES)
    gamma = out["gamma"]
    beta = out["beta"]
    alpha = out["alpha"]
    assert torch.allclose(beta, 1.0 - gamma[..., 0], atol=1e-5)
    assert torch.allclose(alpha, gamma[..., 1:] / (beta.unsqueeze(-1) + 1e-8), atol=1e-5)
    assert torch.allclose(alpha.sum(dim=-1), torch.ones(P2_NUM_NODES, 3), atol=1e-4)
    assert torch.isfinite(alpha).all()


def test_p2_isolated_node_all_local() -> None:
    model = _make_p2_model("adaptive_uot")
    model.eval()
    edge_index = torch.randint(0, 16, (2, 60))  # node 16 isolated
    x = _make_p2_x()
    z, _, _, _, _ = model(x, edge_index)
    assert torch.isfinite(z).all()
    f_block = torch.randn(P2_NUM_NODES, 3, P2_FACTOR_DIM)
    out = model._graph_update(f_block, edge_index, P2_NUM_NODES)
    assert torch.allclose(out["gamma"][16, :, 0], torch.ones(3), atol=1e-6)
    assert torch.equal(out["gamma"][16, :, 1:], torch.zeros(3, 4))
    assert torch.equal(out["relation_confidence"][16], torch.zeros(()))


def test_p2_gate_modules_removed() -> None:
    model = _make_p2_model()
    assert not hasattr(model, "graph_budget")
    assert not hasattr(model, "factor_selector")
    assert not hasattr(model, "proj_q")
    assert not hasattr(model, "fusion_q")
    param_names = set(name for name, _ in model.named_parameters())
    assert not any("graph_budget" in name or "factor_selector" in name for name in param_names)


def test_p2_only_shared_w0_graph_operator() -> None:
    model = _make_p2_model()
    # The only transform applied to graph messages is the shared W0.
    assert isinstance(model.graph_w0, torch.nn.Linear)
    assert model.graph_w0.weight.shape == (P2_FACTOR_DIM, P2_FACTOR_DIM)
    # No per-factor / per-relation graph operators anywhere (plan §4/§44).
    # NOTE: M1's factor MLPs contain 128->128 Linears — those are the frozen
    # semantic factorizer internals, NOT graph operators; check by name.
    param_names = set(name for name, _ in model.named_parameters())
    graph_ops = {name for name in param_names if name.startswith("graph_w")}
    assert graph_ops == {"graph_w0.weight"}
    assert not any(
        "graph_w_c" in name or "graph_w_k" in name or "graph_w_fk" in name or "graph_w_pt" in name
        for name in param_names
    )
    # P2's own additions: exactly the shared scorer + per-factor null scalars.
    p2_added = {name.split(".")[0] for name in param_names if name.startswith(("transport_scorer", "null_score"))}
    assert p2_added == {"transport_scorer", "null_score"}


def test_p2_gradient_flow_to_all_components() -> None:
    model = _make_p2_model("adaptive_uot")
    model.train()
    x = _make_p2_x()
    edge_index = _make_p2_edge_index()
    z, _, _, aux, _ = model(x, edge_index)
    (z.square().mean() + aux).backward()
    components = [
        model.factorizer.text_projector,
        model.factorizer.visual_projector,
        model.factorizer.common_encoder,
        model.factorizer.private_text_encoder,
        model.factorizer.private_visual_encoder,
        model.recon_text_head,
        model.recon_visual_head,
        model.struct_signature_mlp,
        model.edge_token_mlp,
        model.relation_prototypes,
        model.transport_scorer,
        model.graph_w0,
        model.fusion,
    ]
    for component in components:
        params = list(component.parameters())
        assert params, f"no parameters in {component}"
        assert all(p.grad is not None for p in params), f"missing gradient in {component}"
        assert all(torch.isfinite(p.grad).all() for p in params), f"non-finite gradient in {component}"
    assert model.null_score.grad is not None and torch.isfinite(model.null_score.grad).all()
    assert model.null_score.grad.norm() > 1e-9


def test_p2_inference_forward_equivalence() -> None:
    model = _make_p2_model("fixed_uot")
    model.eval()
    x = _make_p2_x()
    edge_index = _make_p2_edge_index()
    z_fwd, _, _, _, _ = model(x, edge_index)
    z_inf = model.inference(x, edge_index, device=torch.device("cpu"))
    assert z_inf.shape == (P2_NUM_NODES, P2_HIDDEN_DIM)
    assert torch.allclose(z_fwd, z_inf, atol=1e-6)


def test_p2_aux_loss_preserved() -> None:
    model = _make_p2_model()
    model.train()
    x = _make_p2_x()
    _, _, _, aux, aux_info = model(x, _make_p2_edge_index())
    assert aux.ndim == 0 and torch.isfinite(aux)
    for key in ("p0_common_loss", "p0_orth_loss", "p0_recon_loss"):
        assert key in aux_info


def test_p2_rejects_bad_modes() -> None:
    with pytest.raises(AssertionError):
        _make_p2_model("balanced_ot")


def test_p2_diagnostics_structure_and_json_safe() -> None:
    import json

    model = _make_p2_model("adaptive_uot")
    model.eval()
    diag = model.compute_p2_diagnostics(_make_p2_x(), _make_p2_edge_index())
    assert set(diag["plan"].keys()) == {"C", "Pt", "Pv"}
    for name in ("C", "Pt", "Pv"):
        assert 0.0 <= diag["plan"][name]["null_mean"] <= 1.0
        assert 0.0 <= diag["plan"][name]["graph_mass_mean"] <= 1.0
        assert diag["plan_entropy"][name] >= 0
        assert diag["alpha_entropy"][name] >= 0
    assert diag["capacity_kl"] >= 0
    assert diag["capacity_l1"] >= 0
    assert 0.0 <= diag["relation_confidence"]["mean"] <= 1.0
    assert 0.0 <= diag["theta"]["mean"] <= 1.0
    assert 1.0 <= diag["relation"]["effective_num"] <= 4.0 + 1e-6
    assert len(diag["usage_matrix"]["values"]) == 3
    json.dumps(diag)


def test_p2_fixed_uot_theta_is_constant() -> None:
    model = _make_p2_model("fixed_uot")
    model.eval()
    edge_index = _make_p2_edge_index()
    f_block = torch.randn(P2_NUM_NODES, 3, P2_FACTOR_DIM)
    out = model._graph_update(f_block, edge_index, P2_NUM_NODES)
    expected = 1.0 / (1.0 + 0.2)
    assert torch.allclose(out["theta"], torch.full((P2_NUM_NODES,), expected), atol=1e-6)


def test_p2_null_softmax_theta_zero() -> None:
    model = _make_p2_model("null_softmax")
    model.eval()
    edge_index = _make_p2_edge_index()
    f_block = torch.randn(P2_NUM_NODES, 3, P2_FACTOR_DIM)
    out = model._graph_update(f_block, edge_index, P2_NUM_NODES)
    assert torch.equal(out["theta"], torch.zeros(P2_NUM_NODES))


def test_p2_composition_uot_mode_forward() -> None:
    model = _make_p2_model("composition_uot")
    model.eval()
    x = _make_p2_x()
    edge_index = _make_p2_edge_index()
    z, _, _, _, _ = model(x, edge_index)
    assert z.shape == (P2_NUM_NODES, P2_HIDDEN_DIM)
    assert torch.isfinite(z).all()
    out = model._graph_update(torch.randn(P2_NUM_NODES, 3, P2_FACTOR_DIM), edge_index, P2_NUM_NODES)
    assert torch.allclose(out["gamma"].sum(dim=-1), torch.ones(P2_NUM_NODES, 3), atol=1e-5)
    # graph-column theta constant; diagnostics expose the switch via theta.
    assert torch.allclose(
        out["theta"], torch.full((P2_NUM_NODES,), 1.0 / 1.2), atol=1e-6
    )


def test_p2_null_score_init_read_from_config() -> None:
    model = _make_p2_model("null_softmax", null_score_init=1.5)
    assert torch.equal(model.null_score.data, torch.full((3,), 1.5))  # review §17a


def test_p2_diagnostics_js_active() -> None:
    import json

    model = _make_p2_model("fixed_uot")
    model.eval()
    diag = model.compute_p2_diagnostics(_make_p2_x(), _make_p2_edge_index())
    assert set(diag["alpha_js_active"].keys()) == {"C_Pt", "C_Pv", "Pt_Pv"}
    for key, value in diag["alpha_js_active"].items():
        assert value is None or value >= 0
    assert set(diag["graph_active_frac"].keys()) == {"C", "Pt", "Pv"}
    json.dumps(diag)  # None values must survive JSON round-trip (null)


def test_p2_composition_uot_preserves_ns_total_graph_mass() -> None:
    """review §9/§10: composition_uot keeps the total graph mass at the
    unconstrained NullSoftmax value M_i^NS, while fixed_uot pulls it towards
    F*(1-pi0). Verified with zero scores (analytic regime)."""
    import sys
    sys.path.insert(0, ".")

    from src.models.biaxis_p2_components import (
        build_augmented_scores,
        build_reference_capacity,
        null_augmented_softmax,
        semi_relaxed_transport,
    )

    num_nodes, num_factors, k = 7, 3, 4
    a = torch.rand(num_nodes, k)
    a = a / a.sum(dim=-1, keepdim=True)
    s_aug = build_augmented_scores(torch.zeros(num_nodes, num_factors, k), torch.zeros(num_factors))
    gamma_ns = null_augmented_softmax(s_aug, 0.2)
    m_ns = gamma_ns[..., 1:].sum(dim=(1, 2)).detach()
    nu_rel = torch.cat([torch.zeros(num_nodes, 1), m_ns.unsqueeze(-1) * a], dim=-1)
    theta_col = torch.full((num_nodes, k + 1), 1.0 / 1.2)
    theta_col[:, 0] = 0.0
    g_comp = semi_relaxed_transport(s_aug, nu_rel, 0.2, 1.0, 200, theta_col)
    g_fixed = semi_relaxed_transport(s_aug, build_reference_capacity(a, num_factors=3), 0.2, 1.0, 200)
    graph_comp = g_comp[..., 1:].sum(dim=(1, 2))
    graph_fixed = g_fixed[..., 1:].sum(dim=(1, 2))
    # composition: total graph mass stays at the NS value (~2.4).
    assert torch.allclose(graph_comp.mean(), torch.tensor(2.4), atol=0.15)
    # fixed: pulled down towards F*(1-pi0) = 1.5.
    assert abs(graph_fixed.mean().item() - 1.5) < 0.25
    assert graph_comp.mean() > graph_fixed.mean()


def test_p2_rejects_old_relation_uot_mode() -> None:
    with pytest.raises(AssertionError):
        _make_p2_model("relation_uot")


# ---------------------------------------------------------------------------
# Deterministic aggregation (verification mode)
# ---------------------------------------------------------------------------


def test_deterministic_weighted_mean_matches_atomic_version() -> None:
    from src.models.biaxis_p1_components import relation_weighted_mean
    from src.models.biaxis_p2_components import deterministic_relation_weighted_mean

    edge_index = torch.randint(0, N, (2, 100))
    features = torch.randn(N, 8)
    r = torch.rand(100, K)
    r = r / r.sum(dim=-1, keepdim=True)
    g_atomic, m_atomic = relation_weighted_mean(edge_index, r, features, N)
    g_det, m_det = deterministic_relation_weighted_mean(edge_index, r, features, N, edge_chunk_size=33)
    assert torch.allclose(g_atomic, g_det, atol=1e-4)
    assert torch.allclose(m_atomic, m_det, atol=1e-5)


def test_deterministic_weighted_mean_bitwise_repeatable() -> None:
    from src.models.biaxis_p2_components import deterministic_relation_weighted_mean

    edge_index = torch.randint(0, N, (2, 100))
    features = torch.randn(N, 8)
    r = torch.rand(100, K)
    r = r / r.sum(dim=-1, keepdim=True)
    g1, m1 = deterministic_relation_weighted_mean(edge_index, r, features, N)
    g2, m2 = deterministic_relation_weighted_mean(edge_index, r, features, N)
    assert torch.equal(g1, g2)
    assert torch.equal(m1, m2)


def test_deterministic_confidence_matches_and_repeatable() -> None:
    from src.models.biaxis_p2_components import (
        compute_node_relation_confidence,
        deterministic_node_relation_confidence,
    )

    r = _rand_r()
    edge_index = torch.randint(0, N, (2, 30))
    q_atomic = compute_node_relation_confidence(r, edge_index, N)
    q_det_1 = deterministic_node_relation_confidence(r, edge_index, N)
    q_det_2 = deterministic_node_relation_confidence(r, edge_index, N)
    assert torch.allclose(q_atomic, q_det_1, atol=1e-5)
    assert torch.equal(q_det_1, q_det_2)


def test_p2_deterministic_flag_forward() -> None:
    model = _make_p2_model("composition_uot", deterministic=True)
    assert model.p2_deterministic
    model.eval()
    x = _make_p2_x()
    edge_index = _make_p2_edge_index()
    z, _, _, _, _ = model(x, edge_index)
    assert torch.isfinite(z).all()
    out = model._graph_update(torch.randn(P2_NUM_NODES, 3, P2_FACTOR_DIM), edge_index, P2_NUM_NODES)
    assert torch.allclose(out["gamma"].sum(dim=-1), torch.ones(P2_NUM_NODES, 3), atol=1e-5)
