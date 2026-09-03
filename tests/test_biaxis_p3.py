"""Unit tests for the P3 factor-relation operator layer (plan §29).

D1 scope (Prompt 2): FullResidualFactorRelationOperator — zero-residual
equivalence, shared linear identity, mode isolation, pair cell routing,
gradients, shapes/no-NaN, memory discipline (no giant node-wise tensors).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.biaxis_p3_components import FullResidualFactorRelationOperator

N, F, K, D = 11, 3, 4, 32
EPS = 1e-8


def _rand_g(n=N, f=F, k=K, d=D, seed=1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n, f, k, d, generator=generator)


def _rand_gamma(n=N, f=F, k=K, seed=2) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    gamma = torch.rand(n, f, k, generator=generator)
    return gamma / gamma.sum(dim=-1, keepdim=True)


def _make_w0(d=D, seed=3) -> nn.Linear:
    generator = torch.Generator().manual_seed(seed)
    w0 = nn.Linear(d, d, bias=False)
    with torch.no_grad():
        w0.weight.copy_(torch.randn(d, d, generator=generator))
    return w0


def _make_op(mode: str) -> FullResidualFactorRelationOperator:
    return FullResidualFactorRelationOperator(F, K, D, mode=mode)


def _shared_path(g_perm, gamma_graph, w0) -> torch.Tensor:
    """The P2 shared-operator reference: W0(sum_k Gamma g)."""
    agg = (gamma_graph.unsqueeze(-1) * g_perm).sum(dim=2)
    return w0(agg.reshape(N * F, D)).reshape(N, F, D)


# ---------------------------------------------------------------------------
# Zero-residual equivalence (plan §29.1)
# ---------------------------------------------------------------------------


def test_zero_residual_all_modes_equal_shared_path() -> None:
    """All residuals are zero-initialized: every mode must reproduce the
    shared W0 path EXACTLY (zero contributions are exact 0.0)."""
    g_perm, gamma_graph, w0 = _rand_g(), _rand_gamma(), _make_w0()
    reference = _shared_path(g_perm, gamma_graph, w0)
    for mode in FullResidualFactorRelationOperator.MODES:
        op = _make_op(mode)
        out = op(g_perm, gamma_graph, w0)
        assert torch.equal(out, reference), f"{mode} deviates from shared path"


def test_zero_residual_after_random_w0() -> None:
    """Same equivalence with a different W0 seed (order-of-magnitude sanity)."""
    g_perm, gamma_graph, w0 = _rand_g(), _rand_gamma(), _make_w0(seed=7)
    reference = _shared_path(g_perm, gamma_graph, w0)
    for mode in FullResidualFactorRelationOperator.MODES:
        assert torch.equal(_make_op(mode)(g_perm, gamma_graph, w0), reference)


def test_nonzero_residual_changes_only_its_mode() -> None:
    """Setting a residual changes the output only for modes that use it."""
    g_perm, gamma_graph, w0 = _rand_g(), _rand_gamma(), _make_w0()
    reference = _shared_path(g_perm, gamma_graph, w0)

    op_factor = _make_op("factor")
    with torch.no_grad():
        op_factor.A[0] += 0.1 * torch.randn_like(op_factor.A[0])
    assert not torch.allclose(op_factor(g_perm, gamma_graph, w0), reference, atol=1e-5)

    op_relation = _make_op("relation")
    with torch.no_grad():
        op_relation.B[1] += 0.1 * torch.randn_like(op_relation.B[1])
    assert not torch.allclose(op_relation(g_perm, gamma_graph, w0), reference, atol=1e-5)

    op_pair = _make_op("full_interaction")
    with torch.no_grad():
        op_pair.C[2, 3] += 0.1 * torch.randn_like(op_pair.C[2, 3])
    assert not torch.allclose(op_pair(g_perm, gamma_graph, w0), reference, atol=1e-5)


# ---------------------------------------------------------------------------
# Shared linear identity (plan §29.2)
# ---------------------------------------------------------------------------


def test_shared_linear_identity() -> None:
    """W0(sum_k Gamma g) == sum_k Gamma W0(g_k) (float-equal up to op order)."""
    g_perm, gamma_graph, w0 = _rand_g(), _rand_gamma(), _make_w0()
    op = _make_op("shared")
    aggregated = op(g_perm, gamma_graph, w0)
    per_cell = torch.zeros_like(aggregated)
    for k in range(K):
        per_cell = per_cell + gamma_graph[:, :, k : k + 1] * w0(
            g_perm[:, :, k].reshape(N * F, D)
        ).reshape(N, F, D)
    assert torch.allclose(aggregated, per_cell, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Mode isolation (plan §29.3)
# ---------------------------------------------------------------------------


def test_mode_parameter_isolation() -> None:
    def names(mode: str) -> set:
        return {name for name, _ in _make_op(mode).named_parameters()}

    assert names("shared") == set()
    assert names("factor") == {"A"}
    assert names("relation") == {"B"}
    assert names("additive") == {"A", "B"}
    assert names("full_interaction") == {"A", "B", "C"}


def test_mode_param_counts_match_plan() -> None:
    """Plan §8: extra params OF=3d^2, OR=4d^2, OADD=7d^2, OFR=19d^2."""
    d2 = D * D
    assert _make_op("shared").extra_residual_params() == 0
    assert _make_op("factor").extra_residual_params() == 3 * d2
    assert _make_op("relation").extra_residual_params() == 4 * d2
    assert _make_op("additive").extra_residual_params() == 7 * d2
    assert _make_op("full_interaction").extra_residual_params() == 19 * d2


# ---------------------------------------------------------------------------
# Pair cell routing (plan §29.4)
# ---------------------------------------------------------------------------


def test_pair_cell_routing() -> None:
    """Only Gamma[f=1, k=2] nonzero -> only cell (1, 2) residual affects the
    output, and only for factor 1."""
    g_perm, w0 = _rand_g(), _make_w0()
    gamma_graph = torch.zeros(N, F, K)
    gamma_graph[:, 1, 2] = 1.0

    op = _make_op("full_interaction")
    with torch.no_grad():
        op.A[1] += 0.3 * torch.randn_like(op.A[1])  # factor main effect at f=1
        op.B[2] += 0.3 * torch.randn_like(op.B[2])  # relation main effect at k=2
        op.C[1, 2] += 0.5 * torch.randn_like(op.C[1, 2])
        op.C[0, 0] += 1.0 * torch.randn_like(op.C[0, 0])  # unused cell
        op.C[2, 3] += 1.0 * torch.randn_like(op.C[2, 3])  # unused cell

    out = op(g_perm, gamma_graph, w0)  # [N, F, D]

    # Manual: only cells (1, 2) receive mass; T = W0 + A_1 + B_2 + C_1,2.
    expected_f1 = w0(g_perm[:, 1, 2])
    expected_f1 = expected_f1 + g_perm[:, 1, 2] @ op.A[1].t()
    expected_f1 = expected_f1 + g_perm[:, 1, 2] @ op.B[2].t()
    expected_f1 = expected_f1 + g_perm[:, 1, 2] @ op.C[1, 2].t()
    assert torch.allclose(out[:, 1], expected_f1, atol=1e-5)

    # Factors 0 and 2 get zero graph mass -> their output is exactly 0.
    assert torch.equal(out[:, 0], torch.zeros(N, D))
    assert torch.equal(out[:, 2], torch.zeros(N, D))


# ---------------------------------------------------------------------------
# Gradients (plan §29.5)
# ---------------------------------------------------------------------------


def test_gradients_finite_nonzero_all_modes() -> None:
    g_perm = _rand_g().requires_grad_(True)
    gamma_graph = _rand_gamma()
    for mode in FullResidualFactorRelationOperator.MODES:
        w0 = _make_w0()
        op = _make_op(mode)
        out = op(g_perm, gamma_graph, w0)
        (out.square().sum()).backward()  # real nonconstant loss
        # W0 always receives gradient (it is used by every mode).
        assert w0.weight.grad is not None and torch.isfinite(w0.weight.grad).all()
        assert w0.weight.grad.norm() > 1e-9
        for p in op.parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), f"{mode} non-finite grad"
            assert p.grad.norm() > 1e-9, f"{mode} zero gradient on residual"


def test_gradients_zero_gamma_cell_gives_zero_residual_grad() -> None:
    """Cells with zero Gamma contribute nothing: their residual gets no
    gradient (pair routing is real in autograd, not just in values)."""
    g_perm = _rand_g().requires_grad_(True)
    w0 = _make_w0()
    op = _make_op("full_interaction")
    gamma_graph = torch.zeros(N, F, K)
    gamma_graph[:, 0, 0] = 1.0  # only cell (0, 0) has mass
    out = op(g_perm, gamma_graph, w0)
    out.square().sum().backward()
    assert op.C.grad[0, 0].norm() > 1e-9
    assert torch.equal(op.C.grad[1, 2], torch.zeros(D, D))
    assert torch.equal(op.B.grad[1], torch.zeros(D, D))  # relation 1 unused


# ---------------------------------------------------------------------------
# Shapes / no NaN / memory discipline (plan §29.10)
# ---------------------------------------------------------------------------


def test_forward_shapes_and_no_nan() -> None:
    g_perm, gamma_graph, w0 = _rand_g(), _rand_gamma(), _make_w0()
    for mode in FullResidualFactorRelationOperator.MODES:
        out = _make_op(mode)(g_perm, gamma_graph, w0)
        assert out.shape == (N, F, D)
        assert torch.isfinite(out).all()


def test_no_giant_tensor_on_large_batch() -> None:
    """A [N, F, K, d, d] node-wise operator tensor at N=4096 would be ~3 TB;
    the loops must run with only [N, d]-sized transients. (CPU memory smoke.)"""
    big_n = 4096
    generator = torch.Generator().manual_seed(9)
    g_perm = torch.randn(big_n, F, K, D, generator=generator)
    gamma_graph = torch.rand(big_n, F, K, generator=generator)
    gamma_graph = gamma_graph / gamma_graph.sum(dim=-1, keepdim=True)
    w0 = _make_w0()
    out = _make_op("full_interaction")(g_perm, gamma_graph, w0)
    assert out.shape == (big_n, F, D)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Diagnostics (plan §13)
# ---------------------------------------------------------------------------


def test_diagnostics_zero_residuals() -> None:
    g_perm, gamma_graph, w0 = _rand_g(), _rand_gamma(), _make_w0()
    for mode in FullResidualFactorRelationOperator.MODES:
        diag = _make_op(mode).compute_diagnostics(g_perm, gamma_graph, w0)
        if mode != "shared":
            for value in diag["residual_norms"]["factor"]:
                assert value == 0.0
            for value in diag["residual_norms"]["relation"]:
                assert value == 0.0
            for row in diag["residual_norms"]["pair"]:
                assert all(v == 0.0 for v in row)
        assert diag["pair_strength"] == 0.0
        assert diag["message_deviation_usage_weighted"] == 0.0


def test_diagnostics_operator_distance_identity_zero_residuals() -> None:
    """Zero residuals -> all T_fk identical -> distance 0, cosine 1."""
    g_perm, gamma_graph, w0 = _rand_g(), _rand_gamma(), _make_w0()
    diag = _make_op("full_interaction").compute_diagnostics(g_perm, gamma_graph, w0)
    for stats in diag["operator_distance"]["same_relation_across_factors"]:
        assert stats["norm_frob_dist"] < 1e-6
        assert abs(stats["flattened_cosine"] - 1.0) < 1e-5
    for stats in diag["operator_distance"]["same_factor_across_relations"]:
        assert stats["norm_frob_dist"] < 1e-6
        assert abs(stats["flattened_cosine"] - 1.0) < 1e-5


def test_diagnostics_json_safe_and_bounded() -> None:
    import json

    g_perm, gamma_graph, w0 = _rand_g(), _rand_gamma(), _make_w0()
    op = _make_op("full_interaction")
    with torch.no_grad():
        op.A += 0.2 * torch.randn_like(op.A)
        op.B += 0.2 * torch.randn_like(op.B)
        op.C += 0.2 * torch.randn_like(op.C)
    diag = op.compute_diagnostics(g_perm, gamma_graph, w0)
    json.dumps(diag)
    assert diag["w0_norm"] > 0
    assert diag["pair_strength"] >= 0
    assert diag["message_deviation_usage_weighted"] >= 0
    assert diag["extra_residual_params"] == 19 * D * D
    assert len(diag["usage"]) == F and all(len(row) == K for row in diag["usage"])
    for stats in diag["operator_distance"]["same_relation_across_factors"]:
        assert 0.0 <= stats["norm_frob_dist"] <= 1.0 + 1e-6
        assert -1.0 - 1e-6 <= stats["flattened_cosine"] <= 1.0 + 1e-6


def test_diagnostics_message_deviation_positive_nonzero_residuals() -> None:
    g_perm, gamma_graph, w0 = _rand_g(), _rand_gamma(), _make_w0()
    op = _make_op("additive")
    with torch.no_grad():
        op.A += 0.5 * torch.randn_like(op.A)
    diag = op.compute_diagnostics(g_perm, gamma_graph, w0)
    assert diag["message_deviation_usage_weighted"] > 0


# ---------------------------------------------------------------------------
# Regularization hooks (plan §7)
# ---------------------------------------------------------------------------


def test_regularization_hooks() -> None:
    op = _make_op("full_interaction")
    with torch.no_grad():
        op.A.fill_(0.5)
        op.B.fill_(0.25)
        op.C.fill_(0.1)
    expected_op = float(F) * D * D * 0.25 + float(K) * D * D * 0.0625
    expected_int = float(F * K) * D * D * 0.01
    assert abs(float(op.reg_operator().item()) - expected_op) / expected_op < 1e-5
    assert abs(float(op.reg_interaction().item()) - expected_int) / expected_int < 1e-5
    assert _make_op("shared").reg_operator().item() == 0.0
    assert _make_op("shared").reg_interaction().item() == 0.0
    assert _make_op("additive").reg_interaction().item() == 0.0


# ===========================================================================
# biaxis_p3 model (D2 scope, Prompt 3: plan §28/§29)
# ===========================================================================

from omegaconf import OmegaConf

from src.models.biaxis_p2 import Model as P2Model
from src.models.biaxis_p3 import Model as P3Model

P3_TEXT_DIM = 13
P3_VISUAL_DIM = 19
P3_NUM_NODES = 17
P3_FACTOR_DIM = 128
P3_HIDDEN_DIM = 256
P3_MODES = ["shared", "factor", "relation", "additive", "full_interaction"]


def _make_p3_cfg(operator_mode: str = "shared", **p2_overrides) -> object:
    model_cfg = {
        "name": "biaxis_p3",
        "hidden_dim": P3_HIDDEN_DIM,
        "factor_dim": P3_FACTOR_DIM,
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
            "mode": "null_softmax",
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
        "p3": {
            "operator_mode": operator_mode,
            "lowrank_rank": 16,
            "operator_reg_weight": 0.0,
            "interaction_reg_weight": 0.0,
        },
    }
    model_cfg["p2"].update(p2_overrides)
    return OmegaConf.create({"model": model_cfg})


def _make_p3_data_info(**overrides) -> dict:
    info = {
        "input_dim": P3_TEXT_DIM + P3_VISUAL_DIM,
        "num_nodes": P3_NUM_NODES,
        "num_classes": 5,
        "text_dim": P3_TEXT_DIM,
        "visual_dim": P3_VISUAL_DIM,
    }
    info.update(overrides)
    return info


def _make_p3_x(num_nodes: int = P3_NUM_NODES, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(num_nodes, P3_TEXT_DIM + P3_VISUAL_DIM, generator=generator)


def _make_p3_edge_index(num_edges: int = 60, num_nodes: int = P3_NUM_NODES, seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, num_nodes, (2, num_edges), generator=generator)


def _make_p3_model(operator_mode: str = "shared", **p2_overrides) -> P3Model:
    return P3Model(_make_p3_cfg(operator_mode, **p2_overrides), _make_p3_data_info())


def test_p3_five_modes_forward() -> None:
    for mode in P3_MODES:
        model = _make_p3_model(mode)
        model.eval()
        x = _make_p3_x()
        edge_index = _make_p3_edge_index()
        z, second, third, aux, _ = model(x, edge_index)
        assert second is None and third is None
        assert z.shape == (P3_NUM_NODES, P3_HIDDEN_DIM)
        assert torch.isfinite(z).all()
        assert aux.ndim == 0


def test_p3_shared_matches_p2_null_softmax() -> None:
    """O0 (shared) with the SAME weights must reproduce the P2 NullSoftmax
    model — the shared term keeps P2's op order (plan §6/audit §6).
    Tolerance (not bitwise): the relation aggregation uses GPU-atomic
    index_add in real training, which is ~1e-6 nondeterministic across
    calls; the operator equivalence itself is bitwise-tested at component
    level with controlled inputs."""
    p2_cfg = _make_p3_cfg("shared")
    p2_cfg.model.name = "biaxis_p2"
    del p2_cfg.model["p3"]
    p2_model = P2Model(p2_cfg, _make_p3_data_info())
    p3_model = _make_p3_model("shared")
    p3_model.load_state_dict(p2_model.state_dict())

    x = _make_p3_x()
    edge_index = _make_p3_edge_index()
    for model in (p2_model, p3_model):
        model.eval()
    z_p2, _, _, _, _ = p2_model(x, edge_index)
    z_p3, _, _, _, _ = p3_model(x, edge_index)
    assert torch.allclose(z_p2, z_p3, atol=1e-5, rtol=1e-5)


def test_p3_zero_init_equivalence_model_level() -> None:
    """All residuals zero-initialized: every mode == shared at model level
    (graph update included; gamma is identical across modes). Weights are
    copied from the shared model; the operator residuals add exact 0.0."""
    x = _make_p3_x()
    edge_index = _make_p3_edge_index()
    shared = _make_p3_model("shared")
    shared.eval()
    z_shared, _, _, _, _ = shared(x, edge_index)
    for mode in P3_MODES:
        if mode == "shared":
            continue
        model = _make_p3_model(mode)
        model.load_state_dict(shared.state_dict(), strict=False)
        model.eval()
        z, _, _, _, _ = model(x, edge_index)
        assert torch.allclose(z, z_shared, atol=1e-5, rtol=1e-5)


def test_p3_inference_forward_equivalence() -> None:
    model = _make_p3_model("full_interaction")
    model.eval()
    x = _make_p3_x()
    edge_index = _make_p3_edge_index()
    z_fwd, _, _, _, _ = model(x, edge_index)
    z_inf = model.inference(x, edge_index, device=torch.device("cpu"))
    assert z_inf.shape == (P3_NUM_NODES, P3_HIDDEN_DIM)
    assert torch.allclose(z_fwd, z_inf, atol=1e-6)


def test_p3_gradient_flow_to_operator_residuals() -> None:
    model = _make_p3_model("full_interaction")
    model.train()
    x = _make_p3_x()
    edge_index = _make_p3_edge_index()
    z, _, _, aux, _ = model(x, edge_index)
    (z.square().mean() + aux).backward()
    for name, p in model.operator.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"{name} missing/non-finite grad"
        assert p.grad.norm() > 1e-9, f"{name} zero gradient"
    assert model.graph_w0.weight.grad is not None
    assert torch.isfinite(model.graph_w0.weight.grad).all()


def test_p3_param_counts_match_plan() -> None:
    base = sum(p.numel() for p in _make_p3_model("shared").parameters())
    d2 = P3_FACTOR_DIM * P3_FACTOR_DIM
    assert sum(p.numel() for p in _make_p3_model("factor").parameters()) == base + 3 * d2
    assert sum(p.numel() for p in _make_p3_model("relation").parameters()) == base + 4 * d2
    assert sum(p.numel() for p in _make_p3_model("additive").parameters()) == base + 7 * d2
    assert sum(p.numel() for p in _make_p3_model("full_interaction").parameters()) == base + 19 * d2


def test_p3_rejects_non_null_softmax_mode() -> None:
    import pytest

    with pytest.raises(AssertionError):
        _make_p3_model("shared", mode="adaptive_uot")


def test_p3_composition_uot_compat_mode_forward() -> None:
    """§24 compatibility check path: composition_uot admitted (never for the
    main study), plan solver branch works with the cell operator."""
    model = _make_p3_model("full_interaction", mode="composition_uot")
    model.eval()
    x = _make_p3_x()
    edge_index = _make_p3_edge_index()
    z, _, _, _, _ = model(x, edge_index)
    assert z.shape == (P3_NUM_NODES, P3_HIDDEN_DIM)
    assert torch.isfinite(z).all()
    out = model._graph_update(torch.randn(P3_NUM_NODES, 3, P3_FACTOR_DIM), edge_index, P3_NUM_NODES)
    assert torch.allclose(out["gamma"].sum(dim=-1), torch.ones(P3_NUM_NODES, 3), atol=1e-5)
    # graph-column theta constant at the P2 composition_uot value.
    assert torch.allclose(out["theta"], torch.full((P3_NUM_NODES,), 1.0 / 1.2), atol=1e-6)


def test_p3_rejects_deterministic_mode() -> None:
    import pytest

    with pytest.raises(AssertionError):
        _make_p3_model("shared", deterministic=True)


def test_p3_default_config_fast_path_policy() -> None:
    """Plan §29.11: the shipped biaxis_p3.yaml must keep the fast path
    (p2.deterministic=false, p2.mode=null_softmax)."""
    from hydra import compose, initialize

    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="config", overrides=["dataset=Movies", "task=nc", "model=biaxis_p3"])
    assert cfg.model.p2.mode == "null_softmax"
    assert cfg.model.p2.deterministic is False
    assert cfg.model.p3.operator_mode == "shared"
    assert cfg.model.p3.operator_reg_weight == 0.0
    assert cfg.model.p3.interaction_reg_weight == 0.0


def test_p3_gamma_row_sum_preserved() -> None:
    model = _make_p3_model("full_interaction")
    model.eval()
    edge_index = _make_p3_edge_index()
    f_block = torch.randn(P3_NUM_NODES, 3, P3_FACTOR_DIM)
    out = model._graph_update(f_block, edge_index, P3_NUM_NODES)
    gamma = out["gamma"]
    assert gamma.shape == (P3_NUM_NODES, 3, 5)
    assert torch.allclose(gamma.sum(dim=-1), torch.ones(P3_NUM_NODES, 3), atol=1e-5)
    assert torch.allclose(out["beta"], 1.0 - gamma[..., 0], atol=1e-5)
    assert torch.allclose(out["alpha"], gamma[..., 1:] / (out["beta"].unsqueeze(-1) + 1e-8), atol=1e-5)
    # g_perm exposed for diagnostics (P3 addition to the P2 return dict).
    assert out["g_perm"].shape == (P3_NUM_NODES, 3, 4, P3_FACTOR_DIM)


def test_p3_diagnostics_structure_and_json_safe() -> None:
    import json

    model = _make_p3_model("full_interaction")
    model.eval()
    diag = model.compute_p3_diagnostics(_make_p3_x(), _make_p3_edge_index())
    assert "operator" in diag
    op = diag["operator"]
    assert op["mode"] == "full_interaction"
    assert op["extra_residual_params"] == 19 * P3_FACTOR_DIM * P3_FACTOR_DIM
    assert len(op["residual_norms"]["factor"]) == 3
    assert len(op["residual_norms"]["relation"]) == 4
    assert len(op["residual_norms"]["pair"]) == 3
    assert len(op["operator_distance"]["same_relation_across_factors"]) == 4
    assert len(op["operator_distance"]["same_factor_across_relations"]) == 3
    json.dumps(diag)


def test_p3_aux_loss_preserved() -> None:
    model = _make_p3_model("additive")
    model.train()
    x = _make_p3_x()
    _, _, _, aux, aux_info = model(x, _make_p3_edge_index())
    assert aux.ndim == 0 and torch.isfinite(aux)
    for key in ("p0_common_loss", "p0_orth_loss", "p0_recon_loss"):
        assert key in aux_info


def test_p3_operator_regularizer_hooks_in_forward() -> None:
    model = _make_p3_model("full_interaction")
    # Small fills keep the reg magnitude low so the float32 subtraction
    # (aux - aux0) is not dominated by ulp error.
    with torch.no_grad():
        model.operator.A.fill_(0.01)
        model.operator.B.fill_(0.005)
        model.operator.C.fill_(0.002)
    expected = float(model.operator.reg_operator().item()) + 2.0 * float(model.operator.reg_interaction().item())
    assert expected > 0
    model.train()
    x = _make_p3_x()
    edge_index = _make_p3_edge_index()
    model.p3_operator_reg_weight = 1.0
    model.p3_interaction_reg_weight = 2.0
    # Same RNG seed -> same dropout masks -> only the reg terms differ.
    torch.manual_seed(0)
    _, _, _, aux, _ = model(x, edge_index)
    model.p3_operator_reg_weight = 0.0
    model.p3_interaction_reg_weight = 0.0
    torch.manual_seed(0)
    _, _, _, aux0, _ = model(x, edge_index)
    assert abs((aux - aux0).item() - expected) < 1e-4


# ===========================================================================
# Low-rank operator (P3-B scope, Prompt 7: plan §14-§19/§29.6-§29.8)
# ===========================================================================

from src.models.biaxis_p3_components import LowRankFactorRelationOperator

LR_RANK = 8


def _make_lr_op(mode: str) -> LowRankFactorRelationOperator:
    return LowRankFactorRelationOperator(F, K, D, rank=LR_RANK, mode=mode)


def test_lr_zero_equivalence() -> None:
    """a=b=0 -> c=0 -> LR-ADD == LR-INT == shared W0 path (plan §29.6)."""
    g_perm, gamma_graph, w0 = _rand_g(), _rand_gamma(), _make_w0()
    reference = _shared_path(g_perm, gamma_graph, w0)
    for mode in LowRankFactorRelationOperator.MODES:
        op = _make_lr_op(mode)
        assert torch.equal(op(g_perm, gamma_graph, w0), reference)


def test_lr_param_matching() -> None:
    """LR-ADD and LR-INT share the exact same parameter set (plan §29.7)."""
    op_add = _make_lr_op("lowrank_add")
    op_int = _make_lr_op("lowrank_interaction")
    assert op_add.extra_residual_params() == op_int.extra_residual_params()
    assert op_add.extra_residual_params() == 2 * D * LR_RANK + (F + K) * LR_RANK
    assert {n for n, _ in op_add.named_parameters()} == {n for n, _ in op_int.named_parameters()}


def test_lr_formula() -> None:
    """c^ADD = a+b; c^INT = a+b+a*b (plan §29.8) — match a manual reference."""
    op_add = _make_lr_op("lowrank_add")
    op_int = _make_lr_op("lowrank_interaction")
    with torch.no_grad():
        op_add.a.copy_(torch.randn(F, LR_RANK))
        op_add.b.copy_(torch.randn(K, LR_RANK))
        op_int.a.copy_(op_add.a)
        op_int.b.copy_(op_add.b)
    c_add = op_add._cell_coefficients()
    c_int = op_int._cell_coefficients()
    manual_add = op_add.a.unsqueeze(1) + op_add.b.unsqueeze(0)
    manual_int = manual_add + op_int.a.unsqueeze(1) * op_int.b.unsqueeze(0)
    assert torch.equal(c_add, manual_add)
    assert torch.equal(c_int, manual_int)
    # forward equivalence with a manual cell-wise computation
    g_perm, gamma_graph, w0 = _rand_g(), _rand_gamma(), _make_w0()
    with torch.no_grad():
        op_add.U.copy_(torch.randn(D, LR_RANK))
        op_add.V.copy_(torch.randn(D, LR_RANK))
        op_int.U.copy_(op_add.U)
        op_int.V.copy_(op_add.V)
    for op, c in ((op_add, c_add), (op_int, c_int)):
        out = op(g_perm, gamma_graph, w0)
        manual = _shared_path(g_perm, gamma_graph, w0)
        for f in range(F):
            for k in range(K):
                latent = (g_perm[:, f, k] @ op.V) * c[f, k]  # [N, r]
                manual[:, f] = manual[:, f] + gamma_graph[:, f, k : k + 1] * (latent @ op.U.t())
        assert torch.allclose(out, manual, atol=1e-5)


def test_lr_gradients_with_active_coefficients() -> None:
    """Nonzero a/b: U, V, a, b all receive finite nonzero gradients."""
    op = _make_lr_op("lowrank_interaction")
    with torch.no_grad():
        op.a.fill_(0.1)
        op.b.fill_(0.1)
    g_perm = _rand_g().requires_grad_(True)
    gamma_graph = _rand_gamma()
    w0 = _make_w0()
    out = op(g_perm, gamma_graph, w0)
    out.square().sum().backward()
    for name in ("U", "V", "a", "b"):
        p = getattr(op, name)
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"{name} bad grad"
        assert p.grad.norm() > 1e-9, f"{name} zero grad"
    assert w0.weight.grad is not None and w0.weight.grad.norm() > 1e-9


def test_lr_zero_init_gradient_expectations() -> None:
    """Plan §18: with a=b=0, a/b DO receive gradient (through V^T g), while
    U/V's residual coefficient is exactly 0 so their gradient starts at 0 —
    expected behavior, documented, not 'fixed'."""
    op = _make_lr_op("lowrank_interaction")
    g_perm = _rand_g().requires_grad_(True)
    gamma_graph = _rand_gamma()
    out = op(g_perm, gamma_graph, _make_w0())
    out.square().sum().backward()
    assert op.a.grad is not None and op.a.grad.norm() > 1e-9
    assert op.b.grad is not None and op.b.grad.norm() > 1e-9
    assert torch.equal(op.U.grad, torch.zeros(D, LR_RANK))
    assert torch.equal(op.V.grad, torch.zeros(D, LR_RANK))


def test_lr_no_giant_tensor_on_large_batch() -> None:
    big_n = 4096
    generator = torch.Generator().manual_seed(9)
    g_perm = torch.randn(big_n, F, K, D, generator=generator)
    gamma_graph = torch.rand(big_n, F, K, generator=generator)
    gamma_graph = gamma_graph / gamma_graph.sum(dim=-1, keepdim=True)
    out = _make_lr_op("lowrank_interaction")(g_perm, gamma_graph, _make_w0())
    assert out.shape == (big_n, F, D)
    assert torch.isfinite(out).all()


def test_lr_diagnostics_json_safe() -> None:
    import json

    g_perm, gamma_graph, w0 = _rand_g(), _rand_gamma(), _make_w0()
    op = _make_lr_op("lowrank_interaction")
    with torch.no_grad():
        op.a.fill_(0.05)
        op.b.fill_(0.05)
    diag = op.compute_diagnostics(g_perm, gamma_graph, w0)
    json.dumps(diag)
    assert diag["rank"] == LR_RANK
    assert diag["extra_residual_params"] == 2 * D * LR_RANK + (F + K) * LR_RANK
    assert diag["pair_strength"] >= 0


def test_p3_model_lowrank_modes_forward() -> None:
    for mode in ("lowrank_add", "lowrank_interaction"):
        model = P3Model(_make_p3_cfg(mode), _make_p3_data_info())
        model.eval()
        z, _, _, _, _ = model(_make_p3_x(), _make_p3_edge_index())
        assert z.shape == (P3_NUM_NODES, P3_HIDDEN_DIM)
        assert torch.isfinite(z).all()


def test_p3_model_lowrank_param_matching() -> None:
    """Model-level param matching: LR-ADD == LR-INT exactly (plan §17)."""
    n_add = sum(p.numel() for p in P3Model(_make_p3_cfg("lowrank_add"), _make_p3_data_info()).parameters())
    n_int = sum(p.numel() for p in P3Model(_make_p3_cfg("lowrank_interaction"), _make_p3_data_info()).parameters())
    assert n_add == n_int


def test_p3_model_rejects_unknown_operator_mode() -> None:
    import pytest

    with pytest.raises(AssertionError):
        P3Model(_make_p3_cfg("lowrank_foobar"), _make_p3_data_info())


def test_lr_diagnostics_device_safe_on_gpu() -> None:
    """Regression guard: compute_diagnostics must not mix CPU/GPU tensors
    (2026-09-03 smoke found torch.tensor(r_c) on CPU vs usage on GPU)."""
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    g_perm = _rand_g().cuda()
    gamma_graph = _rand_gamma().cuda()
    w0 = _make_w0().cuda()
    op = _make_lr_op("lowrank_interaction").cuda()
    diag = op.compute_diagnostics(g_perm, gamma_graph, w0)
    assert diag["pair_strength"] >= 0
    assert diag["w0_norm"] > 0


def test_lr_interaction_diagnostic_zero_for_additive() -> None:
    """review §12: for the additive form the effective operators decompose
    as T_fk = W0 + A_f + B_k, whose double-centered interaction is EXACTLY
    zero. The low-rank ADD/INT operators only differ via a*b: with a,b
    nonzero the interaction diagnostic must be nonzero and nonnegative."""
    g_perm, gamma_graph, w0 = _rand_g(), _rand_gamma(), _make_w0()
    for mode in LowRankFactorRelationOperator.MODES:
        op = _make_lr_op(mode)
        diag = op.compute_diagnostics(g_perm, gamma_graph, w0)
        assert diag["interaction"]["usage_weighted_strength"] >= 0
        assert len(diag["interaction"]["norms"]) == F
    # a=b=0 -> T_fk identical -> interaction zero up to float-mean ulp noise
    # (the double centering divides by F/K then re-adds; 3M/3 != M exactly).
    op = _make_lr_op("lowrank_interaction")
    diag = op.compute_diagnostics(g_perm, gamma_graph, _make_w0())
    assert diag["interaction"]["usage_weighted_strength"] < 1e-6


# ===========================================================================
# Basis-decomposed cell operator (review §20)
# ===========================================================================

from src.models.biaxis_p3_components import BasisCellOperator

BASIS_B = 4


def _make_basis_op(num_bases: int = BASIS_B) -> BasisCellOperator:
    return BasisCellOperator(F, K, D, num_bases=num_bases)


def test_basis_zero_equivalence() -> None:
    """c=0 -> T_fk = W0 for every cell: output bitwise == shared path."""
    g_perm, gamma_graph, w0 = _rand_g(), _rand_gamma(), _make_w0()
    reference = _shared_path(g_perm, gamma_graph, w0)
    assert torch.equal(_make_basis_op()(g_perm, gamma_graph, w0), reference)


def test_basis_param_counts() -> None:
    """B*d^2 + F*K*B (review §20)."""
    assert _make_basis_op(4).extra_residual_params() == 4 * D * D + F * K * 4
    assert _make_basis_op(8).extra_residual_params() == 8 * D * D + F * K * 8
    assert _make_basis_op(16).extra_residual_params() == 16 * D * D + F * K * 16
    # capacity-curve values at d=128 (review §20): 65.6K / 131.2K / 262.3K
    assert BasisCellOperator(F, K, 128, num_bases=4).extra_residual_params() == 65584
    assert BasisCellOperator(F, K, 128, num_bases=8).extra_residual_params() == 131168
    assert BasisCellOperator(F, K, 128, num_bases=16).extra_residual_params() == 262336


def test_basis_formula() -> None:
    """T_fk = W0 + sum_b c_fkb * V_b, matched against a manual reference."""
    op = _make_basis_op()
    with torch.no_grad():
        op.c.copy_(torch.randn(F, K, BASIS_B))
        op.V.copy_(torch.randn(BASIS_B, D, D))
    g_perm, gamma_graph, w0 = _rand_g(), _rand_gamma(), _make_w0()
    out = op(g_perm, gamma_graph, w0)
    manual = _shared_path(g_perm, gamma_graph, w0)
    resid = torch.einsum("bde,fkb->fkde", op.V, op.c)
    for f in range(F):
        for k in range(K):
            manual[:, f] = manual[:, f] + gamma_graph[:, f, k : k + 1] * (g_perm[:, f, k] @ resid[f, k].t())
    assert torch.allclose(out, manual, atol=1e-5)


def test_basis_gradients_with_active_coefficients() -> None:
    op = _make_basis_op()
    with torch.no_grad():
        op.c.fill_(0.1)
    g_perm = _rand_g().requires_grad_(True)
    out = op(g_perm, _rand_gamma(), _make_w0())
    out.square().sum().backward()
    assert op.V.grad is not None and torch.isfinite(op.V.grad).all() and op.V.grad.norm() > 1e-9
    assert op.c.grad is not None and torch.isfinite(op.c.grad).all() and op.c.grad.norm() > 1e-9


def test_basis_zero_init_gradient_expectations() -> None:
    """Same plan-§18 dynamics as the low-rank operator: c receives gradient
    immediately (through V_b x), V_b starts with an exactly-zero residual
    coefficient."""
    op = _make_basis_op()
    g_perm = _rand_g().requires_grad_(True)
    out = op(g_perm, _rand_gamma(), _make_w0())
    out.square().sum().backward()
    assert op.c.grad is not None and op.c.grad.norm() > 1e-9
    assert torch.equal(op.V.grad, torch.zeros(BASIS_B, D, D))


def test_basis_no_giant_tensor_on_large_batch() -> None:
    big_n = 4096
    generator = torch.Generator().manual_seed(9)
    g_perm = torch.randn(big_n, F, K, D, generator=generator)
    gamma_graph = torch.rand(big_n, F, K, generator=generator)
    gamma_graph = gamma_graph / gamma_graph.sum(dim=-1, keepdim=True)
    out = _make_basis_op()(g_perm, gamma_graph, _make_w0())
    assert out.shape == (big_n, F, D)
    assert torch.isfinite(out).all()


def test_basis_diagnostics_json_safe() -> None:
    import json

    g_perm, gamma_graph, w0 = _rand_g(), _rand_gamma(), _make_w0()
    op = _make_basis_op()
    with torch.no_grad():
        op.c.fill_(0.05)
    diag = op.compute_diagnostics(g_perm, gamma_graph, w0)
    json.dumps(diag)
    assert diag["num_bases"] == BASIS_B
    assert diag["extra_residual_params"] == BASIS_B * D * D + F * K * BASIS_B
    assert diag["pair_strength"] >= 0
    assert diag["interaction"]["usage_weighted_strength"] >= 0


def test_p3_model_basis_modes_forward() -> None:
    for num_bases in (4, 8, 16):
        cfg = _make_p3_cfg("basis")
        cfg.model.p3.basis_num_bases = num_bases
        model = P3Model(cfg, _make_p3_data_info())
        model.eval()
        z, _, _, _, _ = model(_make_p3_x(), _make_p3_edge_index())
        assert z.shape == (P3_NUM_NODES, P3_HIDDEN_DIM)
        assert torch.isfinite(z).all()
        expected_extra = num_bases * P3_FACTOR_DIM * P3_FACTOR_DIM + 3 * 4 * num_bases
        assert model.operator.extra_residual_params() == expected_extra


def test_biaxis_final_config_frozen_structure() -> None:
    """review §18: model=biaxis_final must resolve to the frozen final
    operator (full_interaction + null_softmax + deterministic=false) and the
    factory import path must exist."""
    from hydra import compose, initialize

    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="config", overrides=["dataset=Movies", "task=nc", "model=biaxis_final"])
    assert cfg.model.p2.mode == "null_softmax"
    assert cfg.model.p2.deterministic is False
    assert cfg.model.p3.operator_mode == "full_interaction"
    assert cfg.model.p3.operator_reg_weight == 0.0
    assert cfg.model.p3.interaction_reg_weight == 0.0

    from src.models.factory import build_model
    from src.models.biaxis_final import Model as FinalModel

    model = build_model(cfg, _make_p3_data_info())
    assert isinstance(model, FinalModel)
    assert model.p3_operator_mode == "full_interaction"
    assert model.operator.mode == "full_interaction"
