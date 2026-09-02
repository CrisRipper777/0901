"""Unit tests for P1 Bi-Axis: topology-only structural relation decomposition
(M2) and relation-weighted sparse aggregation (plan §24 / §36).

Prompt 2 scope: relation components. Model-level tests (biaxis_p1) extend this
file in Prompt 3.
"""

from __future__ import annotations

import pytest
import torch

from src.models.biaxis_p1_components import (
    EdgeStructuralToken,
    FactorGraphBudget,
    FactorRelationSelector,
    RelationPrototypes,
    TopologyDiffusionSignature,
    compute_degree,
    compute_raw_struct_signature,
    neighbor_mean,
    relation_availability,
    relation_mass,
    relation_weighted_mean,
    zscore_columns,
)

NUM_NODES = 17
RELATION_DIM = 32
NUM_RELATIONS = 4


def _make_edge_index(num_edges: int = 40, num_nodes: int = NUM_NODES, seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, num_nodes, (2, num_edges), generator=generator)


def _tiny_graph() -> torch.Tensor:
    """Hand-computable graph: edges 0->1, 2->1, 1->3, 0->3 on 4 nodes."""
    return torch.tensor([[0, 2, 1, 0], [1, 1, 3, 3]], dtype=torch.long)


def _make_decomposition_modules(**kwargs) -> tuple:
    kwargs.setdefault("relation_dim", RELATION_DIM)
    kwargs.setdefault("num_relations", NUM_RELATIONS)
    sig = TopologyDiffusionSignature(relation_dim=kwargs["relation_dim"])
    edge = EdgeStructuralToken(relation_dim=kwargs["relation_dim"])
    proto = RelationPrototypes(
        num_relations=kwargs["num_relations"],
        relation_dim=kwargs["relation_dim"],
        temperature=kwargs.get("temperature", 0.5),
    )
    return sig, edge, proto


# ---------------------------------------------------------------------------
# Raw topology signature (plan §6)
# ---------------------------------------------------------------------------


def test_raw_signature_matches_hand_computation() -> None:
    edge_index = _tiny_graph()
    s = compute_raw_struct_signature(edge_index, num_nodes=4)
    log3 = torch.log(torch.tensor(3.0))
    expected_u0 = torch.tensor([0.0, log3, 0.0, log3])
    expected_u1 = torch.tensor([0.0, 0.0, 0.0, log3 / 2])
    expected_u2 = torch.tensor([0.0, 0.0, 0.0, 0.0])
    assert torch.allclose(s[:, 0], expected_u0, atol=1e-6)
    assert torch.allclose(s[:, 1], expected_u1, atol=1e-6)
    assert torch.allclose(s[:, 2], expected_u2, atol=1e-6)


def test_raw_signature_shape_and_finite() -> None:
    edge_index = _make_edge_index()
    s = compute_raw_struct_signature(edge_index, NUM_NODES)
    assert s.shape == (NUM_NODES, 3)
    assert torch.isfinite(s).all()
    assert s.dtype == torch.float32


def test_raw_signature_isolated_nodes_are_zero() -> None:
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)  # node 3 isolated
    s = compute_raw_struct_signature(edge_index, num_nodes=4)
    assert torch.equal(s[3], torch.zeros(3))


def test_raw_signature_empty_graph_zeros() -> None:
    s = compute_raw_struct_signature(torch.empty(2, 0, dtype=torch.long), num_nodes=4)
    assert torch.equal(s, torch.zeros(4, 3))


def test_zscore_zero_mean_unit_population_std() -> None:
    raw = torch.tensor([[1.0, 3.0], [2.0, 4.0], [3.0, 5.0]])
    z = zscore_columns(raw)
    assert torch.allclose(z.mean(dim=0), torch.zeros(2), atol=1e-5)
    assert torch.allclose(z.std(dim=0, unbiased=False), torch.ones(2), atol=1e-5)


def test_zscore_constant_column_not_nan() -> None:
    raw = torch.tensor([[2.0, 1.0], [2.0, 5.0]])
    z = zscore_columns(raw)
    assert torch.isfinite(z).all()


# ---------------------------------------------------------------------------
# Signature / edge token / prototypes
# ---------------------------------------------------------------------------


def test_signature_mlp_output_shape() -> None:
    sig, _, _ = _make_decomposition_modules()
    s = sig(_make_edge_index(), NUM_NODES)
    assert s.shape == (NUM_NODES, RELATION_DIM)


def test_edge_token_shape() -> None:
    _, edge, _ = _make_decomposition_modules()
    sig, _, _ = _make_decomposition_modules()
    s = sig(_make_edge_index(), NUM_NODES)
    e = edge(s, _make_edge_index())
    assert e.shape == (40, RELATION_DIM)


def test_edge_token_reverse_edge_consistency() -> None:
    """Same undirected edge seen in both directions gets identical tokens."""
    sig, edge, _ = _make_decomposition_modules()
    edge_index = torch.tensor([[0, 1, 2], [1, 0, 1]], dtype=torch.long)
    s = sig(edge_index, num_nodes=3)
    e = edge(s, edge_index)
    assert torch.allclose(e[0], e[1], atol=1e-6), "token(i,j) != token(j,i)"


def test_relation_prototypes_rows_sum_to_one() -> None:
    _, edge, proto = _make_decomposition_modules()
    sig, _, _ = _make_decomposition_modules()
    s = sig(_make_edge_index(), NUM_NODES)
    e = edge(s, _make_edge_index())
    r = proto(e)
    assert r.shape == (40, NUM_RELATIONS)
    assert (r >= 0).all() and (r <= 1).all()
    assert torch.allclose(r.sum(dim=-1), torch.ones(40), atol=1e-5)


def test_relation_temperature_sharpens_assignment() -> None:
    sig, edge, _ = _make_decomposition_modules()
    s = sig(_make_edge_index(), NUM_NODES)
    e = edge(s, _make_edge_index())
    proto_cold = RelationPrototypes(NUM_RELATIONS, RELATION_DIM, temperature=0.1)
    proto_hot = RelationPrototypes(NUM_RELATIONS, RELATION_DIM, temperature=10.0)
    r_cold = proto_cold(e)
    r_hot = proto_hot(e)
    assert r_cold.max(dim=-1).values.mean() > r_hot.max(dim=-1).values.mean()


# ---------------------------------------------------------------------------
# Sparse aggregation (plan §9)
# ---------------------------------------------------------------------------


def test_degree_matches_hand_computation() -> None:
    deg = compute_degree(_tiny_graph(), num_nodes=4)
    assert torch.equal(deg, torch.tensor([0.0, 2.0, 0.0, 2.0]))


def test_relation_mass_matches_hand_computation() -> None:
    edge_index = _tiny_graph()
    r = torch.tensor(
        [[0.25, 0.75], [0.5, 0.5], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32
    )
    mass = relation_mass(edge_index, r, num_nodes=4)
    expected = torch.tensor(
        [[0.0, 0.0], [0.75, 1.25], [0.0, 0.0], [1.0, 1.0]], dtype=torch.float32
    )
    assert torch.allclose(mass, expected, atol=1e-6)


def test_relation_availability_sums_to_one_on_non_isolated() -> None:
    edge_index = _tiny_graph()
    r = torch.tensor(
        [[0.25, 0.75], [0.5, 0.5], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32
    )
    mass = relation_mass(edge_index, r, num_nodes=4)
    deg = compute_degree(edge_index, 4)
    a = relation_availability(mass, deg)
    # Non-isolated nodes: sum_k a = 1; isolated nodes: a = 0.
    assert torch.allclose(a.sum(dim=-1)[deg > 0], torch.ones(2), atol=1e-5)
    assert torch.equal(a[0], torch.zeros(2))  # isolated -> a = 0


def test_relation_weighted_mean_matches_hand_computation() -> None:
    edge_index = _tiny_graph()
    features = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
    r = torch.tensor(
        [[0.25, 0.75], [0.5, 0.5], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32
    )
    g, mass = relation_weighted_mean(edge_index, r, features, num_nodes=4)
    # Node 1 receives from src 0 and 2: g_tilde = r0*f0 + r1*f2 = [0.25+1.5, 0.75+1.5] = [1.75, 2.25]
    assert torch.allclose(g[1, 0], torch.tensor([1.75 / 0.75]), atol=1e-5)
    assert torch.allclose(g[1, 1], torch.tensor([2.25 / 1.25]), atol=1e-5)
    # Node 3 receives from src 1 and 0: g_tilde = r2*f1 + r3*f0 = [2.0, 1.0], mass = [1.0, 1.0]
    assert torch.allclose(g[3, 0], torch.tensor([2.0 / 1.0]), atol=1e-5)
    assert torch.allclose(g[3, 1], torch.tensor([1.0 / 1.0]), atol=1e-5)
    # Isolated nodes: g = 0 (0 / eps).
    assert torch.equal(g[0], torch.zeros(2, 1))
    assert torch.equal(g[2], torch.zeros(2, 1))
    assert torch.equal(mass[:, 0], torch.tensor([0.0, 0.75, 0.0, 1.0]))


def test_relation_weighted_mean_chunked_matches_full() -> None:
    edge_index = _make_edge_index(num_edges=100)
    features = torch.randn(NUM_NODES, 8)
    r = torch.rand(100, NUM_RELATIONS)
    r = r / r.sum(dim=-1, keepdim=True)
    g_full, m_full = relation_weighted_mean(edge_index, r, features, NUM_NODES, edge_chunk_size=None)
    g_chunk, m_chunk = relation_weighted_mean(edge_index, r, features, NUM_NODES, edge_chunk_size=33)
    assert torch.allclose(g_full, g_chunk, atol=1e-5)
    assert torch.allclose(m_full, m_chunk, atol=1e-6)


def test_relation_weighted_mean_edge_permutation_invariance() -> None:
    edge_index = _make_edge_index(num_edges=100)
    perm = torch.randperm(100)
    edge_index_perm = edge_index[:, perm]
    features = torch.randn(NUM_NODES, 8)
    r = torch.rand(100, NUM_RELATIONS)
    r = r / r.sum(dim=-1, keepdim=True)
    g_a, _ = relation_weighted_mean(edge_index, r, features, NUM_NODES)
    g_b, _ = relation_weighted_mean(edge_index_perm, r[perm], features, NUM_NODES)
    assert torch.allclose(g_a, g_b, atol=1e-5)


def test_neighbor_mean_matches_hand_computation() -> None:
    edge_index = _tiny_graph()
    features = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
    mean = neighbor_mean(edge_index, features, num_nodes=4)
    assert torch.allclose(mean[1], torch.tensor([(1.0 + 3.0) / 2]), atol=1e-6)
    assert torch.allclose(mean[3], torch.tensor([(2.0 + 1.0) / 2]), atol=1e-6)
    assert torch.equal(mean[0], torch.zeros(1))  # isolated -> 0


def test_relation_averaged_context_equals_neighbor_mean() -> None:
    """Audit §4 simplification: sum_k a_ik g_ik^f == plain neighbor mean."""
    edge_index = _make_edge_index(num_edges=100)
    features = torch.randn(NUM_NODES, 8)
    r = torch.rand(100, NUM_RELATIONS)
    r = r / r.sum(dim=-1, keepdim=True)
    g, mass = relation_weighted_mean(edge_index, r, features, NUM_NODES)
    deg = compute_degree(edge_index, NUM_NODES)
    a = relation_availability(mass, deg)
    g_bar = (a.unsqueeze(-1) * g).sum(dim=1)  # [N, d]
    plain = neighbor_mean(edge_index, features, NUM_NODES)
    assert torch.allclose(g_bar, plain, atol=1e-4)


# ---------------------------------------------------------------------------
# Gradient flow (plan §24.9, M2 part)
# ---------------------------------------------------------------------------


def test_gradients_flow_to_signature_edge_prototypes() -> None:
    sig, edge, proto = _make_decomposition_modules()
    edge_index = _make_edge_index()
    s = sig(edge_index, NUM_NODES)
    e = edge(s, edge_index)
    r = proto(e)
    # NOT r.mean(): sum_k r_k = 1 makes mean(r) a constant with zero effective
    # gradient (review §20). Use a single slice so the loss is non-constant.
    r[:, 0].mean().backward()
    for module in (sig, edge, proto):
        params = list(module.parameters())
        assert params
        for p in params:
            assert p.grad is not None, f"missing gradient in {type(module).__name__}"
            assert torch.isfinite(p.grad).all(), f"non-finite gradient in {type(module).__name__}"
            assert p.grad.norm() > 1e-9, f"zero effective gradient in {type(module).__name__}"


# ---------------------------------------------------------------------------
# M3 shapes and init conventions (Prompt 3 scope, added early for completeness)
# ---------------------------------------------------------------------------


def test_budget_init_approximately_half() -> None:
    budget = FactorGraphBudget(factor_dim=16, hidden_dim=32)
    f = torch.randn(5, 3, 16)
    g_bar = torch.randn(5, 3, 16)
    beta = budget(f, g_bar)
    assert beta.shape == (5, 3)
    assert torch.allclose(beta, torch.full((5, 3), 0.5), atol=1e-6), "beta must start at ~0.5"


def test_budget_bounds() -> None:
    budget = FactorGraphBudget(factor_dim=16, hidden_dim=32)
    budget.net[-1].weight.data.normal_(0, 1)
    beta = budget(torch.randn(5, 3, 16), torch.randn(5, 3, 16))
    assert (beta > 0).all() and (beta < 1).all()


def test_selector_k1_fast_path() -> None:
    selector = FactorRelationSelector(num_relations=1, factor_dim=16, hidden_dim=32)
    alpha = selector(torch.randn(5, 3, 16), torch.randn(5, 3, 1, 16), torch.randn(5, 1))
    assert alpha.shape == (5, 3, 1)
    assert torch.equal(alpha, torch.ones(5, 3, 1))


def test_selector_simplex() -> None:
    selector = FactorRelationSelector(num_relations=4, factor_dim=16, hidden_dim=32)
    f = torch.randn(5, 3, 16)
    g = torch.randn(5, 3, 4, 16)
    a = torch.rand(5, 4)
    alpha = selector(f, g, a)
    assert alpha.shape == (5, 3, 4)
    assert (alpha >= 0).all()
    assert torch.allclose(alpha.sum(dim=-1), torch.ones(5, 3), atol=1e-5)


def test_selector_gradient_flow() -> None:
    selector = FactorRelationSelector(num_relations=4, factor_dim=16, hidden_dim=32)
    alpha = selector(torch.randn(5, 3, 16), torch.randn(5, 3, 4, 16), torch.randn(5, 4))
    # NOT alpha.mean(): sum_k alpha_k = 1 makes mean(alpha) a constant
    # (review §20). Use a single slice so the gradient is non-zero.
    alpha[:, :, 0].mean().backward()
    for name, p in selector.named_parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
        if "bias" in name:
            # Softmax translation invariance: the final bias gradient is
            # exactly zero for ANY loss depending only on alpha — expected.
            continue
        assert p.grad.norm() > 1e-9, f"zero effective gradient in selector {name}"


# ---------------------------------------------------------------------------
# biaxis_p1 model (Prompt 3 scope: plan §24 / §37)
# ---------------------------------------------------------------------------

from omegaconf import OmegaConf

from src.models.biaxis_p1 import Model as P1Model

P1_TEXT_DIM = 13
P1_VISUAL_DIM = 19
P1_NUM_NODES = 17
P1_FACTOR_DIM = 128
P1_HIDDEN_DIM = 256


def _make_p1_cfg(**p1_overrides) -> object:
    model_cfg = {
        "name": "biaxis_p1",
        "hidden_dim": P1_HIDDEN_DIM,
        "factor_dim": P1_FACTOR_DIM,
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
            "budget_hidden_dim": 64,
            "use_graph_budget": True,
            "budget_shared": False,
            "eps": 1.0e-8,
            "relation_balance_weight": 0.0,
            "alpha_entropy_weight": 0.0,
            "budget_reg_weight": 0.0,
            "edge_chunk_size": None,
        },
    }
    model_cfg["p1"].update(p1_overrides)
    return OmegaConf.create({"model": model_cfg})


def _make_p1_data_info(**overrides) -> dict:
    info = {
        "input_dim": P1_TEXT_DIM + P1_VISUAL_DIM,
        "num_nodes": P1_NUM_NODES,
        "num_classes": 5,
        "text_dim": P1_TEXT_DIM,
        "visual_dim": P1_VISUAL_DIM,
    }
    info.update(overrides)
    return info


def _make_p1_x(num_nodes: int = P1_NUM_NODES, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(num_nodes, P1_TEXT_DIM + P1_VISUAL_DIM, generator=generator)


def _make_p1_edge_index(num_edges: int = 60, num_nodes: int = P1_NUM_NODES, seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, num_nodes, (2, num_edges), generator=generator)


def _make_p1_model(**p1_overrides) -> P1Model:
    return P1Model(_make_p1_cfg(**p1_overrides), _make_p1_data_info())


def test_p1_f1r1_shapes() -> None:
    model = _make_p1_model()  # F1R1 default
    model.eval()
    edge_index = _make_p1_edge_index()
    x = _make_p1_x()
    z, second, third, aux, _ = model(x, edge_index)
    assert second is None and third is None
    assert z.shape == (P1_NUM_NODES, P1_HIDDEN_DIM)
    r, a, _ = model._decompose_relations(edge_index, P1_NUM_NODES)
    assert r.shape == (60, 4)
    f_block = torch.randn(P1_NUM_NODES, 3, P1_FACTOR_DIM)
    out = model._graph_update(f_block, edge_index, P1_NUM_NODES)
    assert out["beta"].shape == (P1_NUM_NODES, 3)
    assert out["alpha"].shape == (P1_NUM_NODES, 3, 4)
    assert out["f_tilde"].shape == (P1_NUM_NODES, 3, P1_FACTOR_DIM)
    assert torch.isfinite(z).all()


def test_p1_relation_simplex() -> None:
    model = _make_p1_model()
    model.eval()
    edge_index = _make_p1_edge_index()
    r, _, _ = model._decompose_relations(edge_index, P1_NUM_NODES)
    assert (r >= 0).all() and (r <= 1).all()
    assert torch.allclose(r.sum(dim=-1), torch.ones(60), atol=1e-5)


def test_p1_alpha_simplex() -> None:
    model = _make_p1_model()
    model.eval()
    edge_index = _make_p1_edge_index()
    f_block = torch.randn(P1_NUM_NODES, 3, P1_FACTOR_DIM)
    out = model._graph_update(f_block, edge_index, P1_NUM_NODES)
    alpha = out["alpha"]
    assert (alpha >= 0).all() and (alpha <= 1).all()
    assert torch.allclose(alpha.sum(dim=-1), torch.ones(P1_NUM_NODES, 3), atol=1e-5)


def test_p1_budget_bounds() -> None:
    model = _make_p1_model()
    model.eval()
    edge_index = _make_p1_edge_index()
    f_block = torch.randn(P1_NUM_NODES, 3, P1_FACTOR_DIM)
    out = model._graph_update(f_block, edge_index, P1_NUM_NODES)
    beta = out["beta"]
    assert (beta > 0).all() and (beta < 1).all()


def test_p1_budget_init_is_half_and_trainable() -> None:
    model = _make_p1_model()
    model.train()
    edge_index = _make_p1_edge_index()
    f_block = torch.randn(P1_NUM_NODES, 3, P1_FACTOR_DIM)
    out = model._graph_update(f_block, edge_index, P1_NUM_NODES)
    assert torch.allclose(out["beta"], torch.full((P1_NUM_NODES, 3), 0.5), atol=1e-6)
    # After a training step beta must move off 0.5.
    x = _make_p1_x()
    z, _, _, aux, _ = model(x, edge_index)
    (z.square().mean() + aux).backward()
    assert model.graph_budget.net[-1].weight.grad is not None
    assert model.graph_budget.net[-1].weight.grad.abs().sum() > 0


def test_p1_k1_fast_path() -> None:
    model = _make_p1_model(num_relations=1)
    model.eval()
    edge_index = _make_p1_edge_index()
    r, a, deg = model._decompose_relations(edge_index, P1_NUM_NODES)
    assert r.shape == (60, 1)
    assert torch.equal(r, torch.ones(60, 1))
    f_block = torch.randn(P1_NUM_NODES, 3, P1_FACTOR_DIM)
    out = model._graph_update(f_block, edge_index, P1_NUM_NODES)
    assert torch.equal(out["alpha"], torch.ones(P1_NUM_NODES, 3, 1))
    # K=1 relation weighted mean must equal the plain neighbor mean.
    f_cat = f_block.reshape(P1_NUM_NODES, 3 * P1_FACTOR_DIM)
    from src.models.biaxis_p1_components import neighbor_mean as _neighbor_mean
    plain = _neighbor_mean(edge_index, f_cat, P1_NUM_NODES)
    g_cat, _mass = relation_weighted_mean(edge_index, r, f_cat, P1_NUM_NODES)
    assert torch.allclose(g_cat[:, 0], plain, atol=1e-4)


def test_p1_relation_topology_only() -> None:
    """Same edge_index with different x must leave the relation decomposition
    (and its cached raw signature input) exactly identical (plan §24.6)."""
    model = _make_p1_model()
    model.eval()
    edge_index = _make_p1_edge_index()
    x1 = _make_p1_x(seed=0)
    x2 = _make_p1_x(seed=99)
    r1, _, _ = model._decompose_relations(edge_index, P1_NUM_NODES)
    cache1 = model._sig_cache_raw.clone()
    model(x1, edge_index)
    model(x2, edge_index)
    r2, _, _ = model._decompose_relations(edge_index, P1_NUM_NODES)
    assert torch.equal(r1, r2)
    assert torch.equal(cache1, model._sig_cache_raw)


def test_p1_edge_permutation_invariance() -> None:
    model = _make_p1_model()
    model.eval()
    edge_index = _make_p1_edge_index()
    perm = torch.randperm(edge_index.size(1))
    x = _make_p1_x()
    z_a, _, _, _, _ = model(x, edge_index)
    z_b, _, _, _, _ = model(x, edge_index[:, perm])
    assert torch.allclose(z_a, z_b, atol=1e-4)


def test_p1_reverse_edge_consistency() -> None:
    """Both directions of the same undirected edge get identical r rows."""
    model = _make_p1_model()
    model.eval()
    # Every edge (i, j) is present in both directions.
    src = torch.randint(0, 8, (30,))
    dst = torch.randint(0, 8, (30,))
    keep = src != dst
    edge_index = torch.stack([torch.cat([src[keep], dst[keep]]), torch.cat([dst[keep], src[keep]])], dim=0)
    r, _, _ = model._decompose_relations(edge_index, 8)
    num_pairs = int(keep.sum())
    assert torch.allclose(r[:num_pairs], r[num_pairs:], atol=1e-6)


def test_p1_gradients_flow_to_all_components() -> None:
    model = _make_p1_model()  # F1R1: exercises every graph module
    model.train()
    x = _make_p1_x()
    edge_index = _make_p1_edge_index()
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
        model.graph_budget,
        model.factor_selector,
        model.graph_w0,
        model.fusion,
    ]
    for component in components:
        params = list(component.parameters())
        assert params, f"no parameters in {component}"
        assert all(p.grad is not None for p in params), f"missing gradient in {component}"
        assert all(torch.isfinite(p.grad).all() for p in params), f"non-finite gradient in {component}"


def test_p1_inference_forward_equivalence() -> None:
    model = _make_p1_model()
    model.eval()
    x = _make_p1_x()
    edge_index = _make_p1_edge_index()
    z_fwd, _, _, _, _ = model(x, edge_index)
    z_inf = model.inference(x, edge_index, device=torch.device("cpu"))
    assert z_inf.shape == (P1_NUM_NODES, P1_HIDDEN_DIM)
    assert torch.allclose(z_fwd, z_inf, atol=1e-6)


def test_p1_variant_switches() -> None:
    """All four variants forward cleanly with the right graph-state shapes."""
    for factor_aware in (False, True):
        for num_relations in (1, 4):
            model = _make_p1_model(factor_aware=factor_aware, num_relations=num_relations)
            model.eval()
            edge_index = _make_p1_edge_index()
            x = _make_p1_x()
            z, _, _, _, _ = model(x, edge_index)
            assert z.shape == (P1_NUM_NODES, P1_HIDDEN_DIM)
            assert torch.isfinite(z).all()
            num_factors = 3 if factor_aware else 1
            f_block = torch.randn(P1_NUM_NODES, num_factors, P1_FACTOR_DIM)
            out = model._graph_update(f_block, edge_index, P1_NUM_NODES)
            assert out["beta"].shape == (P1_NUM_NODES, num_factors)
            assert out["alpha"].shape == (P1_NUM_NODES, num_factors, num_relations)
            assert out["r"].shape == (60, num_relations)
            if factor_aware:
                assert torch.isfinite(model.fusion[0].weight).all()
            else:
                assert torch.isfinite(model.proj_q[0].weight).all()
                assert torch.isfinite(model.fusion_q[0].weight).all()


def test_p1_factor_blind_has_single_graph_state() -> None:
    model = _make_p1_model(factor_aware=False)
    model.eval()
    edge_index = _make_p1_edge_index()
    f_block = torch.randn(P1_NUM_NODES, 1, P1_FACTOR_DIM)
    out = model._graph_update(f_block, edge_index, P1_NUM_NODES)
    assert out["beta"].shape == (P1_NUM_NODES, 1)
    assert out["alpha"].shape == (P1_NUM_NODES, 1, 4)
    assert out["f_tilde"].shape == (P1_NUM_NODES, 1, P1_FACTOR_DIM)


def test_p1_aux_loss_preserved() -> None:
    model = _make_p1_model()
    model.train()
    x = _make_p1_x()
    edge_index = _make_p1_edge_index()
    _, _, _, aux, aux_info = model(x, edge_index)
    assert aux.ndim == 0 and torch.isfinite(aux)
    for key in ("p0_common_loss", "p0_orth_loss", "p0_recon_loss", "p0_common_sim", "p0_private_sim"):
        assert key in aux_info, f"missing aux_info key {key}"


def test_p1_isolated_nodes_no_nan() -> None:
    model = _make_p1_model()
    # Node 16 is isolated: no incoming edges.
    edge_index = _make_p1_edge_index(num_edges=60, num_nodes=17)
    edge_index = torch.randint(0, 16, (2, 60))
    x = _make_p1_x()
    model.eval()
    z, _, _, aux_eval, _ = model(x, edge_index)
    assert torch.isfinite(z).all()
    model.train()
    z, _, _, aux_train, _ = model(x, edge_index)
    assert torch.isfinite(z).all()
    assert torch.isfinite(aux_train)
    assert aux_eval.item() == 0.0


def test_p1_regularizers_off_by_default() -> None:
    model = _make_p1_model()
    assert model.relation_balance_weight == 0.0
    assert model.alpha_entropy_weight == 0.0
    assert model.budget_reg_weight == 0.0
    model.eval()
    z1, _, _, _, _ = model(_make_p1_x(), _make_p1_edge_index())
    z2, _, _, _, _ = model(_make_p1_x(seed=5), _make_p1_edge_index(seed=2))
    assert torch.isfinite(z1).all() and torch.isfinite(z2).all()


def test_p1_relation_balance_loss_finite_when_enabled() -> None:
    model = _make_p1_model(relation_balance_weight=1e-3)
    model.train()
    edge_index = _make_p1_edge_index()
    x = _make_p1_x()
    _, _, _, aux, _ = model(x, edge_index)
    assert torch.isfinite(aux)
    reg = model._graph_regularization(
        torch.rand(60, 4) / 4, torch.rand(P1_NUM_NODES, 3, 4) / 4, torch.rand(P1_NUM_NODES, 3)
    )
    assert reg.ndim == 0 and torch.isfinite(reg)


def test_p1_budget_shared_variant() -> None:
    model = _make_p1_model(budget_shared=True)
    model.eval()
    edge_index = _make_p1_edge_index()
    f_block = torch.randn(P1_NUM_NODES, 3, P1_FACTOR_DIM)
    out = model._graph_update(f_block, edge_index, P1_NUM_NODES)
    beta = out["beta"]
    assert beta.shape == (P1_NUM_NODES, 3)
    # Shared budget: all factors carry the same per-node value.
    assert torch.equal(beta[:, 0], beta[:, 1])
    assert torch.equal(beta[:, 1], beta[:, 2])


def test_p1_no_graph_budget_variant() -> None:
    model = _make_p1_model(use_graph_budget=False)
    model.eval()
    edge_index = _make_p1_edge_index()
    f_block = torch.randn(P1_NUM_NODES, 3, P1_FACTOR_DIM)
    out = model._graph_update(f_block, edge_index, P1_NUM_NODES)
    assert torch.equal(out["beta"], torch.ones(P1_NUM_NODES, 3))


# ---------------------------------------------------------------------------
# Mechanism diagnostics (Prompt 4 scope: plan §19)
# ---------------------------------------------------------------------------


def test_p1_diagnostics_f1r1_structure() -> None:
    model = _make_p1_model()
    model.eval()
    x = _make_p1_x()
    edge_index = _make_p1_edge_index()
    diag = model.compute_p1_diagnostics(x, edge_index)
    # Relation (§19.1)
    assert len(diag["relation"]["occ"]) == 4
    assert abs(sum(diag["relation"]["occ"]) - 1.0) < 1e-5
    assert 1.0 <= diag["relation"]["effective_num"] <= 4.0 + 1e-6
    assert diag["relation"]["mean_edge_entropy"] >= 0
    # Budget (§19.2): three factors, all stats present and bounded.
    for name in ("C", "Pt", "Pv"):
        b = diag["budget"][name]
        assert 0.0 <= b["mean"] <= 1.0
        assert 0.0 <= b["p10"] <= b["p50"] <= b["p90"] <= 1.0
        assert 0.0 <= b["low_frac"] <= 1.0 and 0.0 <= b["high_frac"] <= 1.0
    # Selector (§19.3)
    for name in ("C", "Pt", "Pv"):
        assert diag["alpha_entropy"][name] >= 0
    for key in ("C_Pt", "C_Pv", "Pt_Pv"):
        assert diag["alpha_js"][key] >= 0
    # Usage matrix (§19.4)
    um = diag["usage_matrix"]
    assert um["factors"] == ["C", "Pt", "Pv"]
    assert um["relations"] == ["R1", "R2", "R3", "R4"]
    assert len(um["values"]) == 3 and all(len(row) == 4 for row in um["values"])


def test_p1_diagnostics_factor_blind_single_factor() -> None:
    model = _make_p1_model(factor_aware=False)
    model.eval()
    diag = model.compute_p1_diagnostics(_make_p1_x(), _make_p1_edge_index())
    assert diag["usage_matrix"]["factors"] == ["q"]
    assert list(diag["budget"].keys()) == ["q"]
    assert diag["alpha_js"] == {}


def test_p1_diagnostics_json_safe() -> None:
    import json

    model = _make_p1_model()
    model.eval()
    diag = model.compute_p1_diagnostics(_make_p1_x(), _make_p1_edge_index())
    json.dumps(diag)  # must not raise (no tensors / NaN)
