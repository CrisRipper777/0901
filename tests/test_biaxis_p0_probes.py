from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.utils import to_undirected

from src.data import EdgeSplit, MAGData
from src.utils.biaxis_p0_probes import assert_no_edge_leakage, run_lp_factor_probes, run_nc_factor_probes


def _make_synthetic_nc(num_nodes: int = 64, factor_dim: int = 8, num_classes: int = 4, seed: int = 0) -> tuple[dict, MAGData]:
    generator = torch.Generator().manual_seed(seed)
    c = torch.randn(num_nodes, factor_dim, generator=generator)
    p_t = torch.randn(num_nodes, factor_dim, generator=generator)
    p_v = torch.randn(num_nodes, factor_dim, generator=generator)
    factors = {"c": c, "c_t": c, "c_v": c, "p_t": p_t, "p_v": p_v, "z_local": torch.randn(num_nodes, 16, generator=generator)}

    edge_index = torch.randint(0, num_nodes, (2, 200), generator=generator)
    labels = torch.randint(0, num_classes, (num_nodes,), generator=generator)
    perm = torch.randperm(num_nodes, generator=generator)
    train_idx = perm[:32]
    val_idx = perm[32:48]
    test_idx = perm[48:]
    data = MAGData(
        name="synth",
        source="test",
        task="nc",
        x=torch.randn(num_nodes, 20, generator=generator),
        edge_index=edge_index,
        num_nodes=num_nodes,
        y=labels,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        num_classes=num_classes,
    )
    return factors, data


def test_nc_factor_probes_end_to_end(tmp_path) -> None:
    factors, data = _make_synthetic_nc()
    probe_cfg = OmegaConf.create({"epochs": 3, "patience": 1, "lr": 0.001, "weight_decay": 0.0001})
    result = run_nc_factor_probes(
        factors,
        data,
        torch.device("cpu"),
        probe_cfg,
        output_dir=tmp_path,
        include_test=True,
        seed=0,
        batch_size=64,
    )

    rows = result["nc_probe"]
    assert len(rows) == 6  # 3 factors x (local, graph)
    for row in rows:
        assert row["factor"] in {"C", "Pt", "Pv"}
        assert row["mode"] in {"local", "graph"}
        for key in ("val_acc", "val_macro_f1", "test_acc", "test_macro_f1"):
            assert 0.0 <= row[key] <= 1.0

    assert len(result["nc_node_delta"]) == 3
    for name, tensors in result["nc_node_delta"].items():
        assert tensors["delta"].shape == (16,)  # val_idx size
        assert torch.isfinite(tensors["delta"]).all()

    conflict = result["nc_conflict"]
    for key in ("conflict_C_Pt", "conflict_C_Pv", "conflict_Pt_Pv", "pattern_all_help", "pattern_mixed"):
        assert key in conflict
        assert torch.isfinite(torch.tensor(conflict[key]))

    assert (tmp_path / "nc_probe.csv").exists()
    assert (tmp_path / "nc_node_delta.pt").exists()
    assert (tmp_path / "conflict_stats.json").exists()


def test_nc_probe_test_gated_off_by_default(tmp_path) -> None:
    factors, data = _make_synthetic_nc()
    probe_cfg = OmegaConf.create({"epochs": 2, "patience": 1, "lr": 0.001, "weight_decay": 0.0001})
    result = run_nc_factor_probes(
        factors, data, torch.device("cpu"), probe_cfg, output_dir=tmp_path, include_test=False, seed=0
    )
    for row in result["nc_probe"]:
        assert row["test_acc"] is None
        assert row["test_macro_f1"] is None


# ---------------------------------------------------------------------------
# LP probe
# ---------------------------------------------------------------------------


def _make_synthetic_lp(num_nodes: int = 40, factor_dim: int = 8, seed: int = 0) -> tuple[dict, MAGData]:
    generator = torch.Generator().manual_seed(seed)
    c = torch.randn(num_nodes, factor_dim, generator=generator)
    p_t = torch.randn(num_nodes, factor_dim, generator=generator)
    p_v = torch.randn(num_nodes, factor_dim, generator=generator)
    factors = {"c": c, "c_t": c, "c_v": c, "p_t": p_t, "p_v": p_v, "z_local": torch.randn(num_nodes, 16, generator=generator)}

    forbidden: set[tuple[int, int]] = set()

    def sample_edges(num: int) -> tuple[list[int], list[int]]:
        out_src: list[int] = []
        out_dst: list[int] = []
        while len(out_src) < num:
            s = int(torch.randint(0, num_nodes, (1,), generator=generator))
            d = int(torch.randint(0, num_nodes, (1,), generator=generator))
            key = (min(s, d), max(s, d))
            if s != d and key not in forbidden:
                forbidden.add(key)
                out_src.append(s)
                out_dst.append(d)
        return out_src, out_dst

    def pack(src: list[int], dst: list[int], neg: int) -> dict[str, torch.Tensor]:
        return {
            "source_node": torch.tensor(src, dtype=torch.long),
            "target_node": torch.tensor(dst, dtype=torch.long),
            "target_node_neg": torch.randint(0, num_nodes, (len(src), neg), generator=generator),
        }

    train_src, train_dst = sample_edges(60)
    valid_src, valid_dst = sample_edges(12)
    test_src, test_dst = sample_edges(12)
    edge_split = EdgeSplit(
        train=pack(train_src, train_dst, neg=0),
        valid=pack(valid_src, valid_dst, neg=5),
        test=pack(test_src, test_dst, neg=5),
        metadata={"undirected": True},
    )
    train_edge_index = to_undirected(
        torch.stack(
            [torch.tensor(train_src, dtype=torch.long), torch.tensor(train_dst, dtype=torch.long)]
        ),
        num_nodes=num_nodes,
    )
    data = MAGData(
        name="synth-lp",
        source="test",
        task="lp",
        x=torch.randn(num_nodes, 20, generator=generator),
        edge_index=train_edge_index,
        num_nodes=num_nodes,
        edge_split=edge_split,
    )
    return factors, data


def test_lp_probe_end_to_end(tmp_path) -> None:
    factors, data = _make_synthetic_lp()
    probe_cfg = OmegaConf.create(
        {
            "epochs": 3,
            "patience": 1,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "num_train_neg": 1,
            "train_pos_per_epoch": 30,
            "eval_subset_size": None,
        }
    )
    result = run_lp_factor_probes(
        factors,
        data,
        torch.device("cpu"),
        probe_cfg,
        output_dir=tmp_path,
        include_test=True,
        seed=0,
        batch_size=64,
        eval_batch_size=64,
    )
    rows = result["lp_probe"]
    assert len(rows) == 6
    for row in rows:
        for key in ("val_mrr", "val_hits@1", "val_hits@3", "val_hits@10", "test_mrr"):
            assert 0.0 <= row[key] <= 1.0
    for name, tensors in result["lp_edge_delta"].items():
        assert tensors["delta"].shape == (len(data.edge_split.valid["source_node"]),)
        assert torch.isfinite(tensors["delta"]).all()
    conflict = result["lp_conflict"]
    assert "conflict_C_Pt" in conflict
    assert (tmp_path / "lp_probe.csv").exists()
    assert (tmp_path / "lp_edge_delta.pt").exists()
    assert (tmp_path / "conflict_stats.json").exists()


def test_lp_probe_detects_edge_leakage() -> None:
    factors, data = _make_synthetic_lp()
    leaking_edge = torch.tensor(
        [[int(data.edge_split.valid["source_node"][0])], [int(data.edge_split.valid["target_node"][0])]]
    )
    with pytest.raises(ValueError, match="edge leakage"):
        assert_no_edge_leakage(
            torch.cat([data.edge_index, leaking_edge], dim=1), data.edge_split, data.num_nodes
        )
    # train-only graph passes
    assert_no_edge_leakage(data.edge_index, data.edge_split, data.num_nodes)


def test_lp_probe_reverse_direction_overlap_is_inherent() -> None:
    # Public directed splits have valid queries whose REVERSE is a train edge:
    # that overlap must be reported, not flagged (main-protocol semantics).
    train = {"source_node": torch.tensor([1]), "target_node": torch.tensor([2])}
    valid = {
        "source_node": torch.tensor([2]),
        "target_node": torch.tensor([1]),
        "target_node_neg": torch.tensor([[3, 4]]),
    }
    test = {
        "source_node": torch.tensor([0]),
        "target_node": torch.tensor([3]),
        "target_node_neg": torch.tensor([[1, 2]]),
    }
    edge_split = EdgeSplit(train=train, valid=valid, test=test, metadata={"undirected": True})
    message_graph = to_undirected(
        torch.tensor([[1], [2]], dtype=torch.long), num_nodes=5
    )
    report = assert_no_edge_leakage(message_graph, edge_split, num_nodes=5)
    assert report["valid"] == 1  # (2,1) canonical == reverse of train (1,2)
    assert report["test"] == 0
