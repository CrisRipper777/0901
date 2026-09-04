"""R1.5 protocol extensions (plan §4/§9): task.evaluate_test=false must skip
ALL test access/metrics; task.history_path writes the per-epoch history CSV
with the documented columns."""

from __future__ import annotations

import csv

import torch

from src.tasks import nc as nc_task
from src.tasks.nc import _evaluate_split, _run_single_nc


def test_evaluate_split_never_uses_test_idx() -> None:
    classifier = torch.nn.Linear(8, 3)
    z = torch.randn(20, 8)
    labels = torch.randint(0, 3, (20,))
    val_idx = torch.arange(0, 10)
    out = _evaluate_split(classifier, z, labels, val_idx, torch.device("cpu"), 64)
    assert set(out) == {"acc", "macro_f1"}
    assert 0.0 <= out["acc"] <= 1.0


def test_run_nc_val_only_skips_test_access() -> None:
    """evaluate_test=false: results carry ONLY val keys. Use a tiny real
    training run (mlp, 2 epochs) to exercise the full trainer path with a
    history file; the TEST tensors are replaced by sentinels and must never
    be touched (any access would raise)."""
    import logging
    import tempfile
    from pathlib import Path

    import numpy as np
    from omegaconf import OmegaConf

    from src.data import MAGData

    n = 40
    torch.manual_seed(0)
    x = torch.randn(n, 16)
    y = torch.randint(0, 3, (n,))
    cfg = OmegaConf.create({
        "seed": 42,
        "num_runs": 1,
        "device": "cpu",
        "dataset": {"name": "synthetic", "tasks": ["nc"]},
        "model": {
            "name": "mlp",
            "hidden_dim": 16,
            "num_layers": 2,
            "dropout": 0.1,
        },
        "task": {
            "epochs": 2,
            "patience": 5,
            "eval_every": 1,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "optimizer": "adamw",
            "batch_size": 256,
            "inference_mode": "full",
            "inference_batch_size": 64,
            "early_stop_min_epoch": 1,
            "grad_clip": 1.0,
            "max_train_batches": None,
            "loss": {"aux_weight": 1.0},
            "evaluate_test": False,
            "history_path": None,
            "save_ckpt_path": None,
        },
        "paths": {"data_root": ".", "split_root": "."},
    })

    class _Exploding:
        """Any access to test labels blows up -> proves the trainer never
        touches test when evaluate_test=false."""

        def __len__(self):
            raise RuntimeError("test_idx accessed")

        def cpu(self):
            raise RuntimeError("test_idx accessed")

        def to(self, *a, **k):
            raise RuntimeError("test_idx accessed")

        def numpy(self):
            raise RuntimeError("test_idx accessed")

    data = MAGData(
        name="synthetic", source="synthetic", task="nc", x=x, x_i=None, x_t=None,
        edge_index=None, y=y, train_idx=torch.arange(0, 24), val_idx=torch.arange(24, 32),
        test_idx=_Exploding(), num_nodes=n, num_classes=3, info={},
    )
    results = nc_task.run_nc(cfg, data, torch.device("cpu"), logging.getLogger("test"))
    assert set(results) == {"val_acc"}
    assert 0.0 <= results["val_acc"][0] <= 1.0


def test_run_nc_history_csv_written() -> None:
    """history_path set -> CSV with the documented columns, one row per
    evaluated epoch, val-only results."""
    import logging
    import tempfile
    from pathlib import Path

    from omegaconf import OmegaConf

    from src.data import MAGData

    n = 40
    torch.manual_seed(1)
    x = torch.randn(n, 16)
    y = torch.randint(0, 3, (n,))
    with tempfile.TemporaryDirectory() as tmp:
        hist = Path(tmp) / "history.csv"
        cfg = OmegaConf.create({
            "seed": 42,
            "num_runs": 1,
            "device": "cpu",
            "dataset": {"name": "synthetic", "tasks": ["nc"]},
            "model": {"name": "mlp", "hidden_dim": 16, "num_layers": 2, "dropout": 0.1},
            "task": {
                "epochs": 2, "patience": 5, "eval_every": 1, "lr": 0.001,
                "weight_decay": 0.0001,
                "optimizer": "adamw", "batch_size": 256, "inference_mode": "full",
                "inference_batch_size": 64, "early_stop_min_epoch": 1,
                "grad_clip": 1.0, "max_train_batches": None,
                "loss": {"aux_weight": 1.0},
                "evaluate_test": False,
                "history_path": str(hist),
                "save_ckpt_path": None,
            },
            "paths": {"data_root": ".", "split_root": "."},
        })
        data = MAGData(
            name="synthetic", source="synthetic", task="nc", x=x, x_i=None, x_t=None,
            edge_index=None, y=y, train_idx=torch.arange(0, 24), val_idx=torch.arange(24, 32),
            test_idx=torch.arange(32, 40), num_nodes=n, num_classes=3, info={},
        )
        results = nc_task.run_nc(cfg, data, torch.device("cpu"), logging.getLogger("test"))
        assert set(results) == {"val_acc"}
        assert hist.exists()
        with hist.open(newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert rows[0][:9] == [
            "epoch", "train_total_loss", "train_ce_loss", "train_aux_loss",
            "train_acc", "val_acc", "val_macro_f1", "lr", "patience_left",
        ]
        assert len(rows) == 1 + 2  # header + 2 evaluated epochs
        for row in rows[1:]:
            assert int(row[0]) in (1, 2)
            assert float(row[5]) >= 0.0  # val_acc


def test_scheduler_warmup_cosine_changes_lr() -> None:
    """task.scheduler=warmup_cosine must change the lr across epochs (the
    history CSV records per-epoch lr)."""
    import logging
    import tempfile
    from pathlib import Path

    from omegaconf import OmegaConf

    from src.data import MAGData

    n = 40
    torch.manual_seed(2)
    x = torch.randn(n, 16)
    y = torch.randint(0, 3, (n,))
    with tempfile.TemporaryDirectory() as tmp:
        hist = Path(tmp) / "history.csv"
        cfg = OmegaConf.create({
            "seed": 42, "num_runs": 1, "device": "cpu",
            "dataset": {"name": "synthetic", "tasks": ["nc"]},
            "model": {"name": "mlp", "hidden_dim": 16, "num_layers": 2, "dropout": 0.1},
            "task": {
                "epochs": 6, "patience": 5, "eval_every": 1, "lr": 0.001,
                "weight_decay": 0.0001, "optimizer": "adamw", "batch_size": 256,
                "inference_mode": "full", "inference_batch_size": 64,
                "early_stop_min_epoch": 1, "grad_clip": 1.0,
                "max_train_batches": None, "loss": {"aux_weight": 1.0},
                "evaluate_test": False, "history_path": str(hist),
                "save_ckpt_path": None,
                "scheduler": "warmup_cosine", "scheduler_warmup_epochs": 10,
                "scheduler_min_lr": 1.0e-5,
            },
            "paths": {"data_root": ".", "split_root": "."},
        })
        data = MAGData(
            name="synthetic", source="synthetic", task="nc", x=x, x_i=None, x_t=None,
            edge_index=None, y=y, train_idx=torch.arange(0, 24),
            val_idx=torch.arange(24, 32), test_idx=torch.arange(32, 40),
            num_nodes=n, num_classes=3, info={},
        )
        nc_task.run_nc(cfg, data, torch.device("cpu"), logging.getLogger("test"))
        import csv as csv_mod
        with hist.open(newline="", encoding="utf-8") as f:
            rows = list(csv_mod.reader(f))
        lrs = [float(r[7]) for r in rows[1:]]
        assert len(set(lrs)) >= 2, f"lr should vary across epochs, got {lrs}"
        assert all(0.0 < lr <= 0.001 + 1e-9 for lr in lrs)
