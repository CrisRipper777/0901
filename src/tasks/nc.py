from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from src.data import MAGData
from src.models import build_model
from src.tasks.common import clone_state_dict, load_state_dict_cpu
from src.utils.metrics import accuracy, format_pct, macro_f1
from src.utils.seeds import set_seed
from src.utils.summary import count_parameters, mean_std


def _uses_graph_encoder(cfg) -> bool:
    return str(cfg.model.name).lower() != "mlp"


@torch.no_grad()
def _evaluate(model, classifier, data: MAGData, device: torch.device, uses_graph: bool) -> dict[str, float]:
    model.eval()
    classifier.eval()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device) if uses_graph else None
    y = data.y.to(device)
    z, _, _, _, _ = model(x, edge_index)
    logits = classifier(z)

    results: dict[str, float] = {}
    for split_name, idx in {
        "val": data.val_idx,
        "test": data.test_idx,
    }.items():
        split_idx = idx.to(device)
        split_logits = logits[split_idx]
        split_y = y[split_idx]
        results[f"{split_name}_acc"] = accuracy(split_logits, split_y)
        results[f"{split_name}_macro_f1"] = macro_f1(split_logits, split_y)
    return results


def _run_single_nc(cfg, data: MAGData, device: torch.device, logger: logging.Logger, run_id: int) -> dict[str, float]:
    seed = int(cfg.seed) + run_id
    set_seed(seed)

    data_info = {"input_dim": data.input_dim, "num_nodes": data.num_nodes, "num_classes": data.num_classes}
    model = build_model(cfg, data_info).to(device)
    classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(classifier.parameters()),
        lr=float(cfg.task.lr),
        weight_decay=float(cfg.task.weight_decay),
    )
    criterion = nn.CrossEntropyLoss()
    uses_graph = _uses_graph_encoder(cfg)

    if uses_graph:
        pyg_data = Data(x=data.x, edge_index=data.edge_index, y=data.y)
        loader = NeighborLoader(
            pyg_data,
            input_nodes=data.train_idx,
            num_neighbors=[int(v) for v in cfg.task.num_neighbors],
            batch_size=int(cfg.task.batch_size),
            shuffle=True,
        )
        loader_name = "NeighborLoader"
    else:
        loader = TorchDataLoader(
            data.train_idx,
            batch_size=int(cfg.task.batch_size),
            shuffle=True,
        )
        loader_name = "NodeDataLoader"
    x_all = data.x.to(device) if not uses_graph else None
    y_all = data.y.to(device) if not uses_graph else None

    logger.info(
        "[Run %d/%d] seed=%d | model params=%d",
        run_id + 1,
        int(cfg.num_runs),
        seed,
        count_parameters(model) + count_parameters(classifier),
    )
    logger.info("Loader: %s", loader_name)
    logger.info("Training...")

    best_val = -1.0
    best_test: dict[str, float] = {}
    best_model_state = None
    best_head_state = None
    patience_total = int(cfg.task.patience)
    patience_left = patience_total
    max_train_batches = cfg.task.get("max_train_batches")

    for epoch in range(1, int(cfg.task.epochs) + 1):
        model.train()
        classifier.train()
        total_loss = 0.0
        total_examples = 0
        for step, batch in enumerate(loader):
            if max_train_batches is not None and step >= int(max_train_batches):
                break
            optimizer.zero_grad(set_to_none=True)
            if uses_graph:
                batch = batch.to(device)
                z, _, _, aux_loss, _ = model(batch.x, batch.edge_index)
                logits = classifier(z[: batch.batch_size])
                labels = batch.y[: batch.batch_size]
            else:
                batch_idx = batch.to(device)
                x_batch = x_all[batch_idx]
                labels = y_all[batch_idx]
                z, _, _, aux_loss, _ = model(x_batch, None)
                logits = classifier(z)
            loss = criterion(logits, labels) + float(cfg.task.loss.aux_weight) * aux_loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(labels.numel())
            total_examples += int(labels.numel())

        train_loss = total_loss / max(total_examples, 1)
        if epoch % int(cfg.task.eval_every) != 0:
            logger.info("Epoch %05d | Train Loss %.4f", epoch, train_loss)
            continue

        metrics = _evaluate(model, classifier, data, device, uses_graph)
        logger.info("Epoch %05d | Train Loss %.4f", epoch, train_loss)
        logger.info(
            "Val Acc %.2f | Val F1 %.2f",
            format_pct(metrics["val_acc"]),
            format_pct(metrics["val_macro_f1"]),
        )
        logger.info("Test at Current")
        logger.info(
            "Test Acc %.2f | Test F1 %.2f",
            format_pct(metrics["test_acc"]),
            format_pct(metrics["test_macro_f1"]),
        )

        if metrics["val_acc"] > best_val:
            best_val = metrics["val_acc"]
            best_test = {
                "test_acc": metrics["test_acc"],
                "test_macro_f1": metrics["test_macro_f1"],
                "val_acc": metrics["val_acc"],
                "val_macro_f1": metrics["val_macro_f1"],
            }
            best_model_state = clone_state_dict(model)
            best_head_state = clone_state_dict(classifier)
            patience_left = patience_total
        else:
            patience_left -= 1
            patience_used = patience_total - patience_left
            logger.info("Patience %d/%d | Best Val Acc %.2f", patience_used, patience_total, format_pct(best_val))
            if patience_left <= 0:
                logger.info("Early stopping at epoch %03d", epoch)
                break

    if not best_test:
        metrics = _evaluate(model, classifier, data, device, uses_graph)
        best_test = {
            "test_acc": metrics["test_acc"],
            "test_macro_f1": metrics["test_macro_f1"],
            "val_acc": metrics["val_acc"],
            "val_macro_f1": metrics["val_macro_f1"],
        }

    if best_model_state is not None and best_head_state is not None:
        load_state_dict_cpu(model, best_model_state)
        load_state_dict_cpu(classifier, best_head_state)

    logger.info(
        "[Run %d] Best Val Acc %.2f",
        run_id + 1,
        format_pct(best_test["val_acc"]),
    )
    logger.info(
        "[Run %d] Final Test Acc %.2f | Final Test F1 %.2f",
        run_id + 1,
        format_pct(best_test["test_acc"]),
        format_pct(best_test["test_macro_f1"]),
    )
    return best_test


def run_nc(cfg, data: MAGData, device: torch.device, logger: logging.Logger) -> dict[str, tuple[float, float]]:
    if data.y is None or data.train_idx is None or data.val_idx is None or data.test_idx is None:
        raise ValueError("NC data must contain y/train_idx/val_idx/test_idx")

    run_results = [_run_single_nc(cfg, data, device, logger, run_id) for run_id in range(int(cfg.num_runs))]
    test_acc = [item["test_acc"] for item in run_results]
    test_f1 = [item["test_macro_f1"] for item in run_results]
    val_acc = [item["val_acc"] for item in run_results]
    val_mean, val_std = mean_std(val_acc)
    acc_mean, acc_std = mean_std(test_acc)
    f1_mean, f1_std = mean_std(test_f1)
    logger.info("============================================================")
    logger.info("Final Results over %d runs", int(cfg.num_runs))
    logger.info("============================================================")
    logger.info("Highest Valid Acc: %.2f ± %.2f", format_pct(val_mean), format_pct(val_std))
    logger.info("Test Acc: %.2f ± %.2f", format_pct(acc_mean), format_pct(acc_std))
    logger.info("Test Macro-F1: %.2f ± %.2f", format_pct(f1_mean), format_pct(f1_std))
    logger.info("============================================================")
    return {
        "val_acc": (val_mean, val_std),
        "test_acc": (acc_mean, acc_std),
        "test_macro_f1": (f1_mean, f1_std),
    }
