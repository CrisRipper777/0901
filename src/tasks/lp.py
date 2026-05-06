from __future__ import annotations

import logging

import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader

from src.data import EdgeSplit, MAGData
from src.data.graph_utils import edge_dict_to_index
from src.models import LinkPredictor, build_model
from src.tasks.common import clone_state_dict, load_state_dict_cpu
from src.utils.metrics import format_pct
from src.utils.seeds import set_seed
from src.utils.summary import count_parameters, mean_std


def _uses_graph_encoder(cfg) -> bool:
    return str(cfg.model.name).lower() != "mlp"


def _edge_keys(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    return edge_index[0].long() * int(num_nodes) + edge_index[1].long()


def _all_positive_edge_index(edge_split: EdgeSplit) -> torch.Tensor:
    return torch.cat(
        [
            edge_dict_to_index(edge_split.train),
            edge_dict_to_index(edge_split.valid),
            edge_dict_to_index(edge_split.test),
        ],
        dim=1,
    ).cpu()


def _build_forbidden_edge_keys(edge_split: EdgeSplit, num_nodes: int, undirected: bool) -> torch.Tensor:
    edge_index = _all_positive_edge_index(edge_split)
    if undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return torch.unique(_edge_keys(edge_index, num_nodes).contiguous(), sorted=True)


def _is_forbidden_edge(src: torch.Tensor, dst: torch.Tensor, num_nodes: int, forbidden_keys: torch.Tensor) -> torch.Tensor:
    keys = src.long() * int(num_nodes) + dst.long()
    positions = torch.searchsorted(forbidden_keys, keys)
    in_bounds = positions < forbidden_keys.numel()
    matches = torch.zeros_like(in_bounds, dtype=torch.bool)
    if bool(in_bounds.any()):
        matches[in_bounds] = forbidden_keys[positions[in_bounds]] == keys[in_bounds]
    return matches


def _sample_filtered_negative_targets(
    src: torch.Tensor,
    num_nodes: int,
    num_neg: int,
    forbidden_keys: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    if num_neg <= 0:
        raise ValueError(f"num_neg must be positive, got {num_neg}")

    src = src.cpu().long().contiguous()
    forbidden_keys = forbidden_keys.cpu().long().contiguous()
    negatives = torch.empty((src.numel(), num_neg), dtype=torch.long)
    row_index = torch.arange(src.numel(), dtype=torch.long)

    for neg_col in range(num_neg):
        pending = row_index
        attempts = 0
        while pending.numel() > 0:
            attempts += 1
            if attempts > 1000:
                raise RuntimeError(
                    "Unable to sample filtered train negatives after 1000 attempts; "
                    "the graph may be too dense for the requested num_train_neg."
                )
            pending_src = src[pending]
            candidate = torch.randint(0, num_nodes, (pending.numel(),), generator=generator)
            ok = candidate != pending_src
            ok &= ~_is_forbidden_edge(pending_src, candidate, num_nodes, forbidden_keys)
            if neg_col > 0:
                ok &= ~(negatives[pending, :neg_col] == candidate.view(-1, 1)).any(dim=1)

            accepted = pending[ok]
            if accepted.numel() > 0:
                negatives[accepted, neg_col] = candidate[ok]
            pending = pending[~ok]

    return negatives


def _build_epoch_train_labels(
    edge_split: EdgeSplit,
    num_nodes: int,
    num_neg: int,
    forbidden_keys: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    pos_edge_index = edge_dict_to_index(edge_split.train).cpu()
    pos_src = pos_edge_index[0]
    neg_dst = _sample_filtered_negative_targets(pos_src, num_nodes, num_neg, forbidden_keys, generator)
    neg_edge_index = torch.stack(
        [pos_src.repeat_interleave(num_neg), neg_dst.reshape(-1)],
        dim=0,
    )
    edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1).contiguous()
    edge_label = torch.cat(
        [
            torch.ones(pos_edge_index.size(1), dtype=torch.float32),
            torch.zeros(neg_edge_index.size(1), dtype=torch.float32),
        ],
        dim=0,
    ).contiguous()
    return edge_label_index, edge_label


def _exclude_positive_label_edges_from_message_graph(
    edge_index: torch.Tensor,
    edge_label_index: torch.Tensor,
    edge_label: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    positive_mask = edge_label > 0.5
    if not bool(positive_mask.any()):
        return edge_index.contiguous()

    positive_edges = edge_label_index[:, positive_mask].long()
    forbidden_edges = torch.cat([positive_edges, positive_edges.flip(0)], dim=1)
    forbidden_keys = torch.unique(_edge_keys(forbidden_edges, num_nodes), sorted=True)
    edge_keys = _edge_keys(edge_index, num_nodes)
    positions = torch.searchsorted(forbidden_keys, edge_keys)
    in_bounds = positions < forbidden_keys.numel()
    drop_mask = torch.zeros(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
    if bool(in_bounds.any()):
        drop_mask[in_bounds] = forbidden_keys[positions[in_bounds]] == edge_keys[in_bounds]
    return edge_index[:, ~drop_mask].contiguous()


def _build_link_loader(
    cfg,
    data: MAGData,
    edge_label_index: torch.Tensor,
    edge_label: torch.Tensor,
) -> LinkNeighborLoader:
    pyg_data = Data(x=data.x, edge_index=data.edge_index)
    return LinkNeighborLoader(
        pyg_data,
        num_neighbors=[int(v) for v in cfg.task.num_neighbors],
        batch_size=int(cfg.task.batch_size),
        shuffle=True,
        edge_label_index=edge_label_index,
        edge_label=edge_label,
    )


def _build_edge_loader(cfg, edge_label_index: torch.Tensor, edge_label: torch.Tensor) -> TorchDataLoader:
    return TorchDataLoader(
        TensorDataset(edge_label_index.t().contiguous(), edge_label.contiguous()),
        batch_size=int(cfg.task.batch_size),
        shuffle=True,
    )


@torch.no_grad()
def _infer_all(model, data: MAGData, device: torch.device, uses_graph: bool, batch_size: int) -> torch.Tensor:
    model.eval()
    edge_index = data.edge_index if uses_graph else None
    return model.inference(data.x, edge_index, device=device, batch_size=batch_size)


def _prepare_eval_embeddings(
    z: torch.Tensor,
    device: torch.device,
    preload: bool,
    logger: logging.Logger,
) -> torch.Tensor:
    z_eval = z.to(device, non_blocking=True) if preload else z
    logger.info("Eval node embeddings: preload=%s | device=%s | shape=%s", preload, z_eval.device, tuple(z_eval.shape))
    return z_eval


@torch.no_grad()
def _evaluate_split(
    z: torch.Tensor,
    predictor: LinkPredictor,
    split: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    predictor.eval()
    src_all = split["source_node"]
    dst_all = split["target_node"]
    neg_all = split["target_node_neg"]
    total = int(src_all.numel())
    mrr_sum = 0.0
    hits1_sum = 0.0
    hits3_sum = 0.0
    hits10_sum = 0.0

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        if z.device.type == "cuda":
            src = src_all[start:end].to(z.device, non_blocking=True).long()
            dst = dst_all[start:end].to(z.device, non_blocking=True).long()
            neg = neg_all[start:end].to(z.device, non_blocking=True).long()
            pos_src = z[src]
            pos_dst = z[dst]
            src_neg = src.view(-1, 1).expand_as(neg).reshape(-1)
            neg_src = z[src_neg]
            neg_dst = z[neg.reshape(-1)]
        else:
            src = src_all[start:end].cpu().long()
            dst = dst_all[start:end].cpu().long()
            neg = neg_all[start:end].cpu().long()
            pos_src = z[src].to(device, non_blocking=True)
            pos_dst = z[dst].to(device, non_blocking=True)
            src_neg = src.view(-1, 1).expand_as(neg).reshape(-1)
            neg_src = z[src_neg].to(device, non_blocking=True)
            neg_dst = z[neg.reshape(-1)].to(device, non_blocking=True)

        pos_score = predictor.score_pairs(pos_src, pos_dst)
        neg_score = predictor.score_pairs(neg_src, neg_dst).view(neg.size(0), neg.size(1))
        greater = (neg_score > pos_score.view(-1, 1)).sum(dim=1).float()
        equal = (neg_score == pos_score.view(-1, 1)).sum(dim=1).float()
        ranks = 1.0 + greater + 0.5 * equal

        mrr_sum += float((1.0 / ranks).sum().item())
        hits1_sum += float((ranks <= 1).float().sum().item())
        hits3_sum += float((ranks <= 3).float().sum().item())
        hits10_sum += float((ranks <= 10).float().sum().item())

    denom = max(total, 1)
    return {
        "mrr": mrr_sum / denom,
        "hits@1": hits1_sum / denom,
        "hits@3": hits3_sum / denom,
        "hits@10": hits10_sum / denom,
    }


def _run_single_lp(cfg, data: MAGData, device: torch.device, logger: logging.Logger, run_id: int) -> dict[str, float]:
    seed = int(cfg.seed) + run_id
    set_seed(seed)
    data_info = {"input_dim": data.input_dim, "num_nodes": data.num_nodes, "num_classes": data.num_classes}
    model = build_model(cfg, data_info).to(device)
    predictor = LinkPredictor(
        in_dim=model.out_dim,
        hidden_dim=int(cfg.task.decoder.hidden_dim),
        num_layers=int(cfg.task.decoder.num_layers),
        dropout=float(cfg.task.decoder.dropout),
    ).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=float(cfg.task.lr),
        weight_decay=float(cfg.task.weight_decay),
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    uses_graph = _uses_graph_encoder(cfg)
    loader_name = "LinkNeighborLoader" if uses_graph else "EdgeLabelDataLoader"
    x_all = data.x.to(device) if not uses_graph else None
    undirected_filter = bool(data.edge_split.metadata.get("undirected", cfg.dataset.get("make_undirected", True)))
    forbidden_keys = _build_forbidden_edge_keys(data.edge_split, data.num_nodes, undirected=undirected_filter)
    neg_generator = torch.Generator().manual_seed(seed)

    logger.info(
        "[Run %d/%d] seed=%d | model+decoder params=%d",
        run_id + 1,
        int(cfg.num_runs),
        seed,
        count_parameters(model) + count_parameters(predictor),
    )
    logger.info("Loader: %s", loader_name)
    logger.info("Train negative sampling: global filtered | num_neg=%d", int(cfg.task.num_train_neg))
    logger.info("Training...")

    best_val = -1.0
    best_test: dict[str, float] = {}
    best_model_state = None
    best_predictor_state = None
    patience_total = int(cfg.task.patience)
    patience_left = patience_total
    max_train_batches = cfg.task.get("max_train_batches")
    inference_batch_size = int(cfg.task.inference_batch_size)
    eval_preload_node_emb = bool(cfg.task.get("eval_preload_node_emb", False))
    logger.info("LP eval preload node embeddings: %s", eval_preload_node_emb)

    for epoch in range(1, int(cfg.task.epochs) + 1):
        model.train()
        predictor.train()
        total_loss = 0.0
        total_examples = 0
        edge_label_index, edge_label = _build_epoch_train_labels(
            data.edge_split,
            data.num_nodes,
            int(cfg.task.num_train_neg),
            forbidden_keys,
            neg_generator,
        )
        if uses_graph:
            loader = _build_link_loader(cfg, data, edge_label_index, edge_label)
        else:
            loader = _build_edge_loader(cfg, edge_label_index, edge_label)

        for step, batch in enumerate(loader):
            if max_train_batches is not None and step >= int(max_train_batches):
                break
            optimizer.zero_grad(set_to_none=True)
            if uses_graph:
                batch = batch.to(device)
                message_edge_index = _exclude_positive_label_edges_from_message_graph(
                    batch.edge_index,
                    batch.edge_label_index,
                    batch.edge_label,
                    num_nodes=int(batch.x.size(0)),
                )
                z, _, _, aux_loss, _ = model(batch.x, message_edge_index)
                src, dst = batch.edge_label_index
                logits = predictor.score_pairs(z[src], z[dst])
                labels = batch.edge_label.float()
            else:
                edges, labels = batch
                edges = edges.to(device)
                labels = labels.to(device)
                z_all, _, _, aux_loss, _ = model(x_all[edges.reshape(-1)], None)
                z_pairs = z_all.view(edges.size(0), 2, -1)
                logits = predictor.score_pairs(z_pairs[:, 0], z_pairs[:, 1])
            loss = criterion(logits, labels) + float(cfg.task.loss.aux_weight) * aux_loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(labels.numel())
            total_examples += int(labels.numel())

        train_loss = total_loss / max(total_examples, 1)
        if epoch % int(cfg.task.eval_every) != 0:
            logger.info("Epoch %05d | Train Loss %.4f", epoch, train_loss)
            continue

        z = _infer_all(model, data, device, uses_graph, inference_batch_size)
        z_eval = _prepare_eval_embeddings(z, device, eval_preload_node_emb, logger)
        val_metrics = _evaluate_split(z_eval, predictor, data.edge_split.valid, device, int(cfg.task.eval_edge_batch_size))
        logger.info("Epoch %05d | Train Loss %.4f", epoch, train_loss)
        logger.info(
            "Val MRR %.2f | Val H@1 %.2f | Val H@3 %.2f | Val H@10 %.2f",
            format_pct(val_metrics["mrr"]),
            format_pct(val_metrics["hits@1"]),
            format_pct(val_metrics["hits@3"]),
            format_pct(val_metrics["hits@10"]),
        )

        stop_early = False
        if val_metrics["mrr"] > best_val:
            best_val = val_metrics["mrr"]
            best_test = {
                "val_mrr": val_metrics["mrr"],
            }
            best_model_state = clone_state_dict(model)
            best_predictor_state = clone_state_dict(predictor)
            patience_left = patience_total
        else:
            patience_left -= 1
            patience_used = patience_total - patience_left
            logger.info("Patience %d/%d | Best Val MRR %.2f", patience_used, patience_total, format_pct(best_val))
            if patience_left <= 0:
                logger.info("Early stopping at epoch %03d", epoch)
                stop_early = True

        del z_eval
        del z
        if stop_early:
            break

    if best_model_state is not None and best_predictor_state is not None:
        load_state_dict_cpu(model, best_model_state)
        load_state_dict_cpu(predictor, best_predictor_state)
        z = _infer_all(model, data, device, uses_graph, inference_batch_size)
        z_eval = _prepare_eval_embeddings(z, device, eval_preload_node_emb, logger)
        test_metrics = _evaluate_split(
            z_eval,
            predictor,
            data.edge_split.test,
            device,
            int(cfg.task.eval_edge_batch_size),
        )
        best_test["test_mrr"] = test_metrics["mrr"]
        best_test["test_hits@1"] = test_metrics["hits@1"]
        best_test["test_hits@3"] = test_metrics["hits@3"]
        best_test["test_hits@10"] = test_metrics["hits@10"]

    if not best_test:
        z = _infer_all(model, data, device, uses_graph, inference_batch_size)
        z_eval = _prepare_eval_embeddings(z, device, eval_preload_node_emb, logger)
        val_metrics = _evaluate_split(z_eval, predictor, data.edge_split.valid, device, int(cfg.task.eval_edge_batch_size))
        test_metrics = _evaluate_split(z_eval, predictor, data.edge_split.test, device, int(cfg.task.eval_edge_batch_size))
        best_test = {
            "val_mrr": val_metrics["mrr"],
            "test_mrr": test_metrics["mrr"],
            "test_hits@1": test_metrics["hits@1"],
            "test_hits@3": test_metrics["hits@3"],
            "test_hits@10": test_metrics["hits@10"],
        }

    logger.info(
        "[Run %d] Best Val MRR %.2f",
        run_id + 1,
        format_pct(best_test["val_mrr"]),
    )
    logger.info(
        "[Run %d] Final Test MRR %.2f | H@1 %.2f | H@3 %.2f | H@10 %.2f",
        run_id + 1,
        format_pct(best_test["test_mrr"]),
        format_pct(best_test["test_hits@1"]),
        format_pct(best_test["test_hits@3"]),
        format_pct(best_test["test_hits@10"]),
    )
    return best_test


def run_lp(cfg, data: MAGData, device: torch.device, logger: logging.Logger) -> dict[str, tuple[float, float]]:
    if data.edge_split is None:
        raise ValueError("LP data must contain edge_split")
    run_results = [_run_single_lp(cfg, data, device, logger, run_id) for run_id in range(int(cfg.num_runs))]
    output: dict[str, tuple[float, float]] = {}
    logger.info("============================================================")
    logger.info("Final Results over %d runs", int(cfg.num_runs))
    logger.info("============================================================")
    display_names = {
        "val_mrr": "Highest Valid MRR",
        "test_mrr": "Test MRR",
        "test_hits@1": "Test Hits@1",
        "test_hits@3": "Test Hits@3",
        "test_hits@10": "Test Hits@10",
    }
    for key in ["val_mrr", "test_mrr", "test_hits@1", "test_hits@3", "test_hits@10"]:
        values = [item[key] for item in run_results]
        mean, std = mean_std(values)
        output[key] = (mean, std)
        logger.info("%s: %.2f ± %.2f", display_names[key], format_pct(mean), format_pct(std))
    logger.info("============================================================")
    return output
