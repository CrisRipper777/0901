"""LGMRec (Local-Global Multimodal Recommender, RecSys 2024) — ported from
the OpenMAG implementation (OpenMAG/src/model/models.py) for the unified NC
benchmark protocol.

Architecture (faithful to OpenMAG):
    feat_encoder -> 3x LGConv (LightGCN, self-loops added) -> mean pool
      -> local_structure_emb + alpha * L2-normalize(hypergraph emb)
    hypergraph: HGNNLayer = linear hyperedge projector -> gumbel-softmax H
      -> node->hyperedge->node propagation (n_layers=1)
    vision/text heads + decoders -> InfoNCE reconstruction of the frozen
    raw modality features (OpenMAG trainer: lambda_v = lambda_t = 0.5,
    tau = 0.07, batch-wise InfoNCE).

Adaptation to this repo's frozen full-graph NC protocol:
    - full-graph forward, CE on train nodes (task loss unchanged);
    - InfoNCE computed on a random subset of nodes per forward (OpenMAG
      batches train subgraphs; here we sample ``nce_batch_size`` nodes of
      the full graph — the loss is feature-reconstruction only, no labels);
    - eval: deterministic softmax(hyper_score / tau) instead of gumbel
      noise (approximates the training-time expected H);
    - lr/wd preset 5e-3 / 1e-5 (OpenMAG NC task config); no label smoothing
      (unified protocol uses plain CE).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LGConv
from torch_geometric.utils import add_self_loops

from .common import get_activation


class HGNNLayer(nn.Module):
    """Hypergraph layer (OpenMAG): hyperedge assignment H = gumbel_softmax(
    Linear(x), tau), then one node -> hyperedge -> node propagation round."""

    def __init__(self, in_dim: int, hyper_num: int, tau: float = 0.5) -> None:
        super().__init__()
        self.hyper_num = int(hyper_num)
        self.tau = float(tau)
        self.hyper_projector = nn.Linear(int(in_dim), self.hyper_num, bias=False)
        nn.init.xavier_uniform_(self.hyper_projector.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hyper_score = self.hyper_projector(x)
        if self.training:
            h = F.gumbel_softmax(hyper_score, tau=self.tau, dim=1, hard=False)
        else:
            # deterministic approximation of the expected training-time H
            h = F.softmax(hyper_score / self.tau, dim=1)
        lat = torch.mm(h.t(), x)  # hyperedge aggregation
        return torch.mm(h, lat)  # hyperedge -> node distribution


def _infonce(out: torch.Tensor, orig: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """Batch-wise InfoNCE (OpenMAG gnn_trainer.infoNCE_loss): CE over the
    normalized similarity matrix with diagonal positives."""
    out_norm = F.normalize(out, p=2, dim=1)
    orig_norm = F.normalize(orig, p=2, dim=1)
    sim = torch.mm(out_norm, orig_norm.t()) / tau
    labels = torch.arange(sim.size(0), device=sim.device)
    return F.cross_entropy(sim, labels)


class Model(nn.Module):
    """LGMRec NC encoder. Repo interface: forward -> (z, None, None,
    aux_loss, aux_info); inference -> CPU z; ``out_dim`` = hidden_dim."""

    def __init__(self, cfg, data_info):
        super().__init__()
        hidden_dim = int(cfg.model.hidden_dim)
        num_layers = int(cfg.model.num_layers)
        dropout = float(cfg.model.dropout)
        hyper_num = int(cfg.model.hyper_num)
        alpha = float(cfg.model.alpha)
        self.nce_batch_size = int(cfg.model.get("nce_batch_size", 1024))
        self.nce_tau = float(cfg.model.get("nce_tau", 0.07))
        self.lambda_v = float(cfg.model.get("lambda_v", 0.5))
        self.lambda_t = float(cfg.model.get("lambda_t", 0.5))
        self.text_dim = int(data_info["text_dim"])
        self.visual_dim = int(data_info["visual_dim"])

        self.feat_encoder = nn.Linear(int(data_info["input_dim"]), hidden_dim)
        self.convs = nn.ModuleList([LGConv() for _ in range(num_layers)])
        self.hgnn = HGNNLayer(hidden_dim, hyper_num=hyper_num)
        self.vision_head = nn.Linear(hidden_dim, hidden_dim)
        self.text_head = nn.Linear(hidden_dim, hidden_dim)
        self.decoder_v = nn.Linear(hidden_dim, self.visual_dim)
        self.decoder_t = nn.Linear(hidden_dim, self.text_dim)
        self.dropout = dropout
        self.alpha = alpha
        self.out_dim = hidden_dim

    def _encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x_emb = F.dropout(self.feat_encoder(x), p=self.dropout, training=self.training)
        embs = [x_emb]
        current = x_emb
        for conv in self.convs:
            current = conv(current, edge_index)
            embs.append(current)
        local = torch.stack(embs, dim=1).mean(dim=1)
        global_hyper = self.hgnn(x_emb)
        return local + self.alpha * F.normalize(global_hyper, p=2, dim=1)

    def _compute_aux(self, x: torch.Tensor, final: torch.Tensor) -> torch.Tensor:
        """OpenMAG-style modality InfoNCE on a random node subset."""
        n = int(x.size(0))
        size = min(self.nce_batch_size, n)
        idx = torch.randperm(n, device=x.device)[:size]
        x_v = F.dropout(F.relu(self.vision_head(final)), p=self.dropout, training=self.training)
        x_t = F.dropout(F.relu(self.text_head(final)), p=self.dropout, training=self.training)
        loss_v = _infonce(self.decoder_v(x_v[idx]), x[idx, self.text_dim:], tau=self.nce_tau)
        loss_t = _infonce(self.decoder_t(x_t[idx]), x[idx, : self.text_dim], tau=self.nce_tau)
        return self.lambda_v * loss_v + self.lambda_t * loss_t

    def forward(self, x: torch.Tensor, edge_index=None):
        if edge_index is None:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=x.device)
        # LightGCN convention: normalized adjacency includes self-loops
        # (OpenMAG task config self_loop=True; dataset configs here set
        # add_self_loops=false, models add their own).
        edge_index, _ = add_self_loops(edge_index, num_nodes=int(x.size(0)))
        final = self._encode(x, edge_index)
        aux_loss = self._compute_aux(x, final) if self.training else x.new_tensor(0.0)
        aux_info = {"lgmrec_infonce": float(aux_loss.item()) if self.training else 0.0}
        return final, None, None, aux_loss, aux_info

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        device: torch.device | None = None,
        batch_size: int = 65536,
    ) -> torch.Tensor:
        """One exact full-graph forward (LGConv needs full neighborhoods)."""
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        x = x.to(device)
        if edge_index is None:
            edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
        else:
            edge_index = edge_index.to(device)
        z, _, _, _, _ = self.forward(x, edge_index)
        return z.detach().cpu()
