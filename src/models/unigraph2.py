from __future__ import annotations

from collections import deque
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GATConv


class MoE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_experts: int,
        num_selected_experts: int = 2,
    ) -> None:
        super().__init__()
        if num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {num_experts}")
        if num_selected_experts < 1 or num_selected_experts > num_experts:
            raise ValueError(
                f"num_selected_experts must be in [1, {num_experts}], got {num_selected_experts}"
            )

        self.num_selected_experts = num_selected_experts
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_experts)
            ]
        )
        self.gate = nn.Sequential(nn.Linear(input_dim, num_experts), nn.Softmax(dim=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.gate(x)
        top_weights, top_indices = torch.topk(weights, self.num_selected_experts, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        selected_outputs = torch.gather(
            expert_outputs,
            dim=1,
            index=top_indices.unsqueeze(-1).expand(-1, -1, expert_outputs.size(-1)),
        )
        return torch.sum(selected_outputs * top_weights.unsqueeze(-1), dim=1)


class DomainSpecificDecoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class SPDDecoder(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, 1),
        )

    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([x_i, x_j], dim=-1)).squeeze(-1)


class Model(nn.Module):
    def __init__(self, cfg, data_info: dict):
        super().__init__()
        input_dim = int(data_info["input_dim"])
        hidden_dim = int(cfg.model.hidden_dim)
        num_layers = int(cfg.model.get("num_layers", 3))
        num_heads = int(cfg.model.get("num_heads", 4))
        num_experts = int(cfg.model.get("num_experts", 8))
        num_selected_experts = int(cfg.model.get("num_selected_experts", 2))
        dropout = float(cfg.model.get("dropout", 0.0))

        self.text_dim = int(data_info.get("text_dim", 0) or 0)
        self.visual_dim = int(data_info.get("visual_dim", 0) or 0)
        if self.text_dim <= 0 or self.visual_dim <= 0:
            self.text_dim = input_dim // 2
            self.visual_dim = input_dim - self.text_dim
        if self.text_dim + self.visual_dim > input_dim:
            raise ValueError(
                f"text_dim+visual_dim={self.text_dim + self.visual_dim} exceeds input_dim={input_dim}"
            )

        self.hidden_dim = hidden_dim
        self.out_dim = hidden_dim
        self.feat_drop_rate = float(cfg.model.get("feat_drop_rate", 0.1))
        self.edge_mask_rate = float(cfg.model.get("edge_mask_rate", 0.0))
        self.gamma = float(cfg.model.get("gamma", 2.0))
        self.lambda_spd = float(cfg.model.get("lambda_spd", 0.5))
        self.use_reconstruction_loss = bool(cfg.model.get("use_reconstruction_loss", True))
        self.use_spd_loss = bool(cfg.model.get("use_spd_loss", True))
        self.spd_k = int(cfg.model.get("spd_k", 3))
        self.spd_num_sources = int(cfg.model.get("spd_num_sources", 32))
        self.spd_max_pairs = int(cfg.model.get("spd_max_pairs", 4096))
        self.spd_undirected = bool(cfg.model.get("spd_undirected", True))
        self.include_self_spd = bool(cfg.model.get("include_self_spd", False))

        self.projectors = nn.ModuleDict(
            {
                "text": nn.Linear(self.text_dim, hidden_dim),
                "image": nn.Linear(self.visual_dim, hidden_dim),
            }
        )
        self.moe = MoE(hidden_dim, hidden_dim, num_experts, num_selected_experts)
        self.gnn_layers = nn.ModuleList(
            [
                GATConv(
                    hidden_dim,
                    hidden_dim,
                    heads=num_heads,
                    concat=False,
                    dropout=dropout,
                    add_self_loops=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(max(num_layers - 1, 0))]
        )
        if self.use_reconstruction_loss:
            self.domain_decoders = nn.ModuleDict(
                {
                    "text": DomainSpecificDecoder(hidden_dim, self.text_dim),
                    "image": DomainSpecificDecoder(hidden_dim, self.visual_dim),
                }
            )
        else:
            self.domain_decoders = nn.ModuleDict()

        if self.use_spd_loss and self.lambda_spd > 0.0:
            self.spd_decoder = SPDDecoder(hidden_dim)

        if self.feat_drop_rate > 0.0:
            self.mask_token = nn.Parameter(torch.randn(hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def _split_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        text_feat = x[:, : self.text_dim]
        visual_feat = x[:, self.text_dim : self.text_dim + self.visual_dim]
        return {"text": text_feat, "image": visual_feat}

    def _fuse_features(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        projected = [F.normalize(self.projectors[name](feat), dim=-1) for name, feat in features.items()]
        return torch.stack(projected, dim=0).mean(dim=0)

    def _mask_features(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        num_nodes = int(features.size(0))
        num_masked = int(num_nodes * self.feat_drop_rate)
        mask = torch.zeros(num_nodes, dtype=torch.bool, device=features.device)
        if num_masked <= 0:
            return features, mask

        perm = torch.randperm(num_nodes, device=features.device)
        mask[perm[:num_masked]] = True
        masked_features = features.clone()
        masked_features[mask] = self.mask_token.to(dtype=features.dtype)
        return masked_features, mask

    def _mask_edges(self, edge_index: torch.Tensor) -> torch.Tensor:
        if not self.training or self.edge_mask_rate <= 0.0:
            return edge_index
        num_edges = int(edge_index.size(1))
        if num_edges <= 1:
            return edge_index
        keep = torch.rand(num_edges, device=edge_index.device) >= self.edge_mask_rate
        if not bool(keep.any()):
            return edge_index
        return edge_index[:, keep]

    def _encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._split_features(x)
        fused = self._fuse_features(features)
        if self.training:
            h, mask = self._mask_features(fused)
        else:
            h = fused
            mask = torch.zeros(int(x.size(0)), dtype=torch.bool, device=x.device)

        h = self.moe(h)
        for layer_idx, layer in enumerate(self.gnn_layers):
            h = layer(h, edge_index)
            if layer_idx != len(self.gnn_layers) - 1:
                h = self.norms[layer_idx](h)
                h = self.dropout(h)
        return h, mask

    def _compute_reconstruction_loss(
        self,
        embeddings: torch.Tensor,
        original_features: dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_reconstruction_loss or not bool(mask.any()):
            return embeddings.new_tensor(0.0)

        loss = embeddings.new_tensor(0.0)
        for name, decoder in self.domain_decoders.items():
            reconstructed = decoder(embeddings[mask])
            original = original_features[name][mask]
            similarity = F.cosine_similarity(reconstructed, original, dim=-1)
            loss = loss + (1.0 - similarity).clamp_min(0.0).pow(self.gamma).mean()
        return loss

    def _build_adjacency(self, edge_index: torch.Tensor, num_nodes: int) -> list[list[int]]:
        adjacency: list[set[int]] = [set() for _ in range(num_nodes)]
        edge_cpu = edge_index.detach().cpu()
        for src, dst in zip(edge_cpu[0].tolist(), edge_cpu[1].tolist()):
            if src < 0 or dst < 0 or src >= num_nodes or dst >= num_nodes:
                continue
            adjacency[src].add(dst)
            if self.spd_undirected:
                adjacency[dst].add(src)
        return [list(neighbors) for neighbors in adjacency]

    def _iter_shortest_paths(
        self,
        adjacency: list[list[int]],
        sources: Iterable[int],
    ) -> tuple[list[int], list[int], list[float]]:
        rows: list[int] = []
        cols: list[int] = []
        dists: list[float] = []

        for source in sources:
            visited = {source: 0}
            queue: deque[tuple[int, int]] = deque([(source, 0)])
            while queue:
                node, dist = queue.popleft()
                if dist >= self.spd_k:
                    continue
                next_dist = dist + 1
                for neighbor in adjacency[node]:
                    if neighbor in visited:
                        continue
                    visited[neighbor] = next_dist
                    queue.append((neighbor, next_dist))

            for target, dist in visited.items():
                if target == source and not self.include_self_spd:
                    continue
                rows.append(source)
                cols.append(target)
                dists.append(float(dist))
                if len(rows) >= self.spd_max_pairs:
                    return rows, cols, dists

        return rows, cols, dists

    def _compute_spd_loss(self, embeddings: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if (
            not self.use_spd_loss
            or self.lambda_spd <= 0.0
            or self.spd_k <= 0
            or self.spd_num_sources <= 0
            or self.spd_max_pairs <= 0
            or edge_index is None
        ):
            return embeddings.new_tensor(0.0)

        num_nodes = int(embeddings.size(0))
        if num_nodes <= 1:
            return embeddings.new_tensor(0.0)

        num_sources = min(self.spd_num_sources, num_nodes)
        sources = torch.randperm(num_nodes)[:num_sources].tolist()
        adjacency = self._build_adjacency(edge_index, num_nodes)
        rows, cols, dists = self._iter_shortest_paths(adjacency, sources)
        if not rows:
            return embeddings.new_tensor(0.0)

        row_idx = torch.tensor(rows, dtype=torch.long, device=embeddings.device)
        col_idx = torch.tensor(cols, dtype=torch.long, device=embeddings.device)
        target = torch.tensor(dists, dtype=embeddings.dtype, device=embeddings.device)
        pred = self.spd_decoder(embeddings[row_idx], embeddings[col_idx])
        return F.mse_loss(pred, target)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        if edge_index is None:
            raise ValueError("UniGraph2 requires edge_index and cannot run as a feature-only model")

        features = self._split_features(x)
        message_edge_index = self._mask_edges(edge_index)
        z, mask = self._encode(x, message_edge_index)

        reconstruction_loss = z.new_tensor(0.0)
        spd_loss = z.new_tensor(0.0)
        if self.training:
            reconstruction_loss = self._compute_reconstruction_loss(z, features, mask)
            spd_loss = self._compute_spd_loss(z, edge_index)
        aux_loss = reconstruction_loss + self.lambda_spd * spd_loss
        aux_info = {
            "reconstruction_loss": reconstruction_loss.detach(),
            "spd_loss": spd_loss.detach(),
        }
        return z, None, None, aux_loss, aux_info

    @torch.no_grad()
    def _initial_embedding_inference(
        self,
        x: torch.Tensor,
        device: torch.device,
        batch_size: int,
    ) -> torch.Tensor:
        out = torch.empty((x.size(0), self.hidden_dim), dtype=x.dtype, device="cpu")
        for start in range(0, x.size(0), batch_size):
            end = min(start + batch_size, x.size(0))
            features = self._split_features(x[start:end].to(device))
            h = self._fuse_features(features)
            h = self.moe(h)
            out[start:end] = h.detach().cpu()
        return out

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        device: torch.device | None = None,
        batch_size: int = 65536,
    ) -> torch.Tensor:
        self.eval()
        if edge_index is None:
            raise ValueError("UniGraph2 requires edge_index for inference")
        if device is None:
            device = next(self.parameters()).device

        h = self._initial_embedding_inference(x.cpu(), device, batch_size)
        edge_index = edge_index.cpu()
        num_nodes = int(h.size(0))
        input_nodes = torch.arange(num_nodes, dtype=torch.long)

        for layer_idx, layer in enumerate(self.gnn_layers):
            data = Data(x=h, edge_index=edge_index)
            loader = NeighborLoader(
                data,
                input_nodes=input_nodes,
                num_neighbors=[-1],
                batch_size=batch_size,
                shuffle=False,
            )
            out = torch.empty((num_nodes, self.hidden_dim), dtype=h.dtype, device="cpu")
            for batch in loader:
                batch = batch.to(device)
                z = layer(batch.x, batch.edge_index)[: batch.batch_size]
                if layer_idx != len(self.gnn_layers) - 1:
                    z = self.norms[layer_idx](z)
                out[batch.n_id[: batch.batch_size].cpu()] = z.detach().cpu()
            h = out

        return h
