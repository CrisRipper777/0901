"""P2 Bi-Axis model: Null-Augmented Factor-Relation Transport (plan §2).

Replaces P1's independent budget beta and selector alpha with ONE unified
plan:

    Gamma_i in R_+^{F x (K+1)},   sum_k Gamma_i,f,k = 1
    col 0 = Local / No-Transport;  cols 1..K = latent structural relations

with derived diagnostics (plan §28.1):
    beta_i^f  = 1 - Gamma_i,f,0            (graph mass)
    alpha_ifk = Gamma_i,f,k / sum_l Gamma  (conditional relation plan)

Modes (same scorer / null scores / relation decomposition / W0 everywhere;
only the plan solver differs, plan §26):
    null_softmax : independent row softmax over {Local, R1..RK}
    fixed_uot    : semi-relaxed UOT, hard row marginal, soft column KL to nu
                   with global theta = tau_base / (tau_base + eps)
    adaptive_uot : theta_i = tau_i / (tau_i + eps), tau_i = tau_base * q_i^R
                   (relation specialization confidence, plan §18)

Frozen: M1 (P0 factorizer, arch/obj unchanged, jointly optimized), M2 (P1
topology-only relation decomposition, K=4), shared W0. P1's budget/selector
modules are DELETED (no unused params); P1 files are never modified.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .biaxis_p1 import Model as P1Model
from .biaxis_p1_components import relation_weighted_mean
from .biaxis_p2_components import (
    FactorRelationScore,
    build_augmented_scores,
    build_reference_capacity,
    compute_node_relation_confidence,
    null_augmented_softmax,
    semi_relaxed_transport,
)


class Model(P1Model):
    """Bi-Axis P2. Inherits P1 M1/M2/W0/fusion/aux-losses; overrides only the
    coupling (`_graph_update`) with the transport plan."""

    def __init__(self, cfg, data_info):
        super().__init__(cfg, data_info)

        assert self.factor_aware, "biaxis_p2 requires model.p1.factor_aware=true"
        assert self.num_relations == 4, "biaxis_p2 requires model.p1.num_relations=4"

        p2 = cfg.model.p2
        self.p2_mode = str(p2.mode)
        assert self.p2_mode in ("null_softmax", "fixed_uot", "adaptive_uot"), (
            f"p2.mode must be null_softmax|fixed_uot|adaptive_uot, got {self.p2_mode!r}"
        )

        # Drop the P1 gate modules (plan §24): they are replaced by the
        # transport plan; deleting avoids unused trainable params. The F0
        # projectors are unreachable (factor_aware asserted above).
        del self.graph_budget
        del self.factor_selector
        del self.proj_q
        del self.fusion_q

        activation = str(cfg.model.get("activation", "gelu"))
        self.transport_scorer = FactorRelationScore(
            self.factor_dim,
            hidden_dim=int(p2.score_hidden_dim),
            activation=activation,
        )
        # Per-factor global null thresholds (plan §7), init 0.
        self.null_score = nn.Parameter(torch.zeros(3))

        self.p2_epsilon = float(p2.epsilon)
        self.p2_tau_base = float(p2.tau_base)
        self.p2_sinkhorn_iters = int(p2.sinkhorn_iters)
        self.p2_null_prior = float(p2.null_prior)

    # ------------------------------------------------------------------
    # P2 coupling (plan §44): unified plan
    # ------------------------------------------------------------------

    def _graph_update(
        self,
        f_block: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> dict[str, torch.Tensor]:
        num_factors = int(f_block.size(1))
        factor_dim = int(f_block.size(2))
        device = f_block.device

        r, availability, deg = self._decompose_relations(edge_index, num_nodes)

        # Relation-specific factor contexts (reuse P1 aggregation, plan §5).
        f_cat = f_block.reshape(num_nodes, num_factors * factor_dim)
        g_cat, _mass = relation_weighted_mean(
            edge_index, r, f_cat, num_nodes, edge_chunk_size=self.edge_chunk_size
        )  # [N, K, F*d]
        g_perm = g_cat.reshape(num_nodes, self.num_relations, num_factors, factor_dim)
        g_perm = g_perm.permute(0, 2, 1, 3)  # [N, F, K, d]

        # --- scores (plan §6/§7/§8) ----------------------------------------
        s_rel = self.transport_scorer(f_block, g_perm)  # [N, F, K]
        s_aug = build_augmented_scores(s_rel, self.null_score)  # [N, F, K+1]

        # --- capacity reference (availability detached, plan §10) ----------
        nu = build_reference_capacity(
            availability,
            num_factors,
            null_prior=self.p2_null_prior,
            degree=deg,
        )  # [N, K+1]

        # --- relation specialization confidence (detached, plan §17) -------
        q = compute_node_relation_confidence(r, edge_index, num_nodes)  # [N]

        # --- plan solver (plan §26) ----------------------------------------
        if self.p2_mode == "null_softmax":
            gamma = null_augmented_softmax(s_aug, self.p2_epsilon)
            theta = torch.zeros(num_nodes, dtype=f_block.dtype, device=device)
        else:
            theta_override = None
            if self.p2_mode == "adaptive_uot":
                tau_i = self.p2_tau_base * q  # [N]
                theta_override = (tau_i / (tau_i + self.p2_epsilon)).unsqueeze(-1)  # [N,1]
            gamma = semi_relaxed_transport(
                s_aug,
                nu,
                self.p2_epsilon,
                self.p2_tau_base,
                self.p2_sinkhorn_iters,
                theta_override,
            )
            if self.p2_mode == "fixed_uot":
                theta = torch.full(
                    (num_nodes,),
                    self.p2_tau_base / (self.p2_tau_base + self.p2_epsilon),
                    dtype=f_block.dtype,
                    device=device,
                )
            else:
                theta = theta_override.squeeze(-1)  # [N]

        # --- isolated node fast path (plan §21): all Local, no transport ---
        isolated = deg <= 0
        if bool(isolated.any()):
            local_plan = torch.zeros_like(gamma)
            local_plan[..., 0] = 1.0
            gamma = torch.where(isolated[:, None, None], local_plan, gamma)

        # --- message passing (plan §22) ------------------------------------
        graph_mass = 1.0 - gamma[..., 0]  # [N, F]  (derived beta)
        alpha_diag = gamma[..., 1:] / (graph_mass.unsqueeze(-1) + self.eps)  # [N, F, K]
        g_mix = (gamma[..., 1:].unsqueeze(-1) * g_perm).sum(dim=2)  # [N, F, d]
        m_f = self.graph_w0(g_mix.reshape(num_nodes * num_factors, factor_dim))
        m_f = m_f.reshape(num_nodes, num_factors, factor_dim)
        # No separate beta multiplication: the plan already carries the graph
        # mass in gamma[..., 1:] (plan §22).
        f_tilde = self.graph_norm((f_block + m_f).reshape(num_nodes * num_factors, factor_dim))
        f_tilde = f_tilde.reshape(num_nodes, num_factors, factor_dim)

        return {
            "f_tilde": f_tilde,
            "beta": graph_mass,
            "alpha": alpha_diag,
            "r": r,
            "availability": availability,
            "gamma": gamma,
            "null_mass": gamma[..., 0],
            "relation_confidence": q,
            "theta": theta,
        }

    # ------------------------------------------------------------------
    # P2 mechanism diagnostics (plan §30)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_p2_diagnostics(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> dict:
        """Best-checkpoint P2 mechanism diagnostics, JSON-safe aggregates.
        Never uses labels; does not modify model state."""
        self.eval()
        edge_index = torch.as_tensor(edge_index, dtype=torch.long, device=x.device)
        factors, _z_local = self._encode(x)
        num_nodes = int(x.size(0))
        f_block = torch.stack([factors["c"], factors["p_t"], factors["p_v"]], dim=1)
        graph_out = self._graph_update(f_block, edge_index, num_nodes)
        gamma = graph_out["gamma"]  # [N, F, K+1]
        factor_names = ["C", "Pt", "Pv"]

        # --- relation side (P1 metrics kept for comparability) -------------
        r = graph_out["r"]
        occ = r.mean(dim=0)
        if float(occ.sum()) > 0:
            occ = occ / occ.sum()
        k_eff = float(torch.exp(-(occ * torch.log(occ + self.eps)).sum()).item())
        h_r = float((-(r * torch.log(r + self.eps)).sum(dim=-1)).mean().item())
        s_r = 1.0 - h_r / float(torch.log(torch.tensor(float(self.num_relations))).item())

        # --- local / graph mass (§30.1) ------------------------------------
        null_mass = gamma[..., 0]  # [N, F]
        graph_mass = graph_out["beta"]  # [N, F]
        quantiles = torch.tensor([0.1, 0.5, 0.9], dtype=gamma.dtype, device=gamma.device)
        plan: dict[str, dict[str, float]] = {}
        for idx, name in enumerate(factor_names):
            n_mass = null_mass[:, idx].reshape(-1)
            g_mass = graph_mass[:, idx].reshape(-1)
            qs = torch.quantile(n_mass, quantiles)
            plan[name] = {
                "null_mean": float(n_mass.mean().item()),
                "null_std": float(n_mass.std(unbiased=False).item()),
                "null_p10": float(qs[0].item()),
                "null_p50": float(qs[1].item()),
                "null_p90": float(qs[2].item()),
                "null_high_frac": float((n_mass > 0.95).float().mean().item()),
                "graph_mass_mean": float(g_mass.mean().item()),
            }

        # --- plan entropy (§30.2) ------------------------------------------
        log_gamma = torch.log(gamma + self.eps)
        plan_entropy = -(gamma * log_gamma).sum(dim=-1)  # [N, F]
        entropy = {name: float(plan_entropy[:, idx].mean().item()) for idx, name in enumerate(factor_names)}

        # --- conditional relation selectivity (§30.3, P1-comparable JS) ----
        alpha = graph_out["alpha"]  # [N, F, K]
        log_alpha = torch.log(alpha + self.eps)
        alpha_ent = -(alpha * log_alpha).sum(dim=-1)  # [N, F]
        alpha_entropy = {name: float(alpha_ent[:, idx].mean().item()) for idx, name in enumerate(factor_names)}
        js: dict[str, float] = {}
        for i in range(len(factor_names)):
            for j in range(i + 1, len(factor_names)):
                p = alpha[:, i] + self.eps
                qq = alpha[:, j] + self.eps
                m = 0.5 * (p + qq)
                js_val = 0.5 * ((p * torch.log(p / m)).sum(dim=-1) + (qq * torch.log(qq / m)).sum(dim=-1))
                js[f"{factor_names[i]}_{factor_names[j]}"] = float(js_val.mean().item())

        # --- column capacity deviation (§30.4) -----------------------------
        col_mass = gamma.sum(dim=1)  # [N, K+1]
        nu = build_reference_capacity(
            graph_out["availability"],
            len(factor_names),
            null_prior=self.p2_null_prior,
            degree=torch.bincount(edge_index[1], minlength=num_nodes).to(torch.float32),
        )
        capacity_kl = float(
            (col_mass * torch.log((col_mass + self.eps) / (nu + self.eps))).sum(dim=-1).mean().item()
        )
        capacity_l1 = float((col_mass - nu).abs().sum(dim=-1).mean().item())

        # --- relation confidence / theta (§30.5) ---------------------------
        q = graph_out["relation_confidence"]  # [N]
        theta = graph_out["theta"]  # [N]
        qs_q = torch.quantile(q, quantiles)
        qs_t = torch.quantile(theta, quantiles)

        # --- conditional usage matrix (factor x relation, graph-normalized) -
        usage = (graph_mass.unsqueeze(-1) * alpha).mean(dim=0)  # [F, K]

        return {
            "relation": {
                "effective_num": k_eff,
                "mean_edge_entropy": h_r,
                "specialization": s_r,
                "occ": [float(v) for v in occ.cpu().tolist()],
            },
            "plan": plan,
            "plan_entropy": entropy,
            "alpha_entropy": alpha_entropy,
            "alpha_js": js,
            "capacity_kl": capacity_kl,
            "capacity_l1": capacity_l1,
            "relation_confidence": {
                "mean": float(q.mean().item()),
                "p10": float(qs_q[0].item()),
                "p50": float(qs_q[1].item()),
                "p90": float(qs_q[2].item()),
            },
            "theta": {
                "mean": float(theta.mean().item()),
                "p10": float(qs_t[0].item()),
                "p50": float(qs_t[1].item()),
                "p90": float(qs_t[2].item()),
            },
            "usage_matrix": {
                "factors": factor_names,
                "relations": [f"R{i + 1}" for i in range(self.num_relations)],
                "values": [[float(v) for v in row] for row in usage.cpu().tolist()],
            },
        }
