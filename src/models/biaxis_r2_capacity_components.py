"""R2-Design-2.5 structured-capacity components
(docs/BiAxis_R2_Design_2_5_Structured_Capacity_Utilization_Audit.md, D2.5-C).

    HopExpert          : 2-layer MLP expert  Linear(d,2d) -> LN -> GELU -> Linear(2d,d)
    FactorReadout      : per-factor readout  Linear(in,2d) -> LN -> GELU -> Dropout -> Linear(2d,d)
    DeepFusion         : 2-layer factor fusion (SEP_CONCAT / INCEPTION_012 / CAP_H1_DUP)
    ResidualFusion     : input projection + 2 residual blocks (DEEP_FUSION mode)
    WideSourceTransform: 2-layer wide source transform (WIDE_B0 mode)

Discipline (plan D2.5-C):
    - Every block has a defined function; generic capacity lives in the
      parameter-matched controls (CAP_H1_DUP / WIDE_B0), never free-floating.
    - Normal (default PyTorch) initialization everywhere: the plan forbids
      zero-initializing the whole H2 expert (C1); branches must receive
      gradients from step 0. SEP_SUM's beta starts at 0.1 (plan C1).
    - No dropout inside experts (deterministic expert outputs for the
      transmission / ablation diagnostics); dropout lives in readouts and
      fusions only.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import get_activation, make_norm


class HopExpert(nn.Module):
    """Two-layer MLP expert (plan D2.5-C C1):

        Linear(d, 2d) -> LN -> GELU -> Linear(2d, d)

    Default initialization throughout — no zero-init tail (plan C1).
    """

    def __init__(self, factor_dim: int, activation: str = "gelu", norm: str = "layernorm") -> None:
        super().__init__()
        d = int(factor_dim)
        self.net = nn.Sequential(
            nn.Linear(d, 2 * d),
            make_norm(norm, 2 * d),
            get_activation(activation),
            nn.Linear(2 * d, d),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [N, d] -> expert output [N, d]."""
        return self.net(h)


class FactorReadout(nn.Module):
    """Per-factor readout of the concatenated expert block (plan D2.5-C C2):

        Linear(in_d, 2d) -> LN -> GELU -> Dropout -> Linear(2d, d)

    in_d = 3d for SEP_CONCAT / CAP_H1_DUP ([F | e1 | e2]),
    in_d = 4d for INCEPTION_012 ([F | e0 | e1 | e2]).
    """

    def __init__(
        self,
        in_dim: int,
        factor_dim: int,
        dropout: float = 0.2,
        activation: str = "gelu",
        norm: str = "layernorm",
    ) -> None:
        super().__init__()
        d = int(factor_dim)
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), 2 * d),
            make_norm(norm, 2 * d),
            get_activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(2 * d, d),
        )

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """q: [N, in_d] -> factor correction [N, d]."""
        return self.net(q)


class DeepFusion(nn.Module):
    """Two-layer factor fusion replacing the P0 one-layer fusion
    (plan D2.5-C C2: "Use a stronger 2-layer factor fusion instead of the
    current one-layer fusion").

        Linear(3d, mid) -> LN -> GELU -> Dropout
        -> Linear(mid, h) -> LN -> GELU -> Dropout

    mid = 2h by default; WIDE_B0 uses a widened mid to reach the
    parameter-matched budget.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        mid_dim: int | None = None,
        dropout: float = 0.2,
        activation: str = "gelu",
        norm: str = "layernorm",
    ) -> None:
        super().__init__()
        h = int(out_dim)
        mid = int(mid_dim) if mid_dim is not None else 2 * h
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), mid),
            make_norm(norm, mid),
            get_activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(mid, h),
            make_norm(norm, h),
            get_activation(activation),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualFusion(nn.Module):
    """DEEP_FUSION fusion (plan D2.5-C C6): input projection + 2 residual
    MLP blocks, all of width h.

        z0 = Dropout(GELU(LN(Linear(3d, h)(x))))
        z  = z0 + Dropout(GELU(LN(Linear(h, h)(z0))))
        z  = z  + Dropout(GELU(LN(Linear(h, h)(z))))
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.2,
        activation: str = "gelu",
        norm: str = "layernorm",
    ) -> None:
        super().__init__()
        h = int(out_dim)
        self.act = get_activation(activation)
        self.drop = nn.Dropout(float(dropout))
        self.proj = nn.Linear(int(in_dim), h)
        self.norm0 = make_norm(norm, h)

        def _block() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(h, h),
                make_norm(norm, h),
                get_activation(activation),
                nn.Dropout(float(dropout)),
            )

        self.block1 = _block()
        self.block2 = _block()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.drop(self.act(self.norm0(self.proj(x))))
        z = z + self.block1(z)
        z = z + self.block2(z)
        return z


class WideSourceTransform(nn.Module):
    """WIDE_B0 source transform (plan D2.5-C C5): a 2-layer MLP with a
    widened hidden width W, parameter-matched to the SEP_CONCAT budget.

        Linear(d, W) -> LN -> GELU -> Linear(W, d)
    """

    def __init__(
        self,
        factor_dim: int,
        width: int,
        activation: str = "gelu",
        norm: str = "layernorm",
    ) -> None:
        super().__init__()
        d = int(factor_dim)
        w = int(width)
        self.net = nn.Sequential(
            nn.Linear(d, w),
            make_norm(norm, w),
            get_activation(activation),
            nn.Linear(w, d),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [N, d] -> transformed [N, d]."""
        return self.net(h)
