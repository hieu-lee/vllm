# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class _ReqStats:
    alpha_hat: Optional[float] = None  # Exponential moving average of α


class SpeculationController:
    """
    Adaptive controller for speculative decoding depth and gating.

    Implements decisions inspired by the speedup model
    S_{A,B}(d, α) = (1 - α^{d+1}) / ((1 - α)(A + B d)).

    Parameters
    ----------
    max_d: int
        Upper bound on speculative depth provided by the underlying drafter.
    c: float
        Draft/verify cost ratio in the linear latency model; used as a gating
        threshold on α (no-regret gating).
    gamma: float
        Linear slope of verify latency growth with depth, i.e.
        t_target^(d) ≈ t_target^(1) (1 + gamma (d - 1)).
    ops_budget: Optional[float]
        If provided (> 1), an upper bound B_ops on arithmetic ops factor.
    overhead_hat_c: float
        Overhead multiplier for drafting per position, used in ops budget cap.
    ema_beta: float
        Exponential moving average factor for α (in [0, 1]). Larger values
        weigh history more.
    """

    def __init__(
        self,
        *,
        max_d: int,
        c: float = 0.3,
        gamma: float = 0.0,
        ops_budget: Optional[float] = None,
        overhead_hat_c: float = 0.0,
        ema_beta: float = 0.6,
    ) -> None:
        self.max_d = max(0, int(max_d))
        self.c = float(c)
        self.gamma = float(gamma)
        self.ops_budget = float(ops_budget) if ops_budget is not None else None
        self.overhead_hat_c = float(overhead_hat_c)
        self.ema_beta = float(ema_beta)

        # A = 1 - gamma, B = c + gamma in the linearized model
        self.A = 1.0 - self.gamma
        self.B = self.c + self.gamma

        self._stats: Dict[str, _ReqStats] = {}

    def update(self, req_id: str, drafted: int, accepted: int) -> None:
        if drafted <= 0:
            return
        alpha = max(0.0, min(1.0, accepted / drafted))
        s = self._stats.setdefault(req_id, _ReqStats())
        if s.alpha_hat is None:
            s.alpha_hat = alpha
        else:
            s.alpha_hat = self.ema_beta * s.alpha_hat + (1.0 - self.ema_beta) * alpha

    def _ops_cap(self) -> Optional[int]:
        if self.ops_budget is None:
            return None
        if self.ops_budget <= 1.0:
            return 0
        # Sufficient condition from seed note: ops factor <= 1 + d (1 + \hat c)
        # Thus d <= floor((B_ops - 1) / (1 + \hat c)).
        cap = int((self.ops_budget - 1.0) // (1.0 + self.overhead_hat_c))
        return max(0, cap)

    def _best_depth(self, alpha: float, d_cap: int) -> int:
        if alpha <= 0.0:
            return 0
        if d_cap <= 0:
            return 0
        best_d = 0
        best_s = 1.0  # S(0, α) = 1 / A; but compare relative gains safely
        # We will compute the absolute S and pick the argmax.
        for d in range(0, d_cap + 1):
            S = self._speedup(d, alpha)
            if S > best_s:
                best_s = S
                best_d = d
        return best_d

    def _speedup(self, d: int, alpha: float) -> float:
        if d < 0:
            return 1.0
        if alpha >= 1.0:
            # Limit as α -> 1 from below: S = (d + 1) / (A + B d)
            return (d + 1.0) / (self.A + self.B * d)
        denom = (1.0 - alpha) * (self.A + self.B * d)
        if denom <= 0.0:
            return 1.0
        return (1.0 - pow(alpha, d + 1)) / denom

    def decide_depth(self, req_id: str) -> int:
        s = self._stats.get(req_id)
        alpha_hat = s.alpha_hat if s else None
        # If we have no estimate yet, use the maximum depth but respect ops cap.
        d_cap = self.max_d
        ops_cap = self._ops_cap()
        if ops_cap is not None:
            d_cap = min(d_cap, ops_cap)
        if d_cap <= 0:
            return 0
        if alpha_hat is None:
            return d_cap
        # No-regret gating: only speculate if α̂ > c
        if alpha_hat <= self.c:
            return 0
        return self._best_depth(alpha_hat, d_cap)


