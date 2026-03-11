"""
FM density-based nonconformity score via probability flow ODE.

s(x, y) = -log p(y|x)

v2: exact divergence + midpoint solver (fixes two bugs in v1)

v1 失败原因 (vol ~80):
  1. Euler solver (1阶) — 采样用 midpoint (2阶), density 却用 Euler, 精度不匹配
  2. Hutchinson trace estimator — 随机估计引入方差, 100步积分后误差累积

v2 修复:
  1. Midpoint solver — 和采样完全一致的 2 阶 ODE solver
  2. 精确 divergence — d=2 时直接算 ∂v₁/∂y₁ + ∂v₂/∂y₂, 零方差
     (仅需 d 次 autograd, 比 Hutchinson 的 1 次还快且精确)

理论: FM 的 velocity field 定义了一个 CNF, 反向 ODE 积分 + change of
variables 给出精确 log p(y|x). 这和 NF-NLL 本质相同 (density level set),
但不需要 invertible architecture.
"""

import torch
import numpy as np


class FMDensityScore:
    """FM density score: s(x,y) = -log p(y|x) via reverse ODE.

    v2: exact divergence (zero variance) + midpoint solver (2nd order).
    """

    name = "FM-Density"

    def __init__(self, model, device="cpu", n_steps=100, solver="midpoint"):
        """
        Args:
            model: ConditionalFlowMatching
            n_steps: ODE integration steps (50-200, more = more accurate)
            solver: "midpoint" (recommended, 2nd order) or "euler" (1st order)
        """
        self.model = model
        self.device = device
        self.n_steps = n_steps
        self.solver = solver

    def _exact_div(self, y_t, t, x):
        """Compute velocity and EXACT divergence tr(∂v/∂y).

        For d=2: div = ∂v₁/∂y₁ + ∂v₂/∂y₂
        Each term computed via one autograd call with a standard basis vector.
        Total: d autograd calls, ZERO variance (vs Hutchinson's random estimate).

        Returns:
            v: [B, d] velocity (detached)
            div: [B] exact divergence
        """
        B, d = y_t.shape

        y_req = y_t.detach().requires_grad_(True)
        v = self.model.velocity_net(y_req, t, x)

        div = torch.zeros(B, device=y_t.device)
        for i in range(d):
            # e_i = standard basis vector for dimension i
            e_i = torch.zeros_like(v)
            e_i[:, i] = 1.0
            # ∂v/∂y · e_i  gives the i-th column of Jacobian
            # then dot with e_i gives ∂v_i/∂y_i
            dvdy_i = torch.autograd.grad(
                outputs=v, inputs=y_req,
                grad_outputs=e_i,
                create_graph=False, retain_graph=(i < d - 1),
            )[0]  # [B, d]
            div += dvdy_i[:, i]  # ∂v_i/∂y_i

        return v.detach(), div

    def _reverse_ode_log_prob(self, y, x):
        """Reverse ODE from t=1 (data) to t=0 (noise).

        y is original scale — normalized internally.

        log p(y|x) = log p_0(y_0) - ∫₀¹ tr(∂v/∂y) dt + log|det(∂y_norm/∂y)|
        """
        model = self.model
        B, d = y.shape
        device = y.device
        dt = 1.0 / self.n_steps

        y_t = model._normalize(y).clone()
        total_div = torch.zeros(B, device=device)

        for step in range(self.n_steps):
            t_val = 1.0 - step * dt

            if self.solver == "midpoint":
                # ── Stage 1: evaluate at (y_t, t_val) ──
                t1 = torch.full((B,), t_val, device=device)
                v1, div1 = self._exact_div(y_t, t1, x)

                # ── Stage 2: evaluate at midpoint ──
                y_mid = y_t - 0.5 * dt * v1
                t_mid = torch.full((B,), t_val - 0.5 * dt, device=device)
                v2, div2 = self._exact_div(y_mid, t_mid, x)

                # ── Update with midpoint values ──
                y_t = y_t - dt * v2
                total_div += div2 * dt

            else:  # euler
                t1 = torch.full((B,), t_val, device=device)
                v1, div1 = self._exact_div(y_t, t1, x)

                y_t = y_t - dt * v1
                total_div += div1 * dt

        # y_t ≈ y_0 ~ N(0,I)
        log_p0 = -0.5 * (y_t ** 2).sum(-1) - 0.5 * d * np.log(2 * np.pi)

        # Jacobian of Y standardization: log|det| = -sum(log(y_std))
        log_jac_norm = -model.y_std.log().sum()

        log_prob = log_p0 - total_div + log_jac_norm
        return log_prob

    def compute(self, x, y, batch_size=128):
        self.model.to(self.device).eval()
        scores = []
        for i in range(0, x.shape[0], batch_size):
            xb = x[i:i+batch_size].to(self.device)
            yb = y[i:i+batch_size].to(self.device)
            log_prob = self._reverse_ode_log_prob(yb, xb)
            scores.append((-log_prob).cpu())
        return torch.cat(scores).detach().numpy()

    def compute_on_grid(self, x_point, y_grid, batch_size=256, **kwargs):
        self.model.to(self.device).eval()
        scores = []
        for i in range(0, y_grid.shape[0], batch_size):
            yb = y_grid[i:i+batch_size].to(self.device)
            B = yb.shape[0]
            xb = x_point.unsqueeze(0).expand(B, -1).to(self.device)
            log_prob = self._reverse_ode_log_prob(yb, xb)
            scores.append((-log_prob).cpu())
        return torch.cat(scores).detach().numpy()