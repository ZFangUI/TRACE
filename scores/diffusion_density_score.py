"""
Diffusion density-based nonconformity score via probability flow ODE.

s(x, y) = -log p(y|x)

v2: exact divergence + midpoint solver (fixes two bugs in v1)

v1 失败原因 (vol ~80):
  1. Euler solver (1阶) — 精度不足, ODE 轨迹偏移
  2. Hutchinson trace estimator — 随机 divergence 估计引入方差

v2 修复:
  1. Midpoint solver — 2 阶 ODE solver
  2. 精确 divergence — d=2 时 ∂f₁/∂y₁ + ∂f₂/∂y₂, d 次 autograd, 零方差

Probability flow ODE for DDPM:
  dy/dt = f(y,t) = -0.5 β(t) y - 0.5 β(t) ε̂/√(1-ᾱ_t)
  (forward: data → noise, t: 0 → 1)
"""

import torch
import numpy as np


class DiffusionDensityScore:
    """Diffusion density score: s(x,y) = -log p(y|x).

    v2: exact divergence (zero variance) + midpoint solver (2nd order).
    """

    name = "Diff-Density"

    def __init__(self, model, device="cpu", n_steps=200, solver="midpoint"):
        """
        Args:
            model: ConditionalDDPM
            n_steps: ODE integration steps
            solver: "midpoint" (recommended) or "euler"
        """
        self.model = model
        self.device = device
        self.n_steps = n_steps
        self.solver = solver

    def _get_beta(self, t_val):
        """Interpolate beta schedule at continuous time t_val ∈ [0, 1]."""
        model = self.model
        t_idx = t_val * (model.T - 1)
        t_idx_lo = int(t_idx)
        t_idx_hi = min(t_idx_lo + 1, model.T - 1)
        frac = t_idx - t_idx_lo
        return ((1 - frac) * model.betas[t_idx_lo]
                + frac * model.betas[t_idx_hi])

    def _drift_and_exact_div(self, y_t, t_val, x):
        """Compute probability flow ODE drift and EXACT divergence.

        drift = -0.5 β(t) [y + ε̂/√(1-ᾱ_t)]   (toward noise)

        For d=2: div = ∂f₁/∂y₁ + ∂f₂/∂y₂, computed via d autograd calls.

        Returns:
            drift: [B, d] (detached)
            div: [B] exact divergence
        """
        model = self.model
        B, d = y_t.shape
        device = y_t.device

        beta_val = self._get_beta(t_val)
        beta = torch.full((B, 1), beta_val.item(), device=device)

        t_continuous = torch.full((B,), t_val, device=device)
        t_discrete = (t_continuous * (model.T - 1)).long().clamp(0, model.T - 1)
        t_norm = t_discrete.float() / model.T

        sqrt_omab = model.sqrt_one_minus_alpha_bar[t_discrete].unsqueeze(-1)
        sqrt_omab = sqrt_omab.clamp(min=1e-6)

        y_req = y_t.detach().requires_grad_(True)
        eps_hat = model.noise_net(y_req, t_norm, x)
        score = -eps_hat / sqrt_omab
        drift = -0.5 * beta * y_req + 0.5 * beta * score

        # Exact divergence: sum of diagonal of Jacobian
        div = torch.zeros(B, device=device)
        for i in range(d):
            e_i = torch.zeros_like(drift)
            e_i[:, i] = 1.0
            dfdy_i = torch.autograd.grad(
                outputs=drift, inputs=y_req,
                grad_outputs=e_i,
                create_graph=False, retain_graph=(i < d - 1),
            )[0]
            div += dfdy_i[:, i]

        return drift.detach(), div

    def _ode_log_prob(self, y, x):
        """Forward ODE from t=0 (data) to t=1 (noise).

        y is original scale — normalized internally.
        """
        model = self.model
        B, d = y.shape
        device = y.device
        dt = 1.0 / self.n_steps

        y_t = model._normalize(y).clone()
        total_div = torch.zeros(B, device=device)

        for step in range(self.n_steps):
            t_val = step * dt

            if self.solver == "midpoint":
                # ── Stage 1 ──
                f1, d1 = self._drift_and_exact_div(y_t, t_val, x)

                # ── Stage 2: midpoint ──
                y_mid = y_t + 0.5 * dt * f1
                t_mid = t_val + 0.5 * dt
                f2, d2 = self._drift_and_exact_div(y_mid, t_mid, x)

                # ── Update ──
                y_t = y_t + dt * f2
                total_div += d2 * dt

            else:  # euler
                f1, d1 = self._drift_and_exact_div(y_t, t_val, x)
                y_t = y_t + dt * f1
                total_div += d1 * dt

        # y_t ≈ y_T ~ N(0,I)
        log_pT = -0.5 * (y_t ** 2).sum(-1) - 0.5 * d * np.log(2 * np.pi)

        # Jacobian of Y standardization
        log_jac_norm = -model.y_std.log().sum()

        log_prob = log_pT - total_div + log_jac_norm
        return log_prob

    def compute(self, x, y, batch_size=64):
        self.model.to(self.device).eval()
        n = x.shape[0]
        scores = []
        for i in range(0, n, batch_size):
            xb = x[i:i+batch_size].to(self.device)
            yb = y[i:i+batch_size].to(self.device)
            log_prob = self._ode_log_prob(yb, xb)
            scores.append((-log_prob).cpu())
        return torch.cat(scores).detach().numpy()

    def compute_on_grid(self, x_point, y_grid, batch_size=128, **kwargs):
        self.model.to(self.device).eval()
        M = y_grid.shape[0]
        scores = []
        for i in range(0, M, batch_size):
            yb = y_grid[i:i+batch_size].to(self.device)
            B = yb.shape[0]
            xb = x_point.unsqueeze(0).expand(B, -1).to(self.device)
            log_prob = self._ode_log_prob(yb, xb)
            scores.append((-log_prob).cpu())
        return torch.cat(scores).detach().numpy()