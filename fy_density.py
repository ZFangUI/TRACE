"""
FM density-based nonconformity score via probability flow ODE.

s(x, y) = -log p(y|x)

For a flow matching model with velocity v_θ(y_t, t, x), the
probability flow ODE gives:

    log p(y|x) = log p_0(y_0) - ∫₀¹ tr(∂v/∂y) dt

where y_0 is obtained by *reverse* ODE integration from y at t=1
back to t=0, and p_0 = N(0, I).

The divergence tr(∂v/∂y) is estimated via Hutchinson's trace
estimator: tr(A) = E_ε[εᵀ A ε] with ε ~ N(0,I).

This gives Diffusion/FM a density-level-set score analogous to
NF-NLL, enabling non-convex prediction regions (spirals, rings, etc).
"""

import torch
import numpy as np


class FMDensityScore:
    """FM density score: s(x,y) = -log p(y|x) via probability flow ODE.

    Prediction region: C(x) = {y : -log p(y|x) ≤ τ}
    This is a density level set → optimal volume by Neyman-Pearson.
    """

    name = "FM-Density"

    def __init__(self, model, device="cpu", n_steps=100, n_hutchinson=1):
        """
        Args:
            model: ConditionalFlowMatching with .velocity_net
            device: torch device
            n_steps: Euler steps for ODE integration (more = more accurate)
            n_hutchinson: Hutchinson samples for trace estimation (1 is often ok)
        """
        self.model = model
        self.device = device
        self.n_steps = n_steps
        self.n_hutchinson = n_hutchinson

    def _reverse_ode_log_prob(self, y, x):
        """Reverse ODE from t=1 (data) to t=0 (noise), computing log-prob.

        Solves:
            dy/dt = v(y, t, x)         [forward ODE]
            d(log p)/dt = -tr(∂v/∂y)   [instantaneous change of variables]

        We integrate *backward* from t=1 to t=0:
            y_{t-dt} = y_t - v(y_t, t, x) * dt
            log_prob += tr(∂v/∂y) * dt

        At t=0: log p(y|x) = log p_0(y_0) - ∫ tr(∂v/∂y) dt
                            = -0.5‖y_0‖² - (d/2)log(2π) - ∫ tr(∂v/∂y) dt

        Args:
            y: [B, d] data points (at t=1)
            x: [B, x_dim] conditioning
        Returns:
            log_prob: [B] log p(y|x)
        """
        model = self.model
        B, d = y.shape
        device = y.device
        dt = 1.0 / self.n_steps

        y_t = y.clone()
        total_divergence = torch.zeros(B, device=device)

        for step in range(self.n_steps):
            t_val = 1.0 - step * dt  # from 1.0 down to dt
            t = torch.full((B,), t_val, device=device)

            # Compute velocity and divergence simultaneously
            y_t_req = y_t.detach().requires_grad_(True)
            v = model.velocity_net(y_t_req, t, x)

            # Hutchinson trace estimator: tr(∂v/∂y) = E[εᵀ (∂v/∂y) ε]
            div_est = torch.zeros(B, device=device)
            for _ in range(self.n_hutchinson):
                eps = torch.randn_like(y_t_req)
                # ε^T (∂v/∂y) computed via vector-Jacobian product
                vjp = torch.autograd.grad(
                    outputs=v, inputs=y_t_req,
                    grad_outputs=eps,
                    create_graph=False, retain_graph=True,
                )[0]  # [B, d]
                div_est += (eps * vjp).sum(-1)  # εᵀ (∂v/∂y) ε → [B]
            div_est /= self.n_hutchinson

            total_divergence += div_est * dt

            # Reverse Euler step: y_{t-dt} = y_t - v * dt
            y_t = y_t - v.detach() * dt

        # y_t is now y_0 (noise space)
        # log p_0(y_0) = -0.5‖y_0‖² - (d/2)log(2π)
        log_p0 = -0.5 * (y_t ** 2).sum(-1) - 0.5 * d * np.log(2 * np.pi)

        # log p(y|x) = log p_0(y_0) - ∫ div dt
        #   (negative because we integrated backward: change of sign)
        log_prob = log_p0 - total_divergence

        return log_prob

    def compute(self, x, y, batch_size=128, seed=42):
        """Compute NLL scores for (x, y) pairs.

        Returns: numpy array of scores [n] (higher = less likely).
        """
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.model.to(self.device).eval()
        n = x.shape[0]
        scores = []

        for i in range(0, n, batch_size):
            xb = x[i:i+batch_size].to(self.device)
            yb = y[i:i+batch_size].to(self.device)
            log_prob = self._reverse_ode_log_prob(yb, xb)
            s = -log_prob  # NLL as score
            scores.append(s.cpu())

        return torch.cat(scores).detach().numpy()

    def compute_on_grid(self, x_point, y_grid, batch_size=256, seed=42, **kwargs):
        """Compute NLL scores for one x and a grid of y values.

        Args:
            x_point: [x_dim] single x
            y_grid: [M, y_dim] grid of candidate y's
        Returns:
            scores: [M] numpy array
        """
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.model.to(self.device).eval()
        M = y_grid.shape[0]
        scores = []

        for i in range(0, M, batch_size):
            yb = y_grid[i:i+batch_size].to(self.device)
            B = yb.shape[0]
            xb = x_point.unsqueeze(0).expand(B, -1).to(self.device)
            log_prob = self._reverse_ode_log_prob(yb, xb)
            s = -log_prob
            scores.append(s.cpu())

        return torch.cat(scores).detach().numpy()