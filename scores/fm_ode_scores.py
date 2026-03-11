"""
Flow Matching ODE Ball score: deterministic latent-space geometry.

s(x, y) = ‖z‖²  where z = ODE_encode(x, y)

The FM model learns an ODE velocity field v_θ(y_t, t, x) that transports
noise (t=0) to data (t=1). By integrating the ODE *backwards* from
y at t=1 to t=0, we obtain a deterministic latent code z.

If the model is well-trained, z ≈ N(0, I), and ‖z‖² is a natural
nonconformity score analogous to NF-Ball (CONTRA).

Key advantages over FM-Path:
  - Deterministic: no MC variance (ε or y₀ sampling)
  - Geometric: score = latent distance, not training loss
  - Sharp boundaries: level sets are clean, no multi-scale blur
"""

import torch
import numpy as np


class FMODEBallScore:
    """ODE Ball score: ‖z‖² where z = reverse ODE from y.

    s(x, y) = ‖encode(x, y)‖²

    This is the FM analogue of NF-Ball (CONTRA), using the learned
    ODE flow as a transport map instead of a normalizing flow.
    """

    name = "FM-ODE-Ball"

    def __init__(self, model, device="cpu", n_steps=50, solver="midpoint"):
        """
        Args:
            model: ConditionalFlowMatching with .encode() method
            device: torch device
            n_steps: ODE integration steps (20-50 typical)
            solver: "euler" or "midpoint" (midpoint recommended)
        """
        self.model = model
        self.device = device
        self.n_steps = n_steps
        self.solver = solver

    @torch.no_grad()
    def compute(self, x, y, batch_size=256):
        """Compute ODE Ball scores for calibration/test.

        Returns: numpy array [n].
        """
        self.model.to(self.device).eval()
        n = x.shape[0]
        scores = []

        for i in range(0, n, batch_size):
            xb = x[i:i+batch_size].to(self.device)
            yb = y[i:i+batch_size].to(self.device)
            z = self.model.encode(xb, yb,
                                  n_steps=self.n_steps,
                                  solver=self.solver)
            s = (z ** 2).sum(dim=-1)  # ‖z‖²
            scores.append(s.cpu())

        return torch.cat(scores).numpy()

    @torch.no_grad()
    def compute_on_grid(self, x_point, y_grid, batch_size=2048, **kwargs):
        """Compute scores for one x and a grid of y values.

        Args:
            x_point: [x_dim]
            y_grid: [M, y_dim]
            batch_size: processing batch size
        Returns:
            scores: [M] numpy array
        """
        self.model.to(self.device).eval()
        M = y_grid.shape[0]
        all_scores = []

        for i in range(0, M, batch_size):
            yb = y_grid[i:i+batch_size].to(self.device)
            B = yb.shape[0]
            xb = x_point.unsqueeze(0).expand(B, -1).to(self.device)
            z = self.model.encode(xb, yb,
                                  n_steps=self.n_steps,
                                  solver=self.solver)
            s = (z ** 2).sum(dim=-1)
            all_scores.append(s.cpu().numpy())

        return np.concatenate(all_scores)
