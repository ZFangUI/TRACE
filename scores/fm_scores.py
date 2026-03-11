"""
Flow Matching nonconformity score: Path Consistency.

s(x, y) = E_{t, y_0}[‖v_θ(y_t, t, x) - (y - y_0)‖²]

Key insight: for OT-CFM, the true conditional velocity is u_t = y_1 - y_0.
If y (= y_1) is a typical sample from p(y|x), the velocity network should
predict the straight-line velocity accurately. Atypical y → predicted
velocity deviates from straight line → high score.

Implementation uses CRN (Common Random Numbers):
  - A fixed y0_bank[t_idx, r_idx, :] is pre-generated once at init.
  - For each (y, t, r), we use y0_bank[t_idx, r_idx] (broadcast over batch).
  - This makes s(x,y) a deterministic function independent of batch order.
"""

import torch
import numpy as np


class FMPathScore:
    """Path consistency score for conformal prediction.

    s(x, y) = (1/|T|·R) Σ_{t∈T} Σ_{r=1}^{R} ‖v̂(y_t^r, t, x) - (y - y_0^r)‖²

    where y_t^r = (1-t)·y_0^r + t·y,  y_0^r ~ N(0,I)

    Uses CRN: y0_bank is pre-generated at __init__ and shared across all
    compute() calls, making score independent of batch size / order.
    """

    name = "FM-Path"

    def __init__(self, model, device="cpu", n_timesteps=10, n_repeats=5,
                 seed=42):
        """
        Args:
            model: ConditionalFlowMatching
            n_timesteps: number of t values
            n_repeats: source samples y_0 per t
            seed: random seed for CRN table generation
        """
        self.model = model
        self.device = device
        self.n_timesteps = n_timesteps
        self.n_repeats = n_repeats
        self.seed = seed

        # Pre-compute timestep tensor (fixed)
        self.ts = torch.linspace(0.01, 0.99, n_timesteps)  # CPU float tensor

        # Pre-generate CRN y0 bank: [n_timesteps, n_repeats, y_dim]
        y_dim = model.y_dim
        rng = torch.Generator().manual_seed(seed)
        self.y0_bank = torch.randn(
            n_timesteps, n_repeats, y_dim, generator=rng
        )  # stays on CPU

    @torch.no_grad()
    def compute(self, x, y, batch_size=256):
        """Compute path consistency scores.

        Returns: numpy array [n].
        """
        self.model.to(self.device).eval()
        n = x.shape[0]
        scores = []
        bank_dev = self.y0_bank.to(self.device)
        ts_dev = self.ts.to(self.device)

        for i in range(0, n, batch_size):
            xb = x[i:i+batch_size].to(self.device)
            yb = y[i:i+batch_size].to(self.device)
            s = self.model.path_score(
                yb, xb,
                timesteps=ts_dev,
                n_repeats=self.n_repeats,
                y0_bank=bank_dev,
            )
            scores.append(s.cpu())

        return torch.cat(scores).numpy()

    @torch.no_grad()
    def compute_on_grid(self, x_point, y_grid, batch_size=4096, n_avg=1,
                        tau=None):
        """Compute scores for one x and a grid of y values.

        Args:
            x_point: [x_dim]
            y_grid: [M, y_dim]
            batch_size: processing batch size
            n_avg: multiple evaluations averaged (uses different CRN tables).
            tau: if provided, enable early rejection.
        Returns:
            scores: [M] numpy array
        """
        self.model.to(self.device).eval()
        M = y_grid.shape[0]
        bank_dev = self.y0_bank.to(self.device)
        ts_dev = self.ts.to(self.device)

        if n_avg > 1:
            extra_banks = []
            for avg_i in range(1, n_avg):
                rng = torch.Generator().manual_seed(self.seed + avg_i)
                bank = torch.randn(
                    self.n_timesteps, self.n_repeats, self.model.y_dim,
                    generator=rng
                ).to(self.device)
                extra_banks.append(bank)
            all_banks = [bank_dev] + extra_banks
        else:
            all_banks = [bank_dev]

        total_scores = np.zeros(M)
        for bank in all_banks:
            run_scores = []
            for i in range(0, M, batch_size):
                yb = y_grid[i:i+batch_size].to(self.device)
                B = yb.shape[0]
                xb = x_point.unsqueeze(0).expand(B, -1).to(self.device)
                s = self.model.path_score(
                    yb, xb,
                    timesteps=ts_dev,
                    n_repeats=self.n_repeats,
                    y0_bank=bank,
                    tau=tau,
                )
                run_scores.append(s.cpu().numpy())
            total_scores += np.concatenate(run_scores)

        return total_scores / n_avg
