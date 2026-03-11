"""
Diffusion-based nonconformity score: Denoising Consistency.

s(x, y) = E_{t,ε}[‖ε - ε̂_θ(y_t, t, x)‖²]

Key insight: if y is a typical sample from p(y|x), the trained
denoiser should predict the added noise accurately at all scales.
Atypical y → denoiser struggles → high score.

Implementation uses CRN (Common Random Numbers):
  - A fixed eps_bank[t_idx, r_idx, :] is pre-generated once at init.
  - For each (y, t, r), we use eps_bank[t_idx, r_idx] (broadcast over batch).
  - This makes s(x,y) a deterministic function independent of batch order.
"""

import torch
import numpy as np


class DiffusionDenoiseScore:
    """Denoising consistency score for conformal prediction.

    s(x, y) = (1/|T|·R) Σ_{t∈T} Σ_{r=1}^{R} ‖ε_r - ε̂_θ(y_t^r, t, x)‖²

    where y_t^r = √(ᾱ_t)·y + √(1-ᾱ_t)·ε_r

    Uses CRN: eps_bank is pre-generated at __init__ and shared across all
    compute() calls, making score independent of batch size / order.
    """

    name = "Diffusion-Denoise"

    def __init__(self, model, device="cpu", n_timesteps=10, n_repeats=5,
                 seed=42):
        """
        Args:
            model: ConditionalDDPM
            n_timesteps: number of t values to average over
            n_repeats: noise samples per (y, t) pair
            seed: random seed for CRN table generation
        """
        self.model = model
        self.device = device
        self.n_timesteps = n_timesteps
        self.n_repeats = n_repeats
        self.seed = seed

        # Pre-compute timestep tensor (fixed, never changes)
        T = self.model.T
        self.ts = torch.linspace(1, T - 1, n_timesteps).long()  # CPU tensor

        # Pre-generate CRN eps bank: [n_timesteps, n_repeats, y_dim]
        # Same eps used for every (x, y) pair → s(x,y) is a pure function.
        y_dim = model.y_dim
        rng = torch.Generator().manual_seed(seed)
        self.eps_bank = torch.randn(
            n_timesteps, n_repeats, y_dim, generator=rng
        )  # stays on CPU, moved to device once per compute call

    @torch.no_grad()
    def compute(self, x, y, batch_size=256):
        """Compute denoising scores for (x, y) pairs.

        Returns: numpy array of scores [n].
        """
        self.model.to(self.device).eval()
        n = x.shape[0]
        scores = []
        bank_dev = self.eps_bank.to(self.device)
        ts_dev = self.ts.to(self.device)

        for i in range(0, n, batch_size):
            xb = x[i:i+batch_size].to(self.device)
            yb = y[i:i+batch_size].to(self.device)
            s = self.model.denoise_score(
                yb, xb,
                timesteps=ts_dev,
                n_repeats=self.n_repeats,
                eps_bank=bank_dev,
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
                n_avg=1: default (fast)
                n_avg=3~5: recommended for plotting (smoother boundaries)
            tau: if provided, enable early rejection — any y whose running
                 lower bound exceeds tau is immediately assigned score=inf.
        Returns:
            scores: [M] numpy array
        """
        self.model.to(self.device).eval()
        M = y_grid.shape[0]
        bank_dev = self.eps_bank.to(self.device)
        ts_dev = self.ts.to(self.device)

        # For n_avg > 1, generate additional CRN tables with offset seeds
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
                s = self.model.denoise_score(
                    yb, xb,
                    timesteps=ts_dev,
                    n_repeats=self.n_repeats,
                    eps_bank=bank,
                    tau=tau,
                )
                run_scores.append(s.cpu().numpy())
            total_scores += np.concatenate(run_scores)

        return total_scores / n_avg
