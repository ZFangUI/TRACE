"""
Diffusion quantile aggregation score and ODE Ball score.

DiffQuantileScore:
    s(x,y) = Quantile_{t,ε}[‖ε - ε̂_θ(y_t, t, x)‖²]
    Uses quantile (e.g. 0.75) instead of mean over (t, ε) paths.
    Gap points have heavy-tailed loss → high quantile score.

DiffODEBallScore:
    s(x,y) = ‖z‖²  where z = DDIM_encode(x, y)
    Deterministic latent score via probability flow ODE encoding.
"""

import torch
import numpy as np


class DiffQuantileScore:
    """Denoising score with quantile aggregation.

    Instead of E_{t,ε}[loss], computes Quantile_{t,ε}[loss].
    This increases gap/mode contrast for sharper regions.
    """

    name = "Diff-Quantile"

    def __init__(self, model, device="cpu", n_timesteps=10, n_repeats=5,
                 quantile=0.75):
        self.model = model
        self.device = device
        self.n_timesteps = n_timesteps
        self.n_repeats = n_repeats
        self.quantile = quantile

    @torch.no_grad()
    def compute(self, x, y, batch_size=256):
        self.model.to(self.device).eval()
        n = x.shape[0]
        scores = []
        for i in range(0, n, batch_size):
            xb = x[i:i+batch_size].to(self.device)
            yb = y[i:i+batch_size].to(self.device)
            s = self.model.denoise_score_quantile(
                yb, xb,
                n_timesteps=self.n_timesteps,
                n_repeats=self.n_repeats,
                quantile=self.quantile)
            scores.append(s.cpu())
        return torch.cat(scores).numpy()

    @torch.no_grad()
    def compute_on_grid(self, x_point, y_grid, batch_size=512, n_avg=1, **kwargs):
        self.model.to(self.device).eval()
        M = y_grid.shape[0]
        total_scores = np.zeros(M)
        for _ in range(n_avg):
            run_scores = []
            for i in range(0, M, batch_size):
                yb = y_grid[i:i+batch_size].to(self.device)
                B = yb.shape[0]
                xb = x_point.unsqueeze(0).expand(B, -1).to(self.device)
                s = self.model.denoise_score_quantile(
                    yb, xb,
                    n_timesteps=self.n_timesteps,
                    n_repeats=self.n_repeats,
                    quantile=self.quantile)
                run_scores.append(s.cpu().numpy())
            total_scores += np.concatenate(run_scores)
        return total_scores / n_avg


class DiffODEBallScore:
    """ODE Ball score for Diffusion: ‖z‖² where z = DDIM_encode(y).

    Deterministic encoding via probability flow ODE (DDIM forward).
    Analogous to NF-Ball but using diffusion's learned transport.
    """

    name = "Diff-ODE-Ball"

    def __init__(self, model, device="cpu", n_steps=50, solver="midpoint"):
        self.model = model
        self.device = device
        self.n_steps = n_steps
        self.solver = solver

    @torch.no_grad()
    def compute(self, x, y, batch_size=256):
        self.model.to(self.device).eval()
        n = x.shape[0]
        scores = []
        for i in range(0, n, batch_size):
            xb = x[i:i+batch_size].to(self.device)
            yb = y[i:i+batch_size].to(self.device)
            z = self.model.encode(xb, yb, n_steps=self.n_steps,
                                  solver=self.solver)
            s = (z ** 2).sum(dim=-1)
            scores.append(s.cpu())
        return torch.cat(scores).numpy()

    @torch.no_grad()
    def compute_on_grid(self, x_point, y_grid, batch_size=2048, **kwargs):
        self.model.to(self.device).eval()
        M = y_grid.shape[0]
        all_scores = []
        for i in range(0, M, batch_size):
            yb = y_grid[i:i+batch_size].to(self.device)
            B = yb.shape[0]
            xb = x_point.unsqueeze(0).expand(B, -1).to(self.device)
            z = self.model.encode(xb, yb, n_steps=self.n_steps,
                                  solver=self.solver)
            s = (z ** 2).sum(dim=-1)
            all_scores.append(s.cpu().numpy())
        return np.concatenate(all_scores)
