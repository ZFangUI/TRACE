"""
FM extra scores: quantile aggregation and ODE NLL.

FMQuantileScore:
    s(x,y) = Quantile_{t,y0}[‖v̂ - (y-y0)‖²]
    Quantile aggregation for sharper gap/mode contrast.

FMODENLLScore:
    s(x,y) = 0.5‖z‖² - log_det
    where z, log_det = encode_with_logdet(x, y)
    Full NLL in latent space, analogous to NF-NLL.
    Uses Hutchinson trace estimator for log|det(dz/dy)|.
"""

import torch
import numpy as np


class FMQuantileScore:
    """Path consistency score with quantile aggregation.

    Instead of E_{t,y0}[loss], computes Quantile_{t,y0}[loss].
    """

    name = "FM-Quantile"

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
            s = self.model.path_score_quantile(
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
                s = self.model.path_score_quantile(
                    yb, xb,
                    n_timesteps=self.n_timesteps,
                    n_repeats=self.n_repeats,
                    quantile=self.quantile)
                run_scores.append(s.cpu().numpy())
            total_scores += np.concatenate(run_scores)
        return total_scores / n_avg


class FMODENLLScore:
    """ODE NLL score: 0.5‖z‖² - log|det(dz/dy)|.

    Full negative log-likelihood in latent space via ODE transport
    with Hutchinson trace estimator for the Jacobian log-determinant.

    Analogous to NF-NLL but using FM's learned ODE flow.

    Note: requires gradient computation (not @torch.no_grad).
    More expensive than FM-ODE-Ball due to Hutchinson trace,
    but should give tighter regions (density level set).
    """

    name = "FM-ODE-NLL"

    def __init__(self, model, device="cpu", n_steps=50, n_hutchinson=5):
        """
        Args:
            model: ConditionalFlowMatching with .encode_with_logdet()
            n_steps: ODE integration steps (Euler only for trace)
            n_hutchinson: random vectors per step for trace estimation
        """
        self.model = model
        self.device = device
        self.n_steps = n_steps
        self.n_hutchinson = n_hutchinson

    def compute(self, x, y, batch_size=128):
        """Compute ODE NLL scores. Needs grad, smaller batch."""
        self.model.to(self.device).eval()
        n = x.shape[0]
        scores = []
        for i in range(0, n, batch_size):
            xb = x[i:i+batch_size].to(self.device)
            yb = y[i:i+batch_size].to(self.device)
            z, log_det = self.model.encode_with_logdet(
                xb, yb,
                n_steps=self.n_steps,
                n_hutchinson=self.n_hutchinson)
            # NLL = 0.5‖z‖² - log|det(dz/dy)|
            # Higher = less likely
            s = 0.5 * (z ** 2).sum(dim=-1) - log_det
            scores.append(s.detach().cpu())
        return torch.cat(scores).numpy()

    def compute_on_grid(self, x_point, y_grid, batch_size=256, **kwargs):
        self.model.to(self.device).eval()
        M = y_grid.shape[0]
        all_scores = []
        for i in range(0, M, batch_size):
            yb = y_grid[i:i+batch_size].to(self.device)
            B = yb.shape[0]
            xb = x_point.unsqueeze(0).expand(B, -1).to(self.device)
            z, log_det = self.model.encode_with_logdet(
                xb, yb,
                n_steps=self.n_steps,
                n_hutchinson=self.n_hutchinson)
            s = 0.5 * (z ** 2).sum(dim=-1) - log_det
            all_scores.append(s.detach().cpu().numpy())
        return np.concatenate(all_scores)
