"""
Mode Attraction Score for Conformal Prediction.

Core idea: gradient ascent on ∇_y log p(y|x) from y to nearest mode y*.
Score s(x,y) = ‖y - y*‖² — deterministic, multimodal-aware.

Three implementations:
  - NF:        exact ∇_y log p via autograd on NF log-likelihood
  - Diffusion: ε̂ direction (denoising direction at low noise)
  - FM:        velocity field at t* ≈ 1 as ascent direction

Key design choices (learned from v1 failures):
  - NF: gradient clipping to prevent NaN from coupling layer overflow
  - Diff: use -ε̂ directly as direction (do NOT divide by sqrt(1-ᾱ))
          because 1/sqrt(1-ᾱ) ≈ 11-60× at low t, causing explosion
  - FM: normalize velocity per-step to control step size
  - All: score upper-bound clipping for numerical safety
"""

import torch
import numpy as np


# ═══════════════════════════════════════════════════════════════════
# NF Mode Attraction
# ═══════════════════════════════════════════════════════════════════

class NFModeAttractionScore:
    """s(x,y) = ‖y - y*‖²  where y* = argmax_ỹ log p(ỹ|x) via ascent.

    Uses NF's exact log p(y|x) and autograd for the gradient.
    Gradient is clipped per-step to prevent NaN from exp overflow
    in coupling layers.
    """

    name = "NF-ModeAttract"

    def __init__(self, model, device="cpu", n_steps=50, lr=0.05,
                 grad_clip=5.0):
        self.model = model
        self.device = device
        self.n_steps = n_steps
        self.lr = lr
        self.grad_clip = grad_clip

    def _gradient_ascent(self, x, y_init):
        """Gradient ascent on log p(y|x).  Returns y* (nearest mode)."""
        self.model.to(self.device).eval()

        # Freeze model so backward only differentiates w.r.t. y
        orig_flags = {p: p.requires_grad for p in self.model.parameters()}
        for p in self.model.parameters():
            p.requires_grad_(False)

        y_cur = y_init.clone().detach()
        for _ in range(self.n_steps):
            y_cur.requires_grad_(True)
            log_p = self.model.log_prob(x, y_cur)  # [B]

            # Guard against NaN log_prob
            if torch.isnan(log_p).any() or torch.isinf(log_p).any():
                y_cur = y_cur.detach()
                break

            grad = torch.autograd.grad(log_p.sum(), y_cur)[0]

            # Clip gradient to prevent explosions
            grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            grad = grad * torch.clamp(self.grad_clip / grad_norm, max=1.0)

            y_cur = (y_cur + self.lr * grad).detach()

        # Restore model grad flags
        for p, flag in orig_flags.items():
            p.requires_grad_(flag)

        return y_cur

    def compute(self, x, y, batch_size=256):
        """Compute scores for (x, y) pairs.  Returns numpy [n]."""
        self.model.to(self.device)
        n = x.shape[0]
        all_scores = []
        for i in range(0, n, batch_size):
            xb = x[i:i+batch_size].to(self.device)
            yb = y[i:i+batch_size].to(self.device)
            y_star = self._gradient_ascent(xb, yb)
            s = ((yb - y_star) ** 2).sum(-1)
            all_scores.append(s.cpu())
        scores = torch.cat(all_scores).numpy()
        # Safety: replace NaN/Inf with large finite value
        scores = np.nan_to_num(scores, nan=1e6, posinf=1e6, neginf=0.0)
        return scores

    def compute_on_grid(self, x_point, y_grid, batch_size=512, **kwargs):
        """Compute scores for one x and a grid of y's.  Returns numpy [M]."""
        self.model.to(self.device)
        M = y_grid.shape[0]
        all_scores = []
        for i in range(0, M, batch_size):
            yb = y_grid[i:i+batch_size].to(self.device)
            B = yb.shape[0]
            xb = x_point.unsqueeze(0).expand(B, -1).to(self.device)
            y_star = self._gradient_ascent(xb, yb)
            s = ((yb - y_star) ** 2).sum(-1)
            all_scores.append(s.cpu())
        scores = torch.cat(all_scores).numpy()
        scores = np.nan_to_num(scores, nan=1e6, posinf=1e6, neginf=0.0)
        return scores


# ═══════════════════════════════════════════════════════════════════
# Diffusion Mode Attraction
# ═══════════════════════════════════════════════════════════════════

class DiffusionModeAttractionScore:
    """s(x,y) = ‖y_norm - y*_norm‖²  via score-based gradient ascent.

    CRITICAL: we use -ε̂ directly as the ascent direction, NOT
    -ε̂ / sqrt(1-ᾱ_t).  At low t (e.g. t_star=0.05), the factor
    1/sqrt(1-ᾱ) ≈ 11-60×, which causes y* to explode.

    The raw -ε̂ already points toward the data manifold (it predicts
    the noise to remove), which is exactly the ascent direction we need.
    We normalize it to unit norm so lr controls actual step size.
    """

    name = "Diff-ModeAttract"

    def __init__(self, model, device="cpu", n_steps=50, lr=0.01,
                 t_star=0.05):
        """
        Args:
            t_star: noise level as fraction of T.  Range 0.03 ~ 0.15.
            lr: step size (default 0.01, smaller than NF because
                we normalize direction to unit norm).
        """
        self.model = model
        self.device = device
        self.n_steps = n_steps
        self.lr = lr
        self.t_star = t_star

    @torch.no_grad()
    def _ascent_direction(self, y_norm, x, t_idx):
        """Ascent direction = -ε̂ (denoising direction), normalized.

        We do NOT divide by sqrt(1-ᾱ_t) — that's the mathematical
        score ∇ log p_t, but it amplifies by 11-60× at low t, causing
        explosion.  The raw -ε̂ is a unit-scale direction toward data.
        """
        B = y_norm.shape[0]
        t = t_idx.expand(B)
        t_float = t.float() / self.model.T

        v_hat = self.model.noise_net(y_norm, t_float, x)
        eps_hat = self.model._v_to_eps(v_hat, y_norm, t)

        direction = -eps_hat  # toward data
        # Normalize to unit norm per sample
        d_norm = direction.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return direction / d_norm

    @torch.no_grad()
    def _gradient_ascent(self, x, y_norm, t_idx):
        """Deterministic gradient ascent in normalized y-space."""
        y_cur = y_norm.clone()
        for _ in range(self.n_steps):
            direction = self._ascent_direction(y_cur, x, t_idx)
            y_cur = y_cur + self.lr * direction
        return y_cur

    def compute(self, x, y, batch_size=256):
        self.model.to(self.device).eval()
        n = x.shape[0]
        all_scores = []

        t_star_idx = max(1, int(self.t_star * self.model.T))
        t_idx = torch.tensor(t_star_idx, device=self.device).long()

        for i in range(0, n, batch_size):
            xb = x[i:i+batch_size].to(self.device)
            yb = y[i:i+batch_size].to(self.device)
            yb_norm = self.model._normalize(yb)
            y_star_norm = self._gradient_ascent(xb, yb_norm, t_idx)
            s = ((yb_norm - y_star_norm) ** 2).sum(-1)
            all_scores.append(s.cpu())

        scores = torch.cat(all_scores).numpy()
        scores = np.nan_to_num(scores, nan=1e6, posinf=1e6, neginf=0.0)
        return scores

    def compute_on_grid(self, x_point, y_grid, batch_size=512, **kwargs):
        self.model.to(self.device).eval()
        M = y_grid.shape[0]
        all_scores = []

        t_star_idx = max(1, int(self.t_star * self.model.T))
        t_idx = torch.tensor(t_star_idx, device=self.device).long()

        for i in range(0, M, batch_size):
            yb = y_grid[i:i+batch_size].to(self.device)
            B = yb.shape[0]
            xb = x_point.unsqueeze(0).expand(B, -1).to(self.device)
            yb_norm = self.model._normalize(yb)
            y_star_norm = self._gradient_ascent(xb, yb_norm, t_idx)
            s = ((yb_norm - y_star_norm) ** 2).sum(-1)
            all_scores.append(s.cpu())

        scores = torch.cat(all_scores).numpy()
        scores = np.nan_to_num(scores, nan=1e6, posinf=1e6, neginf=0.0)
        return scores


# ═══════════════════════════════════════════════════════════════════
# Flow Matching Mode Attraction
# ═══════════════════════════════════════════════════════════════════

class FMModeAttractionScore:
    """s(x,y) = ‖y_norm - y*_norm‖²  via velocity-based ascent.

    At t close to 1, v(y, t, x) points roughly toward data modes.
    We normalize the velocity per-step so step size is controlled
    by lr alone, not by velocity magnitude.
    """

    name = "FM-ModeAttract"

    def __init__(self, model, device="cpu", n_steps=50, lr=0.01,
                 t_star=0.95):
        """
        Args:
            t_star: time level.  0.95 default.  Try 0.8 if modes merge.
            lr: step size (normalized direction, so this is actual
                distance moved per step in normalized y-space).
        """
        self.model = model
        self.device = device
        self.n_steps = n_steps
        self.lr = lr
        self.t_star = t_star

    @torch.no_grad()
    def _ascent_direction(self, y_norm, x, t_val):
        """Velocity as ascent direction, normalized to unit norm."""
        B = y_norm.shape[0]
        t = torch.full((B,), t_val, device=y_norm.device)
        v = self.model.velocity_net(y_norm, t, x)
        # Normalize so step size is purely controlled by self.lr
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return v / v_norm

    @torch.no_grad()
    def _gradient_ascent(self, x, y_norm):
        """Deterministic ascent using velocity field."""
        y_cur = y_norm.clone()
        for _ in range(self.n_steps):
            direction = self._ascent_direction(y_cur, x, self.t_star)
            y_cur = y_cur + self.lr * direction
        return y_cur

    def compute(self, x, y, batch_size=256):
        self.model.to(self.device).eval()
        n = x.shape[0]
        all_scores = []
        for i in range(0, n, batch_size):
            xb = x[i:i+batch_size].to(self.device)
            yb = y[i:i+batch_size].to(self.device)
            yb_norm = self.model._normalize(yb)
            y_star_norm = self._gradient_ascent(xb, yb_norm)
            s = ((yb_norm - y_star_norm) ** 2).sum(-1)
            all_scores.append(s.cpu())
        scores = torch.cat(all_scores).numpy()
        scores = np.nan_to_num(scores, nan=1e6, posinf=1e6, neginf=0.0)
        return scores

    def compute_on_grid(self, x_point, y_grid, batch_size=512, **kwargs):
        self.model.to(self.device).eval()
        M = y_grid.shape[0]
        all_scores = []
        for i in range(0, M, batch_size):
            yb = y_grid[i:i+batch_size].to(self.device)
            B = yb.shape[0]
            xb = x_point.unsqueeze(0).expand(B, -1).to(self.device)
            yb_norm = self.model._normalize(yb)
            y_star_norm = self._gradient_ascent(xb, yb_norm)
            s = ((yb_norm - y_star_norm) ** 2).sum(-1)
            all_scores.append(s.cpu())
        scores = torch.cat(all_scores).numpy()
        scores = np.nan_to_num(scores, nan=1e6, posinf=1e6, neginf=0.0)
        return scores