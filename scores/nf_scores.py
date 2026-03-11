"""
NF-based nonconformity scores:

1. NFBallScore: s(x,y) = ‖z‖²  where z = NF(y|x)
   - CONTRA score (Fang, Tan & Huang, ICLR 2025)
   - Prediction region is a ball in z-space → connected in y-space

2. NFNLLScore: s(x,y) = 0.5‖z‖² - log|det J|  = -log p(y|x) + const
   - JAPAN score (English et al., 2025)
   - Prediction region is a density level set → can be disconnected
"""

import numpy as np
import torch
from scipy.special import gammaln


class NFBallScore:
    """CONTRA: s(x,y) = ‖z‖² where z = NF(y|x).

    Prediction region: C(x) = {y : ‖NF(y|x)‖² ≤ τ}
    This is a ball in latent space → connected region in y-space.
    """

    name = "NF-Ball (CONTRA)"

    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device

    @torch.no_grad()
    def compute(self, x, y, batch_size=512):
        """Compute scores for (x, y) pairs.

        Returns: numpy array of scores [n].
        """
        self.model.to(self.device).eval()
        n = x.shape[0]
        scores = []
        for i in range(0, n, batch_size):
            xb = x[i:i+batch_size].to(self.device)
            yb = y[i:i+batch_size].to(self.device)
            z, _ = self.model(xb, yb)
            s = (z ** 2).sum(-1)
            scores.append(s.cpu())
        return torch.cat(scores).numpy()

    @torch.no_grad()
    def compute_on_grid(self, x_point, y_grid, **kwargs):
        """Compute scores for one x and a grid of y values."""
        self.model.to(self.device).eval()
        x_exp = x_point.unsqueeze(0).expand(y_grid.shape[0], -1).to(self.device)
        yg = y_grid.to(self.device)
        z, _ = self.model(x_exp, yg)
        return (z ** 2).sum(-1).cpu().numpy()

    def volume_latent(self, tau, y_dim):
        """Analytic volume of ball ‖z‖² ≤ τ in z-space."""
        r = np.sqrt(tau)
        log_v = ((y_dim / 2) * np.log(np.pi)
                 - gammaln(y_dim / 2 + 1) + y_dim * np.log(r))
        return np.exp(log_v)


class NFNLLScore:
    """JAPAN: s(x,y) = -log p(y|x) = 0.5‖z‖² - log|det J| + const.

    Prediction region: C(x) = {y : -log p(y|x) ≤ τ}
    This is a density level set → optimal volume by Neyman-Pearson.
    """

    name = "NF-NLL (JAPAN)"

    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device

    @torch.no_grad()
    def compute(self, x, y, batch_size=512):
        """Compute NLL scores. Returns numpy array [n]."""
        self.model.to(self.device).eval()
        n = x.shape[0]
        scores = []
        for i in range(0, n, batch_size):
            xb = x[i:i+batch_size].to(self.device)
            yb = y[i:i+batch_size].to(self.device)
            z, ld = self.model(xb, yb)
            # s = 0.5‖z‖² - log|det J|  (= -log p(y|x) + const)
            s = 0.5 * (z ** 2).sum(-1) - ld
            scores.append(s.cpu())
        return torch.cat(scores).numpy()

    @torch.no_grad()
    def compute_on_grid(self, x_point, y_grid, **kwargs):
        """Compute NLL scores for one x and a grid of y's."""
        self.model.to(self.device).eval()
        x_exp = x_point.unsqueeze(0).expand(y_grid.shape[0], -1).to(self.device)
        yg = y_grid.to(self.device)
        z, ld = self.model(x_exp, yg)
        s = 0.5 * (z ** 2).sum(-1) - ld
        return s.cpu().numpy()
