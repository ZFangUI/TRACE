"""
Conditional Normalizing Flows for conformal prediction.

Two flow types:
  1. RealNVP  — affine coupling (fast, simple)
  2. NSF      — neural spline flow with rational-quadratic splines
               (more expressive per layer, better for complex distributions)

Architecture:
    x → ConditionNet → h
    y, h → CouplingLayers → z, log|det J|

Provides exact density: log p(y|x) = log p_Z(z) + log|det J|
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Shared: Condition network
# ============================================================================

class ConditionNet(nn.Module):
    """MLP: x → conditioning vector h."""

    def __init__(self, x_dim, cond_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, cond_dim),
        )

    def forward(self, x):
        return self.net(x)


class ConditionResNet(nn.Module):
    """ResNet: x → conditioning vector h.

    Same depth as ConditionNet but with residual connections
    for better gradient flow through high-dim inputs.
    """

    def __init__(self, x_dim, cond_dim, hidden_dim=256, n_blocks=3):
        super().__init__()
        self.proj_in = nn.Linear(x_dim, hidden_dim)
        blocks = []
        for _ in range(n_blocks):
            blocks.append(nn.Sequential(
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ))
        self.blocks = nn.ModuleList(blocks)
        self.proj_out = nn.Sequential(nn.ReLU(), nn.Linear(hidden_dim, cond_dim))

    def forward(self, x):
        h = self.proj_in(x)
        for block in self.blocks:
            h = h + block(h)
        return self.proj_out(h)


# ============================================================================
# RealNVP coupling layer (affine)
# ============================================================================

class AffineCouplingLayer(nn.Module):
    """Affine coupling: z₂ = y₂ ⊙ exp(s) + t."""

    def __init__(self, dim, cond_dim, hidden_dim=256, mask_type="even",
                 s_clamp=3.0):
        super().__init__()
        self.dim = dim
        self.s_clamp = s_clamp
        if mask_type == "even":
            self.register_buffer("mask", torch.arange(dim).float() % 2)
        else:
            self.register_buffer("mask", 1.0 - torch.arange(dim).float() % 2)

        inp = dim + cond_dim
        self.s_net = nn.Sequential(
            nn.Linear(inp, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, dim), nn.Tanh(),
        )
        self.t_net = nn.Sequential(
            nn.Linear(inp, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, y, cond):
        masked = y * self.mask
        h = torch.cat([masked, cond], dim=-1)
        s = self.s_net(h) * self.s_clamp * (1.0 - self.mask)
        t = self.t_net(h) * (1.0 - self.mask)
        z = y * torch.exp(s) + t
        log_det = s.sum(-1)
        return z, log_det

    def inverse(self, z, cond):
        masked = z * self.mask
        h = torch.cat([masked, cond], dim=-1)
        s = self.s_net(h) * self.s_clamp * (1.0 - self.mask)
        t = self.t_net(h) * (1.0 - self.mask)
        y = (z - t) * torch.exp(-s)
        log_det = -s.sum(-1)
        return y, log_det


# Keep old name for backward compatibility
CouplingLayer = AffineCouplingLayer


# ============================================================================
# Neural Spline Flow: Rational-Quadratic Spline coupling layer
# ============================================================================

def _rqs_forward(inputs, widths, heights, derivatives, tail_bound=3.0):
    """Rational-quadratic spline transform (forward).

    Based on Durkan et al. (2019) "Neural Spline Flows".

    Args:
        inputs:      [B, D] values to transform
        widths:      [B, D, K] unnormalized bin widths
        heights:     [B, D, K] unnormalized bin heights
        derivatives: [B, D, K+1] unnormalized derivatives at knot points
        tail_bound:  spline is identity outside [-tail_bound, tail_bound]

    Returns:
        outputs:  [B, D] transformed values
        log_det:  [B, D] per-dimension log|det|
    """
    K = widths.shape[-1]
    W = 2 * tail_bound

    widths = F.softmax(widths, dim=-1) * W            # [B, D, K], sum to W
    heights = F.softmax(heights, dim=-1) * W           # [B, D, K], sum to W
    derivatives = F.softplus(derivatives) + 1e-3       # [B, D, K+1], positive

    # Cumulative widths/heights → knot positions
    cum_widths = torch.cumsum(widths, dim=-1) - tail_bound    # [B, D, K]
    cum_heights = torch.cumsum(heights, dim=-1) - tail_bound

    cum_widths = F.pad(cum_widths, (1, 0), value=-tail_bound)    # [B, D, K+1]
    cum_heights = F.pad(cum_heights, (1, 0), value=-tail_bound)

    inside = (inputs >= -tail_bound) & (inputs <= tail_bound)
    clamped = inputs.clamp(-tail_bound + 1e-6, tail_bound - 1e-6)

    # Find bin index via searchsorted
    bin_idx = torch.searchsorted(cum_widths[..., 1:], clamped.unsqueeze(-1))
    bin_idx = bin_idx.squeeze(-1).clamp(0, K - 1)  # [B, D]

    # Gather bin parameters
    bi = bin_idx.unsqueeze(-1)
    x_k = cum_widths.gather(-1, bi).squeeze(-1)
    y_k = cum_heights.gather(-1, bi).squeeze(-1)
    w_k = widths.gather(-1, bi).squeeze(-1)
    h_k = heights.gather(-1, bi).squeeze(-1)
    d_k = derivatives.gather(-1, bi).squeeze(-1)
    d_k1 = derivatives.gather(-1, bi + 1).squeeze(-1)

    s_k = h_k / w_k

    # Normalized position within bin ∈ [0, 1]
    xi = ((clamped - x_k) / w_k).clamp(1e-6, 1 - 1e-6)

    # RQ spline formula
    num = h_k * (s_k * xi * xi + d_k * xi * (1 - xi))
    den = s_k + (d_k + d_k1 - 2 * s_k) * xi * (1 - xi)
    outputs = y_k + num / den

    # Log derivative
    dnum = s_k.square() * (d_k1 * xi.square()
                           + 2 * s_k * xi * (1 - xi)
                           + d_k * (1 - xi).square())
    log_det_in = torch.log(dnum.clamp(min=1e-8)) - 2 * torch.log(den.abs().clamp(min=1e-8))

    # Identity outside bounds
    outputs = torch.where(inside, outputs, inputs)
    log_det_out = torch.where(inside, log_det_in, torch.zeros_like(log_det_in))

    return outputs, log_det_out


def _rqs_inverse(inputs, widths, heights, derivatives, tail_bound=3.0):
    """Rational-quadratic spline transform (inverse).

    Analytic inverse via quadratic formula.
    """
    K = widths.shape[-1]
    W = 2 * tail_bound

    widths = F.softmax(widths, dim=-1) * W
    heights = F.softmax(heights, dim=-1) * W
    derivatives = F.softplus(derivatives) + 1e-3

    cum_widths = torch.cumsum(widths, dim=-1) - tail_bound
    cum_heights = torch.cumsum(heights, dim=-1) - tail_bound
    cum_widths = F.pad(cum_widths, (1, 0), value=-tail_bound)
    cum_heights = F.pad(cum_heights, (1, 0), value=-tail_bound)

    inside = (inputs >= -tail_bound) & (inputs <= tail_bound)
    clamped = inputs.clamp(-tail_bound + 1e-6, tail_bound - 1e-6)

    # For inverse, search in cumulative heights
    bin_idx = torch.searchsorted(cum_heights[..., 1:], clamped.unsqueeze(-1))
    bin_idx = bin_idx.squeeze(-1).clamp(0, K - 1)

    bi = bin_idx.unsqueeze(-1)
    x_k = cum_widths.gather(-1, bi).squeeze(-1)
    y_k = cum_heights.gather(-1, bi).squeeze(-1)
    w_k = widths.gather(-1, bi).squeeze(-1)
    h_k = heights.gather(-1, bi).squeeze(-1)
    d_k = derivatives.gather(-1, bi).squeeze(-1)
    d_k1 = derivatives.gather(-1, bi + 1).squeeze(-1)

    s_k = h_k / w_k

    # Solve quadratic for xi given output y
    a = h_k * (s_k - d_k) + (clamped - y_k) * (d_k + d_k1 - 2 * s_k)
    b = h_k * d_k - (clamped - y_k) * (d_k + d_k1 - 2 * s_k)
    c = -s_k * (clamped - y_k)

    disc = (b * b - 4 * a * c).clamp(min=0)
    xi = ((2 * c) / (-b - torch.sqrt(disc))).clamp(1e-6, 1 - 1e-6)

    outputs = xi * w_k + x_k

    # Log derivative (same formula as forward, evaluated at the found xi)
    den = s_k + (d_k + d_k1 - 2 * s_k) * xi * (1 - xi)
    dnum = s_k.square() * (d_k1 * xi.square()
                           + 2 * s_k * xi * (1 - xi)
                           + d_k * (1 - xi).square())
    log_det_in = torch.log(dnum.clamp(min=1e-8)) - 2 * torch.log(den.abs().clamp(min=1e-8))

    outputs = torch.where(inside, outputs, inputs)
    log_det_out = torch.where(inside, log_det_in, torch.zeros_like(log_det_in))

    # Inverse log_det = negative of forward log_det
    return outputs, -log_det_out


class SplineCouplingLayer(nn.Module):
    """Neural Spline Flow coupling layer with rational-quadratic splines.

    Transforms the non-masked dimensions using an RQ spline whose
    parameters are predicted from the masked dimensions + conditioning.
    """

    def __init__(self, dim, cond_dim, hidden_dim=256, mask_type="even",
                 n_bins=8, tail_bound=3.0):
        super().__init__()
        self.dim = dim
        self.n_bins = n_bins
        self.tail_bound = tail_bound

        if mask_type == "even":
            self.register_buffer("mask", torch.arange(dim).float() % 2)
        else:
            self.register_buffer("mask", 1.0 - torch.arange(dim).float() % 2)

        inp = dim + cond_dim
        # Per dim: K widths + K heights + (K+1) derivatives
        n_params_per_dim = 3 * n_bins + 1
        self.param_net = nn.Sequential(
            nn.Linear(inp, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, dim * n_params_per_dim),
        )

    def _get_spline_params(self, y_masked, cond):
        h = torch.cat([y_masked, cond], dim=-1)
        params = self.param_net(h)  # [B, dim * (3K+1)]
        B = params.shape[0]
        params = params.reshape(B, self.dim, 3 * self.n_bins + 1)

        K = self.n_bins
        widths = params[..., :K]
        heights = params[..., K:2*K]
        derivatives = params[..., 2*K:]
        return widths, heights, derivatives

    def forward(self, y, cond):
        y_masked = y * self.mask
        widths, heights, derivatives = self._get_spline_params(y_masked, cond)

        out, log_det_per_dim = _rqs_forward(
            y, widths, heights, derivatives, self.tail_bound)

        # Only transform and count log_det for non-masked dims
        out = y * self.mask + out * (1.0 - self.mask)
        log_det = (log_det_per_dim * (1.0 - self.mask)).sum(-1)
        return out, log_det

    def inverse(self, z, cond):
        z_masked = z * self.mask
        widths, heights, derivatives = self._get_spline_params(z_masked, cond)

        out, log_det_per_dim = _rqs_inverse(
            z, widths, heights, derivatives, self.tail_bound)

        out = z * self.mask + out * (1.0 - self.mask)
        log_det = (log_det_per_dim * (1.0 - self.mask)).sum(-1)
        return out, log_det


# ============================================================================
# Flow stack (works with either coupling type)
# ============================================================================

class ConditionalFlow(nn.Module):
    """Stack of coupling layers (RealNVP or NSF)."""

    def __init__(self, dim, cond_dim, hidden_dim=256, n_layers=6,
                 flow_type="realnvp", s_clamp=3.0, n_bins=8,
                 tail_bound=3.0):
        super().__init__()
        self.dim = dim
        layers = []
        for i in range(n_layers):
            mtype = "even" if i % 2 == 0 else "odd"
            if flow_type == "nsf":
                layers.append(SplineCouplingLayer(
                    dim, cond_dim, hidden_dim, mask_type=mtype,
                    n_bins=n_bins, tail_bound=tail_bound))
            else:
                layers.append(AffineCouplingLayer(
                    dim, cond_dim, hidden_dim, mask_type=mtype,
                    s_clamp=s_clamp))
        self.layers = nn.ModuleList(layers)

    def forward(self, y, cond):
        z, total_ld = y, torch.zeros(y.shape[0], device=y.device)
        for layer in self.layers:
            z, ld = layer(z, cond)
            total_ld += ld
        return z, total_ld

    def inverse(self, z, cond):
        y, total_ld = z, torch.zeros(z.shape[0], device=z.device)
        for layer in reversed(self.layers):
            y, ld = layer.inverse(y, cond)
            total_ld += ld
        return y, total_ld


# Keep old name for backward compatibility
ConditionalRealNVP = ConditionalFlow


# ============================================================================
# Full NF model
# ============================================================================

class NFModel(nn.Module):
    """Full NF model: ConditionNet + Flow (RealNVP or NSF).

    Provides:
        forward(x, y) → z, log_det
        inverse(x, z) → y, log_det
        log_prob(x, y) → log p(y|x)
        sample(x, n) → y samples
    """

    def __init__(self, x_dim, y_dim, cond_dim=128, hidden_dim=256,
                 n_layers=6, s_clamp=3.0, flow_type="realnvp",
                 n_bins=8, tail_bound=3.0, cond_net_type="mlp"):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        if cond_net_type == "resnet":
            self.cond_net = ConditionResNet(x_dim, cond_dim, hidden_dim)
        else:
            self.cond_net = ConditionNet(x_dim, cond_dim, hidden_dim)
        self.flow = ConditionalFlow(
            y_dim, cond_dim, hidden_dim, n_layers,
            flow_type=flow_type, s_clamp=s_clamp,
            n_bins=n_bins, tail_bound=tail_bound)

    def forward(self, x, y):
        """Forward: y → z, returns (z, log|det J|)."""
        h = self.cond_net(x)
        return self.flow(y, h)

    def inverse(self, x, z):
        """Inverse: z → y, returns (y, log|det J_inv|)."""
        h = self.cond_net(x)
        return self.flow.inverse(z, h)

    def log_prob(self, x, y):
        """Log-likelihood: log p(y|x) = log p_Z(z) + log|det J|."""
        z, ld = self.forward(x, y)
        log_pz = -0.5 * (z ** 2).sum(-1) - 0.5 * self.y_dim * np.log(2 * np.pi)
        return log_pz + ld

    def sample(self, x, n_samples=1):
        """Sample y ~ p(y|x) for each x.

        Args:
            x: [n, x_dim]
            n_samples: samples per x

        Returns:
            y: [n, n_samples, y_dim]
        """
        device = next(self.parameters()).device
        n = x.shape[0]
        h = self.cond_net(x)  # [n, cond_dim]

        z = torch.randn(n, n_samples, self.y_dim, device=device)
        h_exp = h.unsqueeze(1).expand(-1, n_samples, -1)

        z_flat = z.reshape(-1, self.y_dim)
        h_flat = h_exp.reshape(-1, h.shape[-1])
        y_flat, _ = self.flow.inverse(z_flat, h_flat)
        return y_flat.reshape(n, n_samples, self.y_dim)

    @torch.no_grad()
    def sample_n(self, x_point, n):
        """Sample n points from p(y|x) for a single x.

        Args:
            x_point: [1, x_dim] or [x_dim]
            n: number of samples

        Returns:
            [n, y_dim]
        """
        if x_point.dim() == 1:
            x_point = x_point.unsqueeze(0)
        device = next(self.parameters()).device
        x_point = x_point.to(device)
        samples = self.sample(x_point, n_samples=n)  # [1, n, y_dim]
        return samples.squeeze(0)  # [n, y_dim]
