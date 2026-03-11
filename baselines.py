"""
Y-space baseline conformal prediction methods.

All baselines operate in Y-space (output space). They share a trained NF
for sampling p(Y|X) but apply different conformal score functions and
region shapes.

Methods
-------
RCP         — Residual Conformal Prediction (global ellipsoid in Y)
NLE         — Nonparametric Local Ellipsoidal (kNN local ellipsoid in Y)
PCP         — Probabilistic Conformal Prediction (union of balls in Y)
DistSplit   — Distribution-Split (axis-aligned rectangle in Y)
CQR         — Conformal Quantile Regression (Bonferroni rectangle in Y)
MCQR        — Multi-dim CQR with learned weights (weighted rectangle in Y)

Shared helpers
--------------
sample_ys_nf()    — Sample Y|X from a trained NF
predict_mean_nf() — Point prediction via NF sample mean
sample_ys_diff()  — Sample Y|X from a trained Diffusion model
sample_ys_fm()    — Sample Y|X from a trained FM model
"""

import math
import numpy as np
import torch
import torch.nn as nn

from conformal import conformal_quantile


# ============================================================================
# Shared helpers — sampling from generative models
# ============================================================================

@torch.no_grad()
def sample_ys_nf(model, x, n_samples=1000, device="cpu", batch_size=512,
                 seed=42):
    """Sample from p(Y|X) using a trained NF.

    Returns: [n, n_samples, y_dim] tensor (CPU).
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = model.to(device).eval()
    n = x.shape[0]
    all_samples = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = x[start:end].to(device)                # [B, xd]
        B = xb.shape[0]
        samp = model.sample(xb, n_samples)           # [B, S, yd]
        all_samples.append(samp.cpu())
    return torch.cat(all_samples, dim=0)              # [n, S, yd]


@torch.no_grad()
def sample_ys_diff(model, x, n_samples=1000, device="cpu",
                   batch_size=64, n_steps=100,
                   cfg_scale=1.0, cfg_mode="none",
                   sample_chunk=200, seed=42):
    """Sample from p(Y|X) using a trained Diffusion model.

    Returns: [n, n_samples, y_dim] tensor (CPU).
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = model.to(device).eval()
    n = x.shape[0]
    all_samples = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = x[start:end].to(device)              # [B, xd]
        B = xb.shape[0]
        chunk_samples = []
        remaining = n_samples
        while remaining > 0:
            ns = min(sample_chunk, remaining)
            xb_exp = xb.unsqueeze(1).expand(B, ns, -1).reshape(-1, xb.shape[1])
            y_samp = model.sample_ddim(xb_exp, n_steps=n_steps,
                                        cfg_scale=cfg_scale, cfg_mode=cfg_mode)
            chunk_samples.append(y_samp.cpu().reshape(B, ns, -1))
            remaining -= ns
        samp = torch.cat(chunk_samples, dim=1)     # [B, n_samples, yd]
        all_samples.append(samp)
    return torch.cat(all_samples, dim=0)


@torch.no_grad()
def sample_ys_fm(model, x, n_samples=1000, device="cpu",
                 batch_size=64, n_steps=100,
                 cfg_scale=1.0, cfg_mode="none", solver="midpoint",
                 sample_chunk=200, seed=42):
    """Sample from p(Y|X) using a trained FM model.

    Returns: [n, n_samples, y_dim] tensor (CPU).
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = model.to(device).eval()
    n = x.shape[0]
    all_samples = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = x[start:end].to(device)
        B = xb.shape[0]
        chunk_samples = []
        remaining = n_samples
        while remaining > 0:
            ns = min(sample_chunk, remaining)
            xb_exp = xb.unsqueeze(1).expand(B, ns, -1).reshape(-1, xb.shape[1])
            y_samp = model.sample(xb_exp, n_steps=n_steps,
                                  cfg_scale=cfg_scale, cfg_mode=cfg_mode,
                                  solver=solver)
            chunk_samples.append(y_samp.cpu().reshape(B, ns, -1))
            remaining -= ns
        samp = torch.cat(chunk_samples, dim=1)
        all_samples.append(samp)
    return torch.cat(all_samples, dim=0)


@torch.no_grad()
def predict_mean_nf(model, x, n_samples=1000, device="cpu", batch_size=64):
    """Point prediction: E[Y|X] ≈ mean of NF samples.

    Returns: [n, y_dim] tensor (CPU).
    """
    y_samples = sample_ys_nf(model, x, n_samples=n_samples,
                              device=device, batch_size=batch_size)
    return y_samples.mean(dim=1)


def _ellipsoid_volume(cov_matrix, tau, dim):
    """Analytic volume of {y: (y-c)^T Σ^{-1} (y-c) ≤ τ}.

    = V_d × √det(Σ) × τ^{d/2}  where V_d = π^{d/2}/Γ(d/2+1).
    """
    log_unit_ball = (dim / 2) * math.log(math.pi) - math.lgamma(dim / 2 + 1)
    try:
        L = torch.linalg.cholesky(cov_matrix)
        log_det_half = L.diag().log().sum().item()
    except Exception:
        log_det_half = 0.5 * torch.logdet(cov_matrix).item()
    log_vol = log_unit_ball + log_det_half + (dim / 2) * math.log(max(tau, 1e-30))
    return math.exp(log_vol)


# ============================================================================
# RCP — Residual Conformal Prediction
# ============================================================================

class RCP:
    """Residual Conformal Prediction: global ellipsoid in Y-space.

    Region:  C(x) = {y : (y - ŷ(x))^T Σ^{-1} (y - ŷ(x)) ≤ τ}
    where ŷ(x) = E_NF[Y|x], Σ = Cov(y_train - ŷ_train).
    """

    name = "RCP"

    def __init__(self, alpha=0.1, vol_dims=None):
        """
        Args:
            alpha:    error rate
            vol_dims: list of dimension indices for volume computation.
                      None = all dims.  e.g. [0] for bio (1D real target).
        """
        self.alpha = alpha
        self.vol_dims = vol_dims
        self.Sigma = None
        self.Sigma_inv = None
        self.tau = None
        self.dim = None

    def calibrate(self, cal_y, y_pred_cal, train_y, y_pred_train):
        """Calibrate.

        Args:
            cal_y:        [n_cal, q]
            y_pred_cal:   [n_cal, q] NF predicted means on cal
            train_y:      [n_train, q]
            y_pred_train: [n_train, q] NF predicted means on train
        """
        train_res = train_y.cpu() - y_pred_train.cpu()
        self.dim = train_res.shape[1]
        self.Sigma = torch.cov(train_res.T) + 1e-6 * torch.eye(self.dim)
        self.Sigma_inv = torch.linalg.inv(self.Sigma)

        cal_res = cal_y.cpu() - y_pred_cal.cpu()
        cal_scores = (cal_res @ self.Sigma_inv * cal_res).sum(dim=1).numpy()
        self.tau = conformal_quantile(cal_scores, self.alpha)
        return cal_scores

    def evaluate(self, test_y, y_pred_test):
        """Evaluate coverage and volume."""
        test_res = test_y.cpu() - y_pred_test.cpu()
        scores = (test_res @ self.Sigma_inv * test_res).sum(dim=1).numpy()
        cov = float((scores <= self.tau).mean())

        # Volume
        vd = self.vol_dims
        if vd is not None and len(vd) < self.dim:
            # Sub-dimensional volume (e.g. bio: 1D)
            d_sub = len(vd)
            Sigma_sub = self.Sigma[np.ix_(vd, vd)]
            vol = _ellipsoid_volume(Sigma_sub, self.tau, d_sub)
        else:
            vol = _ellipsoid_volume(self.Sigma, self.tau, self.dim)

        return {"coverage": cov, "tau": float(self.tau), "volume": vol,
                "score_mean": float(scores.mean()),
                "score_std": float(scores.std()),
                "scores": scores}

    def compute_on_grid(self, x_point, y_grid, y_pred_point):
        """For visualization: scores on a y-grid given one x.

        Args:
            x_point:      [x_dim] (unused, kept for interface consistency)
            y_grid:       [M, y_dim]
            y_pred_point: [y_dim] predicted mean for this x
        Returns:
            scores: [M] numpy array
        """
        residuals = y_grid.cpu() - y_pred_point.cpu().unsqueeze(0)
        scores = (residuals @ self.Sigma_inv * residuals).sum(dim=1).numpy()
        return scores


# ============================================================================
# NLE — Nonparametric Local Ellipsoidal
# ============================================================================

class NLE:
    """Nonparametric Local Ellipsoidal: kNN local ellipsoid in Y-space.

    Score: s(x,y) = √((y-ŷ)^T Σ_mix(x)^{-1} (y-ŷ))  [root Mahalanobis]
    """

    name = "NLE"

    def __init__(self, alpha=0.1, lam=0.9, k_frac=0.05, vol_dims=None):
        self.alpha = alpha
        self.lam = lam
        self.k_frac = k_frac
        self.vol_dims = vol_dims
        self.tau = None
        self.dim = None
        self.train_x = None
        self.train_res = None
        self.global_cov = None
        self.k = None

    def _knn_indices(self, query_x, pool_x, k):
        query_x = query_x.cpu()
        pool_x = pool_x.cpu()
        chunk = 256
        all_indices = []
        for i in range(0, len(query_x), chunk):
            dists = torch.cdist(query_x[i:i+chunk], pool_x)
            _, idx = dists.topk(k, largest=False, dim=1)
            all_indices.append(idx)
        return torch.cat(all_indices, dim=0)

    def _compute_scores(self, x, y, y_pred):
        residuals = y.cpu() - y_pred.cpu()
        nn_idx = self._knn_indices(x, self.train_x, self.k)
        n = x.shape[0]
        scores = np.zeros(n)
        for i in range(n):
            local_res = self.train_res[nn_idx[i]]
            local_cov = (torch.cov(local_res.T) if local_res.shape[0] > 1
                         else torch.eye(self.dim))
            Sigma_mix = self.lam * local_cov + (1 - self.lam) * self.global_cov
            Sigma_mix = Sigma_mix + 1e-6 * torch.eye(self.dim)
            Sigma_inv = torch.linalg.inv(Sigma_mix)
            r = residuals[i]
            scores[i] = math.sqrt(float((r @ Sigma_inv @ r).clamp(min=0)))
        return scores

    def calibrate(self, cal_x, cal_y, y_pred_cal,
                  train_x, train_y, y_pred_train):
        self.train_x = train_x.cpu()
        self.train_res = (train_y.cpu() - y_pred_train.cpu())
        self.dim = train_y.shape[1]
        self.global_cov = torch.cov(self.train_res.T) + 1e-6 * torch.eye(self.dim)
        self.k = max(5, int(self.k_frac * len(train_x)))

        cal_scores = self._compute_scores(cal_x, cal_y, y_pred_cal)
        self.tau = conformal_quantile(cal_scores, self.alpha)
        return cal_scores

    def evaluate(self, test_x, test_y, y_pred_test, n_vol=50):
        scores = self._compute_scores(test_x, test_y, y_pred_test)
        cov = float((scores <= self.tau).mean())

        n_vol_actual = min(n_vol, len(test_x))
        nn_idx = self._knn_indices(test_x[:n_vol_actual], self.train_x, self.k)
        total_vol = 0.0
        for i in range(n_vol_actual):
            local_res = self.train_res[nn_idx[i]]
            local_cov = (torch.cov(local_res.T) if local_res.shape[0] > 1
                         else torch.eye(self.dim))
            Sigma_mix = self.lam * local_cov + (1 - self.lam) * self.global_cov
            Sigma_mix = Sigma_mix + 1e-6 * torch.eye(self.dim)

            vd = self.vol_dims
            if vd is not None and len(vd) < self.dim:
                d_sub = len(vd)
                Sigma_sub = Sigma_mix[np.ix_(vd, vd)]
                total_vol += _ellipsoid_volume(Sigma_sub, self.tau ** 2, d_sub)
            else:
                total_vol += _ellipsoid_volume(Sigma_mix, self.tau ** 2, self.dim)

        avg_vol = total_vol / n_vol_actual

        return {"coverage": cov, "tau": float(self.tau), "volume": avg_vol,
                "score_mean": float(scores.mean()),
                "score_std": float(scores.std()),
                "scores": scores}

    def compute_on_grid(self, x_point, y_grid, y_pred_point):
        """For visualization."""
        # Use global cov for single-point grid (approximation)
        residuals = y_grid.cpu() - y_pred_point.cpu().unsqueeze(0)
        nn_idx = self._knn_indices(x_point.unsqueeze(0), self.train_x, self.k)
        local_res = self.train_res[nn_idx[0]]
        local_cov = (torch.cov(local_res.T) if local_res.shape[0] > 1
                     else torch.eye(self.dim))
        Sigma_mix = self.lam * local_cov + (1 - self.lam) * self.global_cov
        Sigma_mix = Sigma_mix + 1e-6 * torch.eye(self.dim)
        Sigma_inv = torch.linalg.inv(Sigma_mix)
        maha = (residuals @ Sigma_inv * residuals).sum(dim=1)
        return torch.sqrt(maha.clamp(min=0)).numpy()


# ============================================================================
# PCP — Probabilistic Conformal Prediction
# ============================================================================

class PCP:
    """Probabilistic Conformal Prediction: union of balls in Y-space.

    Score: s(x,y) = min_j ||y - ŷ_j(x)||,  where ŷ_j are model samples.
    Region: C(x) = ∪_j B(ŷ_j, τ)
    """

    def __init__(self, alpha=0.1, vol_dims=None, gen_name="Diff"):
        """
        Args:
            gen_name: "NF", "Diff", or "FM" — which model's samples to use
        """
        self.alpha = alpha
        self.vol_dims = vol_dims
        self.gen_name = gen_name
        self.name = f"PCP-{gen_name}"
        self.tau = None

    def calibrate(self, cal_y, cal_density):
        """
        Args:
            cal_y:        [n_cal, q]
            cal_density:  [n_cal, S, q] — pre-computed model samples
        """
        cal_y_cpu = cal_y.cpu()
        cal_density_cpu = cal_density.cpu()

        diffs = cal_density_cpu - cal_y_cpu.unsqueeze(1)
        dists = diffs.norm(dim=-1)
        min_dists = dists.min(dim=1).values

        scores_with_inf = torch.cat([min_dists, torch.tensor([float('inf')])])
        self.tau = conformal_quantile(scores_with_inf.numpy(), self.alpha)
        return min_dists.numpy()

    def evaluate(self, test_y, test_density, mc_trials=10000, n_vol=50):
        """Evaluate coverage and MC volume."""
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        test_y_cpu = test_y.cpu()
        test_density_cpu = test_density.cpu()
        q = test_y_cpu.shape[1]
        n_test = len(test_y_cpu)

        # Coverage
        diffs = test_density_cpu - test_y_cpu.unsqueeze(1)
        dists = diffs.norm(dim=-1)
        min_dists = dists.min(dim=1).values
        cov = float((min_dists <= self.tau).float().mean())

        # MC Volume
        vd = self.vol_dims
        n_vol_actual = min(n_vol, n_test)
        total_vol = 0.0
        for i in range(n_vol_actual):
            centers = test_density_cpu[i]  # [S, q]
            if vd is not None and len(vd) < q:
                # Sub-dim volume
                centers_sub = centers[:, vd]
                d_sub = len(vd)
                min_vals = centers_sub.min(dim=0).values - self.tau
                max_vals = centers_sub.max(dim=0).values + self.tau
                box_sides = max_vals - min_vals
                box_vol = box_sides.prod().item()
                rand_pts = torch.rand(mc_trials, d_sub) * box_sides + min_vals
                dist_to_c = torch.cdist(rand_pts, centers_sub)
                inside = (dist_to_c < self.tau).any(dim=1)
                total_vol += inside.float().mean().item() * box_vol
            else:
                min_vals = centers.min(dim=0).values - self.tau
                max_vals = centers.max(dim=0).values + self.tau
                box_sides = max_vals - min_vals
                box_vol = box_sides.prod().item()
                rand_pts = torch.rand(mc_trials, q) * box_sides + min_vals
                dist_to_c = torch.cdist(rand_pts, centers)
                inside = (dist_to_c < self.tau).any(dim=1)
                total_vol += inside.float().mean().item() * box_vol

        avg_vol = total_vol / n_vol_actual

        return {"coverage": cov, "tau": float(self.tau), "volume": avg_vol,
                "score_mean": float(min_dists.mean()),
                "score_std": float(min_dists.std()),
                "scores": min_dists.numpy()}

    def compute_on_grid(self, x_point, y_grid, density_samples):
        """For visualization.

        Args:
            x_point:          [x_dim] (unused)
            y_grid:           [M, y_dim]
            density_samples:  [S, y_dim] model samples for this x
        """
        diffs = y_grid.cpu().unsqueeze(1) - density_samples.cpu().unsqueeze(0)
        dists = diffs.norm(dim=-1)  # [M, S]
        min_dists = dists.min(dim=1).values
        return min_dists.numpy()


# ============================================================================
# DistSplit — Distribution-Split Conformal Prediction
# ============================================================================

class DistSplit:
    """Distribution-Split CP: axis-aligned rectangle via conditional CDF.

    Bonferroni correction across dimensions.
    """

    name = "DistSplit"

    def __init__(self, alpha=0.1, vol_dims=None):
        self.alpha = alpha
        self.vol_dims = vol_dims
        self.lower_q = None
        self.upper_q = None
        self.dim = None

    def calibrate(self, cal_y, cal_density):
        """
        Args:
            cal_y:        [n_cal, q]
            cal_density:  [n_cal, S, q]
        """
        cal_y_cpu = cal_y.cpu()
        cal_density_cpu = cal_density.cpu()
        self.dim = cal_y_cpu.shape[1]

        # Empirical CDF: P(Y_j < y_j | x)
        y_exp = cal_y_cpu.unsqueeze(1)
        p_matrix = (cal_density_cpu < y_exp).float().mean(dim=1)  # [n, q]

        # Bonferroni-corrected per-dim quantile levels
        self.lower_q = torch.quantile(p_matrix,
                                       self.alpha / (2 * self.dim), dim=0)
        self.upper_q = torch.quantile(p_matrix,
                                       1 - self.alpha / (2 * self.dim), dim=0)
        return p_matrix

    def evaluate(self, test_y, test_density):
        test_y_cpu = test_y.cpu()
        test_density_cpu = test_density.cpu()
        n_test = len(test_y_cpu)

        lower_bounds = torch.empty(n_test, self.dim)
        upper_bounds = torch.empty(n_test, self.dim)
        for j in range(self.dim):
            lower_bounds[:, j] = torch.quantile(
                test_density_cpu[:, :, j], self.lower_q[j].item(), dim=1)
            upper_bounds[:, j] = torch.quantile(
                test_density_cpu[:, :, j], self.upper_q[j].item(), dim=1)

        inside = ((lower_bounds <= test_y_cpu) &
                  (test_y_cpu <= upper_bounds)).all(dim=1)
        cov = float(inside.float().mean())

        # Volume
        widths = upper_bounds - lower_bounds
        vd = self.vol_dims
        if vd is not None and len(vd) < self.dim:
            volumes = widths[:, vd].prod(dim=1)
        else:
            volumes = widths.prod(dim=1)
        avg_vol = float(volumes.mean())

        return {"coverage": cov, "tau": 0.0, "volume": avg_vol,
                "score_mean": 0.0, "score_std": 0.0}

    def compute_on_grid(self, x_point, y_grid, density_samples):
        """For visualization: binary in/out.

        Args:
            density_samples: [S, y_dim] NF samples for this x
        """
        density_cpu = density_samples.cpu().unsqueeze(0)  # [1, S, q]
        M = y_grid.shape[0]
        scores = np.full(M, 2.0)  # default outside

        for j in range(self.dim):
            lq = torch.quantile(density_cpu[0, :, j],
                                self.lower_q[j].item())
            uq = torch.quantile(density_cpu[0, :, j],
                                self.upper_q[j].item())
            in_j = (y_grid[:, j] >= lq) & (y_grid[:, j] <= uq)
            scores[~in_j.numpy()] = 2.0

        # Inside = all dims in range → score 0
        all_in = np.ones(M, dtype=bool)
        for j in range(self.dim):
            lq = torch.quantile(density_cpu[0, :, j],
                                self.lower_q[j].item())
            uq = torch.quantile(density_cpu[0, :, j],
                                self.upper_q[j].item())
            all_in &= (y_grid[:, j].numpy() >= lq.item())
            all_in &= (y_grid[:, j].numpy() <= uq.item())
        scores[all_in] = 0.0
        return scores


# ============================================================================
# CQR — Conformal Quantile Regression (Bonferroni)
# ============================================================================

class CQR:
    """CQR with Bonferroni correction for multivariate Y."""

    name = "CQR"

    def __init__(self, alpha=0.1, vol_dims=None):
        self.alpha = alpha
        self.vol_dims = vol_dims
        self.tau_per_dim = None
        self.dim = None

    def calibrate(self, cal_y, cal_density):
        cal_y_cpu = cal_y.cpu()
        cal_density_cpu = cal_density.cpu()
        self.dim = cal_y_cpu.shape[1]

        y_low = torch.quantile(cal_density_cpu, self.alpha / 2, dim=1)
        y_up = torch.quantile(cal_density_cpu, 1 - self.alpha / 2, dim=1)

        bonf_alpha = 1 - (self.dim - self.alpha) / self.dim

        self.tau_per_dim = torch.zeros(self.dim)
        for j in range(self.dim):
            scores_j = torch.maximum(y_low[:, j] - cal_y_cpu[:, j],
                                     cal_y_cpu[:, j] - y_up[:, j])
            self.tau_per_dim[j] = conformal_quantile(scores_j.numpy(),
                                                      bonf_alpha)
        return y_low, y_up

    def evaluate(self, test_y, test_density):
        test_y_cpu = test_y.cpu()
        test_density_cpu = test_density.cpu()

        y_low = torch.quantile(test_density_cpu, self.alpha / 2, dim=1)
        y_up = torch.quantile(test_density_cpu, 1 - self.alpha / 2, dim=1)

        lower = y_low - self.tau_per_dim
        upper = y_up + self.tau_per_dim

        inside = ((lower <= test_y_cpu) & (test_y_cpu <= upper)).all(dim=1)
        cov = float(inside.float().mean())

        widths = upper - lower
        vd = self.vol_dims
        if vd is not None and len(vd) < self.dim:
            volumes = widths[:, vd].prod(dim=1)
        else:
            volumes = widths.prod(dim=1)
        avg_vol = float(volumes.mean())

        return {"coverage": cov, "tau": float(self.tau_per_dim.mean()),
                "volume": avg_vol, "score_mean": 0.0, "score_std": 0.0}

    def compute_on_grid(self, x_point, y_grid, density_samples):
        """For visualization: binary in/out."""
        density_cpu = density_samples.cpu()
        y_low = torch.quantile(density_cpu, self.alpha / 2, dim=0)
        y_up = torch.quantile(density_cpu, 1 - self.alpha / 2, dim=0)
        lower = y_low - self.tau_per_dim
        upper = y_up + self.tau_per_dim

        all_in = ((y_grid >= lower) & (y_grid <= upper)).all(dim=1)
        scores = np.where(all_in.numpy(), 0.0, 2.0)
        return scores


# ============================================================================
# MCQR / EMCQR — Multi-dim CQR with learned weights
# ============================================================================

class LinearWeightNN(nn.Module):
    """Global dimension weight network for MCQR."""

    def __init__(self, dim_x, num_weights):
        super().__init__()
        self.num_weights = num_weights
        self.n_learned = num_weights - 1
        if self.n_learned > 0:
            self.net = nn.Sequential(
                nn.Linear(dim_x, 128), nn.LeakyReLU(),
                nn.Linear(128, 64), nn.LeakyReLU(),
                nn.Linear(64, self.n_learned),
            )
        else:
            self.net = None

    def forward(self, x):
        fixed_w1 = torch.tensor([1.0], device=x.device)
        if self.net is None:
            return fixed_w1
        out = self.net(x).mean(dim=0)
        out = torch.exp(out)
        return torch.cat([fixed_w1, out], dim=0).flatten()


class MCQR:
    """Multi-dimensional CQR with learned per-dim weights.

    When equal_weight=True → EMCQR.
    """

    def __init__(self, alpha=0.1, device="cpu", equal_weight=False,
                 weight_epochs=300, weight_lr=1e-3, vol_dims=None):
        self.alpha = alpha
        self.device = device
        self.equal_weight = equal_weight
        self.weight_epochs = weight_epochs
        self.weight_lr = weight_lr
        self.vol_dims = vol_dims
        self.weights = None
        self.tau_weighted = None
        self.dim = None
        self.name = "EMCQR" if equal_weight else "MCQR"

    def calibrate(self, cal_y, cal_density, train_x):
        cal_y_cpu = cal_y.cpu()
        cal_density_cpu = cal_density.cpu()
        self.dim = cal_y_cpu.shape[1]
        x_dim = train_x.shape[1]

        y_low = torch.quantile(cal_density_cpu, self.alpha / 2, dim=1)
        y_up = torch.quantile(cal_density_cpu, 1 - self.alpha / 2, dim=1)

        n_w = self.dim if self.equal_weight else 2 * self.dim
        weight_net = LinearWeightNN(x_dim, n_w).to(self.device)
        optimizer = torch.optim.Adam(weight_net.parameters(), lr=self.weight_lr)

        train_x_dev = train_x.to(self.device)
        y_low_dev = y_low.to(self.device)
        y_up_dev = y_up.to(self.device)
        cal_y_dev = cal_y_cpu.to(self.device)

        for _ in range(self.weight_epochs):
            optimizer.zero_grad()
            w = weight_net(train_x_dev)
            w_full = w.repeat(2) if self.equal_weight else w

            low_diff = y_low_dev - cal_y_dev
            up_diff = cal_y_dev - y_up_dev
            res = torch.cat([low_diff, up_diff], dim=1)
            cal_scores = (res * w_full).max(dim=1).values

            n2 = len(cal_scores)
            sorted_scores = torch.sort(cal_scores).values
            qi = int(math.ceil((1 - self.alpha) * (n2 + 1))) - 1
            qi = min(max(qi, 0), n2 - 1)
            threshold = sorted_scores[qi]

            adjust = threshold / w_full
            low_bound = y_low_dev - adjust[:self.dim]
            up_bound = y_up_dev + adjust[self.dim:]
            lengths = up_bound - low_bound
            vol = lengths.prod(dim=1).mean()
            vol.backward()
            optimizer.step()

        weight_net.eval()
        with torch.no_grad():
            self.weights = weight_net(train_x_dev).cpu()
            w_full = self.weights.repeat(2) if self.equal_weight else self.weights

            low_diff = y_low - cal_y_cpu
            up_diff = cal_y_cpu - y_up
            res = torch.cat([low_diff, up_diff], dim=1)
            cal_scores = (res * w_full).max(dim=1).values
            self.tau_weighted = conformal_quantile(cal_scores.numpy(),
                                                    self.alpha)
        self._w_full = w_full

    def evaluate(self, test_y, test_density):
        test_y_cpu = test_y.cpu()
        test_density_cpu = test_density.cpu()

        y_low = torch.quantile(test_density_cpu, self.alpha / 2, dim=1)
        y_up = torch.quantile(test_density_cpu, 1 - self.alpha / 2, dim=1)

        adjust = self.tau_weighted / self._w_full
        lower = y_low - adjust[:self.dim]
        upper = y_up + adjust[self.dim:]

        inside = ((lower <= test_y_cpu) & (test_y_cpu <= upper)).all(dim=1)
        cov = float(inside.float().mean())

        widths = upper - lower
        vd = self.vol_dims
        if vd is not None and len(vd) < self.dim:
            volumes = widths[:, vd].prod(dim=1)
        else:
            volumes = widths.prod(dim=1)
        avg_vol = float(volumes.mean())

        return {"coverage": cov, "tau": float(self.tau_weighted),
                "volume": avg_vol, "score_mean": 0.0, "score_std": 0.0}

    def compute_on_grid(self, x_point, y_grid, density_samples):
        """For visualization: binary in/out."""
        density_cpu = density_samples.cpu()
        y_low = torch.quantile(density_cpu, self.alpha / 2, dim=0)
        y_up = torch.quantile(density_cpu, 1 - self.alpha / 2, dim=0)

        adjust = self.tau_weighted / self._w_full
        lower = y_low - adjust[:self.dim]
        upper = y_up + adjust[self.dim:]

        all_in = ((y_grid >= lower) & (y_grid <= upper)).all(dim=1)
        scores = np.where(all_in.numpy(), 0.0, 2.0)
        return scores


# ============================================================================
# KDE — Kernel Density Estimation Conformal Prediction
# ============================================================================

class KDE:
    """KDE-based conformal prediction: kNN density level set in Y-space.

    For each x, find k nearest neighbors in X-space from training set,
    use their Y values to build a Gaussian KDE, then:
        score(x, y) = -log kde(y|neighbors)

    Region: C(x) = {y : -log kde(y) <= tau}  (density level set)

    This is a nonparametric baseline that does not rely on any generative model.
    """

    name = "KDE"

    def __init__(self, alpha=0.1, k_frac=0.05, vol_dims=None):
        self.alpha = alpha
        self.k_frac = k_frac
        self.vol_dims = vol_dims
        self.tau = None
        self.dim = None
        self.train_x = None
        self.train_y = None
        self.k = None

    def _knn_indices(self, query_x, pool_x, k):
        query_x = query_x.cpu()
        pool_x = pool_x.cpu()
        chunk = 256
        all_indices = []
        for i in range(0, len(query_x), chunk):
            dists = torch.cdist(query_x[i:i+chunk], pool_x)
            _, idx = dists.topk(k, largest=False, dim=1)
            all_indices.append(idx)
        return torch.cat(all_indices, dim=0)

    def _kde_scores(self, x, y):
        """Compute -log kde(y) for each (x, y) pair using kNN in X-space."""
        from scipy.stats import gaussian_kde
        nn_idx = self._knn_indices(x, self.train_x, self.k)
        n = x.shape[0]
        scores = np.zeros(n)
        for i in range(n):
            neighbor_y = self.train_y[nn_idx[i]].numpy().T  # [dim, k]
            yi = y[i].cpu().numpy()
            try:
                kde = gaussian_kde(neighbor_y)
                density = kde(yi)
                scores[i] = -np.log(np.maximum(density[0], 1e-30))
            except np.linalg.LinAlgError:
                scores[i] = 1e10
        return scores

    def calibrate(self, cal_x, cal_y, train_x, train_y):
        """Calibrate.

        Args:
            cal_x:   [n_cal, x_dim]
            cal_y:   [n_cal, y_dim]
            train_x: [n_train, x_dim]
            train_y: [n_train, y_dim]
        """
        self.train_x = train_x.cpu()
        self.train_y = train_y.cpu()
        self.dim = cal_y.shape[1]
        self.k = max(10, int(self.k_frac * len(train_x)))

        cal_scores = self._kde_scores(cal_x.cpu(), cal_y.cpu())
        self.tau = conformal_quantile(cal_scores, self.alpha)
        return cal_scores

    def evaluate(self, test_x, test_y, n_vol_mc=5000, n_vol_x=50):
        """Evaluate coverage and volume."""
        scores = self._kde_scores(test_x.cpu(), test_y.cpu())
        cov = float((scores <= self.tau).mean())

        # Volume estimation via MC
        from scipy.stats import gaussian_kde
        n_vol_actual = min(n_vol_x, len(test_x))
        nn_idx = self._knn_indices(test_x[:n_vol_actual], self.train_x, self.k)
        total_vol = 0.0
        for i in range(n_vol_actual):
            neighbor_y = self.train_y[nn_idx[i]].numpy()
            ymin = neighbor_y.min(axis=0) - 2.0
            ymax = neighbor_y.max(axis=0) + 2.0
            box_vol = float(np.prod(ymax - ymin))

            mc_pts = np.random.uniform(ymin, ymax, size=(n_vol_mc, self.dim))
            try:
                kde = gaussian_kde(neighbor_y.T)
                log_densities = np.log(np.maximum(kde(mc_pts.T), 1e-30))
                neg_log_d = -log_densities
                frac_in = (neg_log_d <= self.tau).mean()
                total_vol += box_vol * frac_in
            except np.linalg.LinAlgError:
                pass

        avg_vol = total_vol / max(n_vol_actual, 1)

        return {"coverage": cov, "tau": float(self.tau), "volume": avg_vol,
                "score_mean": float(scores.mean()),
                "score_std": float(scores.std()),
                "scores": scores}

    def compute_on_grid(self, x_point, y_grid):
        """For visualization: scores on a y-grid given one x."""
        from scipy.stats import gaussian_kde
        nn_idx = self._knn_indices(x_point.unsqueeze(0), self.train_x, self.k)
        neighbor_y = self.train_y[nn_idx[0]].numpy().T  # [dim, k]
        try:
            kde = gaussian_kde(neighbor_y)
            y_np = y_grid.cpu().numpy().T  # [dim, M]
            densities = kde(y_np)
            scores = -np.log(np.maximum(densities, 1e-30))
        except np.linalg.LinAlgError:
            scores = np.full(len(y_grid), 1e10)
        return scores