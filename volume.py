"""
MC volume estimation for prediction regions in Y-space.

For NF-based scores:
    Vol(C(x)) = ∫_{C_z} |det(∂y/∂z)| dz
    where C_z = {z : s(z) ≤ τ} in latent space.
    MC: sample z uniformly in a bounding ball, check s(z) ≤ τ,
    map accepted z to y via NF inverse, accumulate |det(∂y/∂z)|.

For Diffusion / FM scores (no invertible mapping):
    Vol(C(x)) = ∫ 1[s(x,y) ≤ τ] dy
    MC: sample y on a grid or bounding box, check s(x,y) ≤ τ,
    accumulate volume of bounding box × acceptance rate.
"""

import numpy as np
import torch
from scipy.special import gammaln


@torch.no_grad()
def mc_volume_nf(model, score_fn_z, tau, x_points, device,
                 n_mc=5000, R_mult=2.0, seed=42):
    """MC volume in Y-space for NF-based scores (Ball, NLL).

    Vol_Y = (Vol(B_R) / M) × Σ_{z: s(z)≤τ} |det(∂y/∂z)|

    Args:
        model: NFModel with .inverse(x, z) → (y, log_det)
        score_fn_z: callable(z_batch, x_batch) → scores [batch]
        tau: calibrated threshold
        x_points: [n_x, x_dim] test x points
        device: torch device
        n_mc: MC samples per x
        R_mult: bounding ball radius multiplier
        seed: random seed for reproducibility

    Returns:
        mean_vol: average volume over x_points
        per_x_vols: list of volumes per x point
    """
    # Fix random seeds for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model.to(device).eval()
    x = x_points.to(device)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    n_x = x.shape[0]
    q = model.y_dim

    R = max(np.sqrt(abs(tau)), 1.0) * R_mult
    log_vol_ball = (q / 2) * np.log(np.pi) - gammaln(q / 2 + 1) + q * np.log(R)
    vol_ball = np.exp(log_vol_ball)

    per_x_vols = []
    for i in range(n_x):
        xi = x[i:i+1]

        # Sample z uniformly in B_R(0)
        u = torch.randn(n_mc, q, device=device)
        u = u / u.norm(dim=-1, keepdim=True)
        u = u * torch.rand(n_mc, 1, device=device).pow(1.0 / q)
        z_cand = R * u

        scores = score_fn_z(z_cand, xi.expand(n_mc, -1))
        inside = scores <= tau
        n_inside = inside.sum().item()

        if n_inside == 0:
            per_x_vols.append(0.0)
            continue

        z_in = z_cand[inside]
        xi_exp = xi.expand(z_in.shape[0], -1)
        _, log_jac_inv = model.inverse(xi_exp, z_in)
        det_values = torch.exp(log_jac_inv)
        vol_y = vol_ball * det_values.mean().item() * (n_inside / n_mc)
        per_x_vols.append(vol_y)

    mean_vol = float(np.mean(per_x_vols)) if per_x_vols else 0.0
    return mean_vol, per_x_vols


def mc_volume_grid(score_fn, tau, x_points, y_train, device,
                   n_mc=10000, margin=0.5, dataset_name=None,
                   gen_model=None, n_probe=500, verbose_name="",
                   seed=42, y_orig_mean=None, y_orig_std=None):
    """MC volume in Y-space via bbox Sobol (QMC) sampling.

    Uses **per-x local bounding box** for accurate estimation.
    Sobol quasi-random sequences give lower discrepancy than pseudo-random
    uniform, yielding more accurate volume estimates for same n_mc budget.

    Two-stage acceleration (automatic for CRN-based scores):
      Stage 1: Coarse screen with T_c timesteps, R_c repeats → reject ~80%
      Stage 2: Full precision on survivors only
    The coarse score uses a PREFIX of the same CRN bank, so it is a lower
    bound of the partial sum.  A safety margin ensures no false rejections
    at the boundary.

    Vol = Vol(BBox_x) × (# accepted / # total)

    Args:
        score_fn: score object with .compute_on_grid(x_point, y_grid)
        tau: calibrated threshold
        x_points: [n_x, x_dim]
        y_train: [n_train, y_dim] for fallback bounding box
        device: torch device
        n_mc: MC samples per x (rounded to power of 2 for Sobol)
        margin: fractional margin around local bbox
        dataset_name: if provided, use sample_true_conditional for local bbox
        gen_model: if provided, model with .sample_n(x, n) for local bbox
        n_probe: samples for determining local bbox
        verbose_name: method name for diagnostic print
        seed: random seed for reproducibility
    """
    from scipy.stats.qmc import Sobol

    # Fix random seeds for reproducibility (still needed for gen_model.sample_n)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    q = y_train.shape[1] if isinstance(y_train, np.ndarray) else y_train.shape[1]

    # Initialize Sobol sampler (deterministic given seed)
    sobol = Sobol(d=q, scramble=True, seed=seed)
    # Sobol needs power-of-2; use floor to avoid over-sampling
    m = int(np.floor(np.log2(max(n_mc, 2))))
    n_sobol = 2 ** m
    # If floor gives < 50% of n_mc, round up instead
    if n_sobol < n_mc * 0.6:
        n_sobol = 2 ** (m + 1)

    x = x_points
    if isinstance(x, torch.Tensor) and x.dim() == 1:
        x = x.unsqueeze(0)
    n_x = x.shape[0]

    # Try to import sample_true_conditional
    _sample_cond = None
    if dataset_name is not None:
        try:
            from datasets import sample_true_conditional
            _sample_cond = sample_true_conditional
        except ImportError:
            pass

    # Pre-generate Sobol samples in [0,1]^q (shared across x points)
    sobol_unit = sobol.random(n_sobol)  # [n_sobol, q] in [0,1]^q

    # ── Detect two-stage capability ──
    # CRN-based scores (DiffusionDenoiseScore, FMPathScore) have:
    #   .eps_bank or .y0_bank  [T, R, yd]
    #   .ts                    [T]
    #   .n_timesteps, .n_repeats
    _has_eps_bank = hasattr(score_fn, 'eps_bank')
    _has_y0_bank = hasattr(score_fn, 'y0_bank')
    _two_stage = (_has_eps_bank or _has_y0_bank) and hasattr(score_fn, 'ts')

    if _two_stage:
        T_full = score_fn.n_timesteps
        R_full = score_fn.n_repeats
        # Coarse: use ~20% of timesteps, ~25% of repeats (min 2 each)
        T_c = max(2, T_full // 5)
        R_c = max(2, R_full // 4)
        # Coarse score = sum of T_c*R_c terms / (T_full*R_full)
        # This is a strict LOWER BOUND of the final score (remaining terms >= 0).
        # So coarse_score > tau ⟹ full_score > tau, guaranteed.
        # No safety margin needed — this is exact.
        coarse_reject_tau = tau

        # Build coarse bank and timesteps (prefix slices of full CRN)
        if _has_eps_bank:
            coarse_bank = score_fn.eps_bank[:T_c, :R_c].to(device)
        else:
            coarse_bank = score_fn.y0_bank[:T_c, :R_c].to(device)
        coarse_ts = score_fn.ts[:T_c].to(device)
        full_bank = (score_fn.eps_bank if _has_eps_bank
                     else score_fn.y0_bank).to(device)
        full_ts = score_fn.ts.to(device)

    per_x_vols = []
    for i in range(n_x):
        xi = x[i]

        # Determine local bounding box for this x
        probe_y = None

        # Priority 1: true conditional (synthetic datasets)
        if _sample_cond is not None:
            probe_y = _sample_cond(dataset_name, xi, n=n_probe)
            # Normalize to match model space if Y was normalized
            if probe_y is not None and y_orig_mean is not None and y_orig_std is not None:
                probe_y = (probe_y - y_orig_mean) / (y_orig_std + 1e-8)

        # Priority 2: generative model sampling
        if probe_y is None and gen_model is not None:
            try:
                samples = gen_model.sample_n(xi.unsqueeze(0).to(device), n_probe)
                probe_y = samples.cpu().numpy()
            except Exception:
                pass

        # Priority 3: fallback to global bbox
        if probe_y is not None:
            lo = probe_y.min(axis=0)
            hi = probe_y.max(axis=0)
            span = hi - lo
            span = np.maximum(span, 1e-6)
            bbox_lo = lo - margin * span
            bbox_hi = hi + margin * span
        else:
            y_np = y_train.numpy() if isinstance(y_train, torch.Tensor) else y_train
            y_min = y_np.min(axis=0)
            y_max = y_np.max(axis=0)
            y_range = y_max - y_min
            bbox_lo = y_min - margin * y_range
            bbox_hi = y_max + margin * y_range

        vol_bbox = float(np.prod(bbox_hi - bbox_lo))

        # Map Sobol [0,1]^q → local bbox
        y_samples = sobol_unit * (bbox_hi - bbox_lo) + bbox_lo
        y_grid = torch.tensor(y_samples, dtype=torch.float32)

        if _two_stage:
            # ── Stage 1: Coarse screen ──
            # Compute partial score using prefix of CRN bank
            coarse_scores = _score_with_bank(
                score_fn, xi, y_grid, coarse_ts, coarse_bank, device,
                T_full, R_full)
            # Points whose lower-bound score already exceeds reject threshold
            # are definitely outside the region → skip expensive full eval
            survive_mask = coarse_scores <= coarse_reject_tau
            n_survive = int(survive_mask.sum())

            if n_survive == 0:
                # All rejected: volume contribution is 0
                per_x_vols.append(0.0)
                continue

            if n_survive < len(y_grid) * 0.9:
                # Enough rejected to be worth the two-stage overhead
                y_survivors = y_grid[survive_mask]
                fine_scores = _score_with_bank(
                    score_fn, xi, y_survivors, full_ts, full_bank, device,
                    T_full, R_full)
                n_accept = int((fine_scores <= tau).sum())
                # accept_rate = accepted / total (not / survivors)
                accept_rate = n_accept / len(y_grid)
            else:
                # Almost all survived → coarse didn't help, do full directly
                full_scores = _score_with_bank(
                    score_fn, xi, y_grid, full_ts, full_bank, device,
                    T_full, R_full)
                accept_rate = float((full_scores <= tau).mean())
        else:
            # ── Single-stage (NF, ODE, etc.) ──
            scores = score_fn.compute_on_grid(xi, y_grid)
            accept_rate = float((scores <= tau).mean())

        vol = vol_bbox * accept_rate
        per_x_vols.append(vol)

    mean_vol = float(np.mean(per_x_vols)) if per_x_vols else 0.0
    return mean_vol, per_x_vols


def _score_with_bank(score_fn, x_point, y_grid, ts, bank, device,
                     T_full, R_full):
    """Compute score using specific timesteps and CRN bank.

    The score is normalized by T_full * R_full (not by the bank's own
    dimensions), so that partial-bank scores are directly comparable to
    the full score and tau.

    Args:
        score_fn: DiffusionDenoiseScore or FMPathScore
        x_point: [x_dim] single x
        y_grid: [M, y_dim]
        ts: [T_c] timestep tensor (can be a prefix of full ts)
        bank: [T_c, R_c, y_dim] CRN bank (can be a prefix of full bank)
        device: torch device
        T_full, R_full: full dimensions for normalization
    Returns:
        scores: [M] numpy array, normalized by T_full * R_full
    """
    M = y_grid.shape[0]
    batch_size = 4096
    all_scores = []
    score_fn.model.to(device).eval()

    for i in range(0, M, batch_size):
        yb = y_grid[i:i+batch_size].to(device)
        B = yb.shape[0]
        xb = x_point.unsqueeze(0).expand(B, -1).to(device)

        if hasattr(score_fn, 'eps_bank'):
            s = score_fn.model.denoise_score(
                yb, xb, timesteps=ts,
                eps_bank=bank)
        else:
            s = score_fn.model.path_score(
                yb, xb, timesteps=ts,
                y0_bank=bank)

        all_scores.append(s.cpu().numpy())

    scores = np.concatenate(all_scores)

    # The model's denoise_score/path_score divides by len(ts)*bank.shape[1].
    # We need to re-normalize to T_full*R_full for comparable thresholding.
    T_c, R_c = len(ts), bank.shape[1]
    if T_c != T_full or R_c != R_full:
        # Undo the model's normalization, re-normalize by full T*R
        scores = scores * (T_c * R_c) / (T_full * R_full)

    return scores
