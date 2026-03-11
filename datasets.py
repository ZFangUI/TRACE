"""
Synthetic 2D datasets for conformal prediction experiments.
Follows CONTRA (Fang, Tan & Huang, ICLR 2025) design:

    Y = f(X) + ε(shape)

where f(X) is a shared nonlinear base function and ε has shape-specific
structure (spiral, ring, moon, mixture). Given x, p(y|x) has interesting
geometry determined by the noise shape.

X ∈ R^2, Y ∈ R^2.  All generators return (X, Y) as float32 tensors.
"""

import numpy as np
import torch


def _safe_mvn(rng, mu, cov, n):
    cov = np.array(cov, dtype=np.float64)
    cov = 0.5 * (cov + cov.T)
    eigv = np.linalg.eigvalsh(cov)
    if eigv.min() < 1e-8:
        cov += (1e-8 - eigv.min()) * np.eye(len(mu))
    return rng.multivariate_normal(mu, cov, size=n)


# ─── Parameters (single source of truth) ────────────────────────────

_PARAMS = {
    "mixture_gaussian": {
        "x_mean": [-2., -1.5], "x_cov_scale": 1.0,
        "components": [
            {"weight": 0.3, "mean": [0, 0], "cov_expr": "0.5*(I+J)"},
            {"weight": 0.4, "mean": [5, 5], "cov_expr": "1.5*(I-J)"},
            {"weight": 0.3, "mean": [10, 0], "cov_expr": "I"},
        ],
    },
    "spiral": {
        "x_mean": [-2., -1.5], "x_cov_scale": 1.0,
        "theta_max": 2 * np.pi,
        "noise_y1": 0.2, "noise_y2": 0.1,
    },
    "moon": {
        "x_mean": [-2., -1.5], "x_cov_scale": 1.0,
        "theta_max": np.pi,
        "noise": 0.1,
    },
    "ring": {
        "x_mean": [-2., -1.5], "x_cov_scale": 1.0,
        "r2_lo": 1.0, "r2_hi": 9.0,
    },
    "heterogeneous": {
        "x_mean": [-1., 0.], "x_cov_scale": 2.0,
        "base_scale": 0.3, "exp_rate": 0.5,
    },
    "banana": {
        "x_mean": [-1., 0.], "x_cov_scale": 2.0,
        "base_sigma": 0.3,          # std along the "spine"
        "ortho_sigma": 0.08,        # std orthogonal to the spine
        "curvature_rate": 0.3,      # how fast curvature grows with |x1|
    },
    "funnel": {
        "x_mean": [-1., 0.], "x_cov_scale": 2.0,
        "log_sigma_base": 0.0,      # log-scale at x=0
        "log_sigma_rate": 0.6,      # how fast log-scale grows with x1
        "corr_rate": 0.3,           # correlation strength depends on x2
    },
    # ── 2×2 ablation datasets ──────────────────────────────────────────
    # pinwheel_sim: f(x)  + pinwheel noise, X ∈ R^2
    # spiral_com:  5f(x) + spiral noise,   X ∈ R^7
    "pinwheel_sim": {
        "x_mean": [-2., -1.5], "x_cov_scale": 1.0,
        # pinwheel noise params (reused)
        "n_arms": 6, "radius": 3.0, "eccentricity": 0.16,
    },
    "spiral_com": {
        "x_dim": 7,
        "f_scale": 5.0,
        # spiral noise params (reused)
        "theta_max": 2 * np.pi,
        "noise_y1": 0.2, "noise_y2": 0.1,
    },
    # ── Higher-dim input datasets (Y = f_scale * f(X) + ε, Y ∈ R^2) ──
    "pinwheel": {
        "x_dim": 7,
        "f_scale": 5.0,                # f:ε ratio ~5:1
        "n_arms": 6,                    # number of Gaussian components
        "radius": 3.0,                  # distance of each mode from origin
        "eccentricity": 0.16,           # controls ellipse elongation (like the octagon paper)
    },
    "checkerboard": {
        "x_dim": 8,
        "f_scale": 2.5,                # f:ε ratio ~2.5:1
        "n_tiles": 4,                   # tiles per side (4×4 = 16 tiles)
        "tile_size": 1.0,               # side length of each tile
    },
    "twomoons": {
        "x_dim": 9,
        "f_scale": 4.5,                # f:ε ratio ~4.5:1
        "noise": 0.1,                   # additive Gaussian noise on moons
        "radius": 1.0,                  # moon radius
        "offset_x": 1.0,               # horizontal offset between moons
        "offset_y": -0.5,              # vertical offset between moons
    },
}


# ─── Base functions ──────────────────────────────────────────────────

def _y_base_nonlinear(x1, x2):
    """Base Y = f(X) for spiral, moon, ring."""
    y1d = 2*x1**3 - 3*x2**2 + 5*x2 + x1*x2
    y2d = x1**2*x2 - 4*x2**2 + 3*x1**2*x2 + 7
    return y1d, y2d


def _y_base_mixture(x1, x2):
    """Base Y = f(X) for mixture_gaussian."""
    y1d = 3*x1**3*x2 - 5*x2**2 + 4*x1*x2 - 6*x2 + 7
    y2d = x1*x2 - x2**3 + 3*x1*x2**2 + 8
    return y1d, y2d


def _y_base_pinwheel(X, scale=1.0):
    """f(X) for pinwheel: uses first 2 dims (rest are nuisance).  X ∈ R^7."""
    x1, x2 = X[:, 0], X[:, 1]
    y1 = 2*x1**3 - 3*x2**2 + 5*x2 + x1*x2
    y2 = x1**2*x2 - 4*x2**2 + 3*x1**2*x2 + 7
    return scale * y1, scale * y2


def _y_base_checkerboard(X, scale=1.0):
    """f(X) for checkerboard: uses first 2 dims (rest are nuisance).  X ∈ R^8."""
    x1, x2 = X[:, 0], X[:, 1]
    y1 = 2*x1**3 - 3*x2**2 + 5*x2 + x1*x2
    y2 = x1**2*x2 - 4*x2**2 + 3*x1**2*x2 + 7
    return scale * y1, scale * y2


def _y_base_twomoons(X, scale=1.0):
    """f(X) for twomoons: uses first 2 dims (rest are nuisance).  X ∈ R^9."""
    x1, x2 = X[:, 0], X[:, 1]
    y1 = 2*x1**3 - 3*x2**2 + 5*x2 + x1*x2
    y2 = x1**2*x2 - 4*x2**2 + 3*x1**2*x2 + 7
    return scale * y1, scale * y2


# Dispatch table for higher-dim base functions
_Y_BASE_HIGHDIM = {
    "pinwheel": _y_base_pinwheel,
    "spiral_com": _y_base_pinwheel,   # same f(X), 5× scale, spiral noise
    "checkerboard": _y_base_checkerboard,
    "twomoons": _y_base_twomoons,
}


# ─── Noise samplers ──────────────────────────────────────────────────

def _sample_noise(name, n, rng, x1=None, x2=None):
    """Sample shaped noise [n, 2] for a synthetic dataset."""
    p = _PARAMS[name]

    if name == "mixture_gaussian":
        I, J = np.eye(2), np.ones((2, 2))
        cov_map = {"0.5*(I+J)": 0.5*(I+J), "1.5*(I-J)": 1.5*(I-J), "I": I}
        weights = [c["weight"] for c in p["components"]]
        comp = rng.choice(len(weights), n, p=weights)
        eps = np.zeros((n, 2))
        for i, c in enumerate(p["components"]):
            mk = comp == i
            if mk.sum():
                eps[mk] = _safe_mvn(rng, c["mean"], cov_map[c["cov_expr"]],
                                    mk.sum())
        return eps

    if name == "spiral":
        th = rng.uniform(0, p["theta_max"], n)
        return np.column_stack([
            rng.normal(th * np.cos(th), p["noise_y1"]),
            rng.normal(th * np.sin(th), p["noise_y2"])])

    if name == "moon":
        th = rng.uniform(0, p["theta_max"], n)
        return np.column_stack([
            rng.normal(np.cos(th), p["noise"]),
            rng.normal(np.sin(th), p["noise"])])

    if name == "ring":
        r = np.sqrt(rng.uniform(p["r2_lo"], p["r2_hi"], n))
        th = rng.uniform(0, 2 * np.pi, n)
        return np.column_stack([r * np.cos(th), r * np.sin(th)])

    if name == "heterogeneous":
        # x-dependent noise: scale, anisotropy ratio, and rotation vary with x
        scale = p["base_scale"] * np.exp(p["exp_rate"] * x1)
        ratio = 1.0 + 2.0 * np.abs(x2)
        angle = 0.5 * (x1 + x2)
        e1 = rng.normal(0, scale * ratio)
        e2 = rng.normal(0, scale / ratio)
        noise1 = e1 * np.cos(angle) - e2 * np.sin(angle)
        noise2 = e1 * np.sin(angle) + e2 * np.cos(angle)
        return np.column_stack([noise1, noise2])

    if name == "banana":
        # Banana-shaped conditional distribution.
        # 1. Sample along a "spine" parameter t ~ N(0, base_sigma²)
        # 2. The spine is a parabola: (t, curvature * t²)
        # 3. Add orthogonal noise
        # 4. Curvature increases with |x1| → more bent for larger |x1|
        curvature = p["curvature_rate"] * (1.0 + np.abs(x1))
        t = rng.normal(0, p["base_sigma"], n)
        # Spine point: (t, curvature * t²)
        spine_y1 = t
        spine_y2 = curvature * t ** 2
        # Orthogonal direction to spine at each t: tangent = (1, 2*curvature*t)
        # Normal = (-2*curvature*t, 1) / norm
        tang_norm = np.sqrt(1 + (2 * curvature * t) ** 2)
        nx = -2 * curvature * t / tang_norm
        ny = 1.0 / tang_norm
        ortho = rng.normal(0, p["ortho_sigma"], n)
        noise1 = spine_y1 + ortho * nx
        noise2 = spine_y2 + ortho * ny
        return np.column_stack([noise1, noise2])

    if name == "funnel":
        # Neal's funnel variant: one dimension controls the scale of the other.
        # log(σ) varies linearly with x1 → exponential range of variances.
        # Correlation between y1 and y2 depends on x2.
        log_sigma = p["log_sigma_base"] + p["log_sigma_rate"] * x1
        sigma = np.exp(np.clip(log_sigma, -3, 3))  # clip to avoid extremes
        rho = np.tanh(p["corr_rate"] * x2)  # correlation in (-1, 1)
        # Sample (y1, y2) from bivariate normal with varying σ and ρ
        e1 = rng.normal(0, 1, n)
        e2 = rng.normal(0, 1, n)
        noise1 = sigma * e1
        noise2 = sigma * (rho * e1 + np.sqrt(1 - rho ** 2) * e2)
        return np.column_stack([noise1, noise2])

    if name == "pinwheel_sim":
        # f(x) + pinwheel noise, X ∈ R^2
        # Reuse pinwheel noise logic with pinwheel_sim params
        K = p["n_arms"]
        R = p["radius"]
        e = p["eccentricity"]
        comp = rng.choice(K, n)
        eps = np.zeros((n, 2))
        for i in range(K):
            mask = comp == i
            ni = mask.sum()
            if ni == 0:
                continue
            theta = i * np.pi / (K / 2)
            mu = np.array([R * np.cos(theta), R * np.sin(theta)])
            c, s = np.cos(theta), np.sin(theta)
            sig_11 = c**2 + e**2 * s**2
            sig_22 = s**2 + e**2 * c**2
            sig_12 = (1 - e**2) * s * c
            cov = np.array([[sig_11, sig_12],
                            [sig_12, sig_22]])
            eps[mask] = _safe_mvn(rng, mu, cov, ni)
        return eps

    if name == "spiral_com":
        # 5f(x) + spiral noise, X ∈ R^7
        # Reuse spiral noise logic with spiral_com params
        th = rng.uniform(0, p["theta_max"], n)
        return np.column_stack([
            rng.normal(th * np.cos(th), p["noise_y1"]),
            rng.normal(th * np.sin(th), p["noise_y2"])])

    if name == "pinwheel":
        # 6-component Gaussian mixture arranged in a hexagon.
        # Each component is an elongated ellipse pointing toward the origin,
        # following the octagon paper's parameterization:
        #   μ_i = (R·cos(θ_i), R·sin(θ_i))
        #   Σ_i = rotation(θ_i) @ diag(σ_long², σ_short²) @ rotation(θ_i)ᵀ
        # where σ_long = cos²(θ) + e²·sin²(θ), σ_short = sin²(θ) + e²·cos²(θ)
        # with e = eccentricity controlling elongation.
        K = p["n_arms"]
        R = p["radius"]
        e = p["eccentricity"]
        comp = rng.choice(K, n)
        eps = np.zeros((n, 2))
        for i in range(K):
            mask = comp == i
            ni = mask.sum()
            if ni == 0:
                continue
            theta = i * np.pi / (K / 2)  # = i * 2π/K
            mu = np.array([R * np.cos(theta), R * np.sin(theta)])
            # Covariance: elongated along radial direction
            c, s = np.cos(theta), np.sin(theta)
            sig_11 = c**2 + e**2 * s**2
            sig_22 = s**2 + e**2 * c**2
            sig_12 = (1 - e**2) * s * c
            cov = np.array([[sig_11, sig_12],
                            [sig_12, sig_22]])
            eps[mask] = _safe_mvn(rng, mu, cov, ni)
        return eps

    if name == "checkerboard":
        # 4x4 checkerboard: uniform in "black" tiles, vectorized rejection
        nt = p["n_tiles"]
        ts = p["tile_size"]
        collected = []
        remaining = n
        while remaining > 0:
            m = remaining * 3
            y1 = rng.uniform(0, nt * ts, m)
            y2 = rng.uniform(0, nt * ts, m)
            ti = (y1 / ts).astype(int) % nt
            tj = (y2 / ts).astype(int) % nt
            mask = (ti + tj) % 2 == 0
            good = np.column_stack([y1[mask], y2[mask]])
            take = min(remaining, len(good))
            collected.append(good[:take])
            remaining -= take
        eps = np.vstack(collected)
        eps[:, 0] -= nt * ts / 2
        eps[:, 1] -= nt * ts / 2
        return eps

    if name == "twomoons":
        # Two interleaving half-circles (sklearn-style)
        radius = p["radius"]
        noise = p["noise"]
        ox, oy = p["offset_x"], p["offset_y"]
        n_upper = n // 2
        n_lower = n - n_upper
        # Upper moon
        th_upper = rng.uniform(0, np.pi, n_upper)
        upper = np.column_stack([
            radius * np.cos(th_upper) + rng.normal(0, noise, n_upper),
            radius * np.sin(th_upper) + rng.normal(0, noise, n_upper)])
        # Lower moon (flipped and offset)
        th_lower = rng.uniform(0, np.pi, n_lower)
        lower = np.column_stack([
            radius * np.cos(th_lower) + ox + rng.normal(0, noise, n_lower),
            -(radius * np.sin(th_lower)) + oy + rng.normal(0, noise, n_lower)])
        eps = np.vstack([upper, lower])
        rng.shuffle(eps)
        return eps

    raise ValueError(f"Unknown dataset: {name}")


# ─── Generator functions ─────────────────────────────────────────────

def gen_mixture_gaussian(n, seed=0):
    """Mixture of 3 Gaussians: p(Y|X) is multi-modal (disconnected)."""
    p = _PARAMS["mixture_gaussian"]
    rng = np.random.RandomState(seed)
    X = _safe_mvn(rng, p["x_mean"], p["x_cov_scale"] * np.eye(2), n)
    x1, x2 = X[:, 0], X[:, 1]
    y1d, y2d = _y_base_mixture(x1, x2)
    eps = _sample_noise("mixture_gaussian", n, rng)
    Y = np.column_stack([y1d + eps[:, 0], y2d + eps[:, 1]])
    return torch.FloatTensor(X.astype(np.float32)), torch.FloatTensor(Y.astype(np.float32))


def gen_spiral(n, seed=0):
    """Spiral noise: p(Y|X) is a spiral-shaped distribution."""
    p = _PARAMS["spiral"]
    rng = np.random.RandomState(seed)
    X = _safe_mvn(rng, p["x_mean"], p["x_cov_scale"] * np.eye(2), n)
    x1, x2 = X[:, 0], X[:, 1]
    y1d, y2d = _y_base_nonlinear(x1, x2)
    eps = _sample_noise("spiral", n, rng)
    Y = np.column_stack([y1d + eps[:, 0], y2d + eps[:, 1]])
    return torch.FloatTensor(X.astype(np.float32)), torch.FloatTensor(Y.astype(np.float32))


def gen_moon(n, seed=0):
    """Moon/crescent noise: p(Y|X) is a half-circle shape."""
    p = _PARAMS["moon"]
    rng = np.random.RandomState(seed)
    X = _safe_mvn(rng, p["x_mean"], p["x_cov_scale"] * np.eye(2), n)
    x1, x2 = X[:, 0], X[:, 1]
    y1d, y2d = _y_base_nonlinear(x1, x2)
    eps = _sample_noise("moon", n, rng)
    Y = np.column_stack([y1d + eps[:, 0], y2d + eps[:, 1]])
    return torch.FloatTensor(X.astype(np.float32)), torch.FloatTensor(Y.astype(np.float32))


def gen_ring(n, seed=0):
    """Ring noise: p(Y|X) is an annulus with a hole. Tests topology."""
    p = _PARAMS["ring"]
    rng = np.random.RandomState(seed)
    X = _safe_mvn(rng, p["x_mean"], p["x_cov_scale"] * np.eye(2), n)
    x1, x2 = X[:, 0], X[:, 1]
    y1d, y2d = _y_base_nonlinear(x1, x2)
    eps = _sample_noise("ring", n, rng)
    Y = np.column_stack([y1d + eps[:, 0], y2d + eps[:, 1]])
    return torch.FloatTensor(X.astype(np.float32)), torch.FloatTensor(Y.astype(np.float32))


def gen_heterogeneous(n, seed=0):
    """Heterogeneous noise: scale, anisotropy, and rotation all depend on x.

    x1 < -2: tight isotropic noise (small σ)
    x1 ∈ [-2, 0]: elongated elliptical noise (anisotropic)
    x1 > 0: large diffuse noise (big σ)

    Tests methods under varying noise structure across x-space.
    """
    p = _PARAMS["heterogeneous"]
    rng = np.random.RandomState(seed)
    X = _safe_mvn(rng, p["x_mean"], p["x_cov_scale"] * np.eye(2), n)
    x1, x2 = X[:, 0], X[:, 1]
    y1d, y2d = _y_base_nonlinear(x1, x2)
    eps = _sample_noise("heterogeneous", n, rng, x1=x1, x2=x2)
    Y = np.column_stack([y1d + eps[:, 0], y2d + eps[:, 1]])
    return torch.FloatTensor(X.astype(np.float32)), torch.FloatTensor(Y.astype(np.float32))


def gen_banana(n, seed=0):
    """Banana-shaped conditional distribution, curvature varies with x.

    p(Y|X) is a curved, non-Gaussian shape:
    - |x1| small: nearly Gaussian (low curvature)
    - |x1| large: strongly banana-shaped (high curvature)

    Tests non-Gaussian conditional distributions and non-convex regions.
    """
    p = _PARAMS["banana"]
    rng = np.random.RandomState(seed)
    X = _safe_mvn(rng, p["x_mean"], p["x_cov_scale"] * np.eye(2), n)
    x1, x2 = X[:, 0], X[:, 1]
    y1d, y2d = _y_base_nonlinear(x1, x2)
    eps = _sample_noise("banana", n, rng, x1=x1, x2=x2)
    Y = np.column_stack([y1d + eps[:, 0], y2d + eps[:, 1]])
    return torch.FloatTensor(X.astype(np.float32)), torch.FloatTensor(Y.astype(np.float32))


def gen_funnel(n, seed=0):
    """Neal's funnel variant: variance changes exponentially with x.

    p(Y|X) is a bivariate Gaussian with x-dependent scale and correlation:
    - x1 controls log(σ): small x1 → tight, large x1 → diffuse
    - x2 controls correlation ρ between y1 and y2

    Tests extreme heteroscedasticity (variance spans orders of magnitude).
    """
    p = _PARAMS["funnel"]
    rng = np.random.RandomState(seed)
    X = _safe_mvn(rng, p["x_mean"], p["x_cov_scale"] * np.eye(2), n)
    x1, x2 = X[:, 0], X[:, 1]
    y1d, y2d = _y_base_nonlinear(x1, x2)
    eps = _sample_noise("funnel", n, rng, x1=x1, x2=x2)
    Y = np.column_stack([y1d + eps[:, 0], y2d + eps[:, 1]])
    return torch.FloatTensor(X.astype(np.float32)), torch.FloatTensor(Y.astype(np.float32))


# ─── 8D-input generators (X ∈ R^8, Y ∈ R^2) ─────────────────────────

def _gen_highdim(name, n, seed=0):
    """Shared generator for higher-dim input datasets.

    Y = f_scale * f(X) + ε,  X ~ N(0, I_d),  ε ~ named distribution.
    Each dataset has its own f function (different nonlinearities).
    """
    p = _PARAMS[name]
    x_dim = p["x_dim"]
    f_scale = p["f_scale"]
    rng = np.random.RandomState(seed)
    X = rng.randn(n, x_dim)
    y1d, y2d = _Y_BASE_HIGHDIM[name](X, scale=f_scale)
    eps = _sample_noise(name, n, rng)
    Y = np.column_stack([y1d + eps[:, 0], y2d + eps[:, 1]])
    return (torch.FloatTensor(X.astype(np.float32)),
            torch.FloatTensor(Y.astype(np.float32)))


def gen_pinwheel_sim(n, seed=0):
    """pinwheel_sim: f(x) + pinwheel noise, X ∈ R^2.

    2×2 ablation: same low-dim X as spiral, but complex multi-modal
    pinwheel noise. Isolates the effect of noise shape vs X dimensionality.
    """
    p = _PARAMS["pinwheel_sim"]
    rng = np.random.RandomState(seed)
    X = _safe_mvn(rng, p["x_mean"], p["x_cov_scale"] * np.eye(2), n)
    x1, x2 = X[:, 0], X[:, 1]
    y1d, y2d = _y_base_nonlinear(x1, x2)
    eps = _sample_noise("pinwheel_sim", n, rng)
    Y = np.column_stack([y1d + eps[:, 0], y2d + eps[:, 1]])
    return torch.FloatTensor(X.astype(np.float32)), torch.FloatTensor(Y.astype(np.float32))


def gen_spiral_com(n, seed=0):
    """spiral_com: 5f(x) + spiral noise, X ∈ R^7.

    2×2 ablation: same high-dim X as pinwheel, but simple single-modal
    spiral noise. Isolates the effect of X dimensionality vs noise shape.
    X ∈ R^7, Y ∈ R^2.
    """
    return _gen_highdim("spiral_com", n, seed)


def gen_pinwheel(n, seed=0):
    """Pinwheel noise (6 arms): p(Y|X) has 6-fold radial symmetry.

    Tests multi-modal distributions with radial structure.
    X ∈ R^7, Y ∈ R^2.
    """
    return _gen_highdim("pinwheel", n, seed)


def gen_checkerboard(n, seed=0):
    """Checkerboard noise (4×4): p(Y|X) is disconnected square regions.

    Tests disconnected prediction regions.
    X ∈ R^8, Y ∈ R^2.
    """
    return _gen_highdim("checkerboard", n, seed)


def gen_twomoons(n, seed=0):
    """Two moons noise: p(Y|X) is two interleaving half-circles.

    Tests non-convex, curved prediction regions.
    X ∈ R^9, Y ∈ R^2.
    """
    return _gen_highdim("twomoons", n, seed)


# ─── Conditional sampler (for visualization) ─────────────────────────

def sample_true_conditional(dataset_name, x_point, n=3000, seed=12345):
    """Sample from true p(Y|X=x). Uses same noise as generators.

    Args:
        dataset_name: key in DATASETS
        x_point: tensor [x_dim] or [1, x_dim]
        n: number of samples
        seed: random seed

    Returns:
        Y_samples: numpy array [n, 2]
    """
    if dataset_name not in DATASETS:
        return None

    x = (x_point.numpy().flatten() if isinstance(x_point, torch.Tensor)
         else np.asarray(x_point).flatten())
    rng = np.random.RandomState(seed)

    # Higher-dim datasets (pinwheel, checkerboard, twomoons)
    if dataset_name in _Y_BASE_HIGHDIM:
        p = _PARAMS[dataset_name]
        X_tile = np.tile(x, (n, 1))  # [n, x_dim]
        y1d, y2d = _Y_BASE_HIGHDIM[dataset_name](X_tile, scale=p["f_scale"])
        eps = _sample_noise(dataset_name, n, rng)
        return np.column_stack([y1d + eps[:, 0], y2d + eps[:, 1]])

    # 2D-input datasets
    x1, x2 = float(x[0]), float(x[1])
    x1_arr = np.full(n, x1)
    x2_arr = np.full(n, x2)

    if dataset_name == "mixture_gaussian":
        y1d, y2d = _y_base_mixture(x1, x2)
    else:
        y1d, y2d = _y_base_nonlinear(x1, x2)

    eps = _sample_noise(dataset_name, n, rng, x1=x1_arr, x2=x2_arr)
    return np.column_stack([y1d + eps[:, 0], y2d + eps[:, 1]])


# ─── Real-world datasets ────────────────────────────────────────────

import os


def _download_file(url, path):
    """Download a file from url to path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import urllib.request
    print(f"  Downloading {url} ...")
    urllib.request.urlretrieve(url, path)
    print(f"  Saved to {path}")


def _read_xlsx_simple(path):
    """Read a simple xlsx file using only stdlib (zipfile + xml.etree)."""
    import zipfile
    import xml.etree.ElementTree as ET
    with zipfile.ZipFile(path) as zf:
        shared = []
        if "xl/sharedStrings.xml" in zf.namelist():
            tree = ET.parse(zf.open("xl/sharedStrings.xml"))
            ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            for si in tree.findall(".//s:si", ns):
                t = si.find("s:t", ns)
                shared.append(t.text if t is not None and t.text else "")
        tree = ET.parse(zf.open("xl/worksheets/sheet1.xml"))
        ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        rows_data = []
        for row_el in tree.findall(".//s:row", ns):
            cells = []
            for c in row_el.findall("s:c", ns):
                v_el = c.find("s:v", ns)
                if v_el is None or v_el.text is None:
                    cells.append(np.nan); continue
                t = c.get("t", "")
                if t == "s":
                    idx = int(v_el.text)
                    try: cells.append(float(shared[idx]))
                    except (ValueError, IndexError): cells.append(np.nan)
                else:
                    try: cells.append(float(v_el.text))
                    except ValueError: cells.append(np.nan)
            if cells:
                rows_data.append(cells)
    if not rows_data:
        raise ValueError(f"No data found in {path}")
    max_cols = max(len(r) for r in rows_data)
    start = 0
    if len(rows_data) > 1 and any(np.isnan(v) for v in rows_data[0]):
        start = 1
    padded = []
    for r in rows_data[start:]:
        padded.append((r + [np.nan] * (max_cols - len(r)))[:max_cols])
    return np.array(padded, dtype=np.float64)


def _fetch_openml_to_numpy(data_id):
    """Fetch OpenML dataset."""
    try:
        from sklearn.datasets import fetch_openml
        ds = fetch_openml(data_id=data_id, as_frame=True, parser='auto')
        data = ds.data
        target = ds.target
        valid = data.dropna().index
        return data.loc[valid], target.loc[valid]
    except ImportError:
        raise RuntimeError("sklearn is required for OpenML datasets. "
                           "Install: pip install scikit-learn")


def load_bio(data_dir="./data", seed=42):
    """UCI CASP Protein: 9 features → 1 target (RMSD).

    Y is padded to 2D: Y[:,1] = small noise. Volume computed on dim 0 only.
    Returns: (X, Y_2d, y_std_prod, vol_dims=[0])
    """
    csv_path = os.path.join(data_dir, "bio", "CASP.csv")
    if not os.path.exists(csv_path):
        _download_file(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv",
            csv_path)
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    Y = data[:, 0:1].astype(np.float32)
    X = data[:, 1:].astype(np.float32)
    valid = ~(np.isnan(X).any(1) | np.isnan(Y).any(1))
    X, Y = X[valid], Y[valid]
    Y_std = Y.std(0)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    Y = (Y - Y.mean(0)) / (Y_std + 1e-8)
    # Pad to 2D with small noise
    rng = np.random.RandomState(seed)
    Y_pad = rng.randn(len(Y), 1).astype(np.float32) * 0.1
    Y_2d = np.concatenate([Y, Y_pad], axis=1)
    return (torch.FloatTensor(X), torch.FloatTensor(Y_2d),
            float(np.prod(Y_std)), [0])


def load_energy(data_dir="./data", seed=42):
    """UCI Energy Efficiency: 8 features → 2 targets."""
    csv_path = os.path.join(data_dir, "energy", "energy.csv")
    xlsx_path = os.path.join(data_dir, "energy", "ENB2012_data.xlsx")
    if os.path.exists(csv_path):
        data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    else:
        if not os.path.exists(xlsx_path):
            _download_file(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
                xlsx_path)
        data = _read_xlsx_simple(xlsx_path)
    X = data[:, :8].astype(np.float32)
    Y = data[:, 8:10].astype(np.float32)
    valid = ~(np.isnan(X).any(1) | np.isnan(Y).any(1))
    X, Y = X[valid], Y[valid]
    Y_std = Y.std(0)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    Y = (Y - Y.mean(0)) / (Y_std + 1e-8)
    return (torch.FloatTensor(X), torch.FloatTensor(Y),
            float(np.prod(Y_std)), None)


def load_taxi(data_dir="./data", seed=42, n_subset=6000, csv_path=None):
    """NYC Taxi: 2 features → 2 targets."""
    if csv_path is None:
        csv_path = os.path.join(data_dir, "taxi", "nyc.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"NYC Taxi CSV not found at: {csv_path}\n"
            f"Please copy nyc.csv there or pass --taxi_csv /path/to/nyc.csv")
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        X = df.iloc[:, 4:6].values.astype(np.float64)
        Y = df.iloc[:, 6:8].values.astype(np.float64)
    except ImportError:
        data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
        X = data[:, 4:6].astype(np.float64)
        Y = data[:, 6:8].astype(np.float64)
    valid = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
    X, Y = X[valid], Y[valid]
    for col in range(X.shape[1]):
        q1, q99 = np.percentile(X[:, col], [1, 99])
        valid = (X[:, col] >= q1) & (X[:, col] <= q99)
        X, Y = X[valid], Y[valid]
    for col in range(Y.shape[1]):
        q1, q99 = np.percentile(Y[:, col], [1, 99])
        valid = (Y[:, col] >= q1) & (Y[:, col] <= q99)
        X, Y = X[valid], Y[valid]
    rng = np.random.RandomState(seed)
    if len(X) > n_subset:
        idx = rng.choice(len(X), n_subset, replace=False)
        X, Y = X[idx], Y[idx]
    Y_mean = Y.mean(0)
    Y_std = Y.std(0)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    Y = (Y - Y_mean) / (Y_std + 1e-8)
    return (torch.FloatTensor(X.astype(np.float32)),
            torch.FloatTensor(Y.astype(np.float32)),
            float(np.prod(Y_std)), None,
            Y_mean.astype(np.float64), Y_std.astype(np.float64))


def load_rf1_2d(data_dir="./data", seed=42):
    """River Flow RF1: 20 features → 2 targets."""
    data, target = _fetch_openml_to_numpy(41483)
    X = data.iloc[:, 0:20].values.astype(np.float64)
    Y = target.iloc[:, [-3, -1]].values.astype(np.float64)
    valid = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
    X, Y = X[valid], Y[valid]
    Y_std = Y.std(0)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    Y = (Y - Y.mean(0)) / (Y_std + 1e-8)
    return (torch.FloatTensor(X.astype(np.float32)),
            torch.FloatTensor(Y.astype(np.float32)),
            float(np.prod(Y_std)), None)


def load_rf1_4d(data_dir="./data", seed=42):
    """River Flow RF1: 20 features → 4 targets."""
    data, target = _fetch_openml_to_numpy(41483)
    X = data.iloc[:, 0:20].values.astype(np.float64)
    Y = target.iloc[:, 1:5].values.astype(np.float64)
    valid = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
    X, Y = X[valid], Y[valid]
    Y_std = Y.std(0)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    Y = (Y - Y.mean(0)) / (Y_std + 1e-8)
    return (torch.FloatTensor(X.astype(np.float32)),
            torch.FloatTensor(Y.astype(np.float32)),
            float(np.prod(Y_std)), None)


def load_scm20d(data_dir="./data", seed=42, n_subset=5000):
    """SCM20D Supply Chain: 61 features → 2 targets."""
    data, target = _fetch_openml_to_numpy(41486)
    X = data.values.astype(np.float64)
    Y = target.iloc[:, [0, 1]].values.astype(np.float64)
    valid = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
    X, Y = X[valid], Y[valid]
    rng = np.random.RandomState(seed)
    if len(X) > n_subset:
        idx = rng.choice(len(X), n_subset, replace=False)
        X, Y = X[idx], Y[idx]
    Y_std = Y.std(0)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    Y = (Y - Y.mean(0)) / (Y_std + 1e-8)
    return (torch.FloatTensor(X.astype(np.float32)),
            torch.FloatTensor(Y.astype(np.float32)),
            float(np.prod(Y_std)), None)


def load_hurricane(data_dir="./data", seed=42, csv_path=None, lead_hours=6):
    """IBTrACS Hurricane: 8 features → 2 targets (displacement).

    X = [lat_now, lon_now, storm_speed, storm_dir, usa_wind, usa_pres,
         dist2land, usa_sshs]
    Y = [dlat, dlon]  (position displacement over lead_hours)

    Temporal dependence is handled via the Markov assumption:
    conditioning on the full kinematic state X_t absorbs the main
    temporal structure, so residual scores are approximately exchangeable.

    Returns: (X, Y, y_std_prod, None, Y_mean, Y_std)
    """
    import pandas as pd

    if csv_path is None:
        csv_path = os.path.join(data_dir, "hurricane",
                                "ibtracs.last3years.list.v04r01.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"IBTrACS CSV not found at: {csv_path}\n"
            f"Please download from https://www.ncei.noaa.gov/data/"
            f"international-best-track-archive-for-climate-stewardship-ibtracs/"
            f"v04r01/access/csv/\n"
            f"or pass --hurricane_csv /path/to/ibtracs.csv")

    df = pd.read_csv(csv_path, skiprows=[1, 2], low_memory=False,
                      na_values=[' ', ''])

    keep_cols = ['SID', 'ISO_TIME', 'LAT', 'LON', 'STORM_SPEED', 'STORM_DIR',
                 'USA_WIND', 'USA_PRES', 'DIST2LAND', 'USA_SSHS']
    df = df[keep_cols]

    # Convert types
    for col in ['LAT', 'LON', 'USA_WIND', 'USA_PRES']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'])
    df = df.sort_values(['SID', 'ISO_TIME']).reset_index(drop=True)

    # Determine lead in number of records (data is every 3 hours)
    lead_steps = lead_hours // 3

    # Build (X, Y) pairs: current state → future displacement
    records = []
    for sid, grp in df.groupby('SID'):
        grp = grp.reset_index(drop=True)
        for i in range(len(grp) - lead_steps):
            row_now = grp.iloc[i]
            row_fut = grp.iloc[i + lead_steps]
            # Verify time gap
            dt_h = (row_fut['ISO_TIME'] - row_now['ISO_TIME']).total_seconds() / 3600
            if abs(dt_h - lead_hours) > 1.5:
                continue
            records.append({
                'lat_now': row_now['LAT'],
                'lon_now': row_now['LON'],
                'storm_speed': row_now['STORM_SPEED'],
                'storm_dir': row_now['STORM_DIR'],
                'usa_wind': row_now['USA_WIND'],
                'usa_pres': row_now['USA_PRES'],
                'dist2land': row_now['DIST2LAND'],
                'usa_sshs': row_now['USA_SSHS'],
                'dlat': row_fut['LAT'] - row_now['LAT'],
                'dlon': row_fut['LON'] - row_now['LON'],
            })

    data = pd.DataFrame(records)

    # X columns and Y columns
    x_cols = ['lat_now', 'lon_now', 'storm_speed', 'storm_dir',
              'usa_wind', 'usa_pres', 'dist2land', 'usa_sshs']
    y_cols = ['dlat', 'dlon']

    # Drop rows with any NaN
    data = data.dropna(subset=x_cols + y_cols).reset_index(drop=True)

    X = data[x_cols].values.astype(np.float64)
    Y = data[y_cols].values.astype(np.float64)

    # Remove outliers (1st-99th percentile per column)
    for col in range(X.shape[1]):
        q1, q99 = np.percentile(X[:, col], [1, 99])
        valid = (X[:, col] >= q1) & (X[:, col] <= q99)
        X, Y = X[valid], Y[valid]
    for col in range(Y.shape[1]):
        q1, q99 = np.percentile(Y[:, col], [1, 99])
        valid = (Y[:, col] >= q1) & (Y[:, col] <= q99)
        X, Y = X[valid], Y[valid]

    # Normalize
    X_mean, X_std = X.mean(0), X.std(0)
    Y_mean, Y_std = Y.mean(0), Y.std(0)

    # Store original lat/lon stats for map plotting (before normalization)
    # We need the raw X lat/lon to place predictions on the map
    lat_col_idx, lon_col_idx = 0, 1  # lat_now, lon_now are first two X cols
    x_orig_mean = X_mean.copy()
    x_orig_std = X_std.copy()

    X = (X - X_mean) / (X_std + 1e-8)
    Y = (Y - Y_mean) / (Y_std + 1e-8)

    return (torch.FloatTensor(X.astype(np.float32)),
            torch.FloatTensor(Y.astype(np.float32)),
            float(np.prod(Y_std)), None,
            Y_mean.astype(np.float64), Y_std.astype(np.float64),
            x_orig_mean.astype(np.float64), x_orig_std.astype(np.float64))


# ─── Registry ────────────────────────────────────────────────────────

DATASETS = {
    "spiral": gen_spiral,
    "ring": gen_ring,
    "mixture_gaussian": gen_mixture_gaussian,
    "moon": gen_moon,
    "heterogeneous": gen_heterogeneous,
    "banana": gen_banana,
    "funnel": gen_funnel,
    "pinwheel_sim": gen_pinwheel_sim,       # 2×2 ablation: f(x)  + pinwheel noise, X∈R^2
    "spiral_com": gen_spiral_com,   # 2×2 ablation: 5f(x) + spiral noise,   X∈R^7
    "pinwheel": gen_pinwheel,
    "checkerboard": gen_checkerboard,
    "twomoons": gen_twomoons,
}

REAL_DATASETS = {
    "bio": load_bio,
    "energy": load_energy,
    "taxi": load_taxi,
    "rf1_2d": load_rf1_2d,
    "rf1_4d": load_rf1_4d,
    "scm20d": load_scm20d,
    "hurricane": load_hurricane,
}

# Datasets where we draw region plots (2D Y + interesting geometry)
PLOT_DATASETS = set(DATASETS.keys()) | {"taxi", "energy", "rf1_2d", "hurricane"}