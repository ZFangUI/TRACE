"""
Visualize conditional p(Y|X=x) for pinwheel, checkerboard, twomoons.
Shows both the raw noise ε and the conditional distribution Y = f(x) + ε.

Usage:
    python viz_new_datasets.py
"""

import numpy as np
import matplotlib.pyplot as plt
from datasets import (
    _PARAMS, _Y_BASE_HIGHDIM, _sample_noise,
    _y_base_pinwheel, _y_base_checkerboard, _y_base_twomoons,
    sample_true_conditional, gen_pinwheel, gen_checkerboard, gen_twomoons,
)

n = 3000
seed = 42
rng = np.random.RandomState(seed)

fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for row, (name, gen_fn, f_fn) in enumerate([
    ("pinwheel", gen_pinwheel, _y_base_pinwheel),
    ("checkerboard", gen_checkerboard, _y_base_checkerboard),
    ("twomoons", gen_twomoons, _y_base_twomoons),
]):
    p = _PARAMS[name]
    x_dim = p["x_dim"]
    f_scale = p["f_scale"]

    # Col 0: raw noise ε (no f(x) shift)
    ax = axes[row, 0]
    rng_eps = np.random.RandomState(seed)
    eps = _sample_noise(name, n, rng_eps)
    ax.scatter(eps[:, 0], eps[:, 1], s=2, alpha=0.4)
    ax.set_title(f"{name} — raw ε", fontsize=12)
    ax.set_xlabel("ε₁")
    ax.set_ylabel("ε₂")
    ax.set_aspect("equal")

    # Col 1: conditional p(Y|X=x) for a random x
    ax = axes[row, 1]
    np.random.seed(seed)
    x_sample = np.random.randn(x_dim)  # one random x
    cond_y = sample_true_conditional(name, x_sample, n=n, seed=seed)
    ax.scatter(cond_y[:, 0], cond_y[:, 1], s=2, alpha=0.4)
    x_str = ", ".join(f"{v:.2f}" for v in x_sample[:3])
    f_val = f_fn(x_sample.reshape(1, -1), scale=f_scale)
    ax.set_title(f"{name} — p(Y|X=x)\nf(x)=({f_val[0][0]:.1f}, {f_val[1][0]:.1f})", fontsize=11)
    ax.set_xlabel("y₁")
    ax.set_ylabel("y₂")
    ax.set_aspect("equal")

    # Col 2: marginal Y (all data, many different x)
    ax = axes[row, 2]
    X, Y = gen_fn(n, seed=seed)
    ax.scatter(Y[:, 0].numpy(), Y[:, 1].numpy(), s=2, alpha=0.3)
    ax.set_title(f"{name} — marginal Y\n(mixed over X)", fontsize=12)
    ax.set_xlabel("y₁")
    ax.set_ylabel("y₂")
    ax.set_aspect("equal")

plt.tight_layout()
plt.savefig("viz_new_datasets.png", dpi=150, bbox_inches="tight")
print("Saved viz_new_datasets.png")
plt.close()