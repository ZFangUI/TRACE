"""
Visualization: prediction regions, score distributions, comparison plots.

Region plot style: light colored fill + clear boundary contour +
small data scatter points. Clean, paper-ready.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def plot_prediction_regions_with_eval(predictors, x_point, data, y_range=None,
                                       grid_res=200, grid_n_avg=1,
                                       smooth_sigma=1.5,
                                       title=None, save_path=None,
                                       dataset_name=None,
                                       nf_model=None, device="cpu",
                                       y_true=None,
                                       y_orig_mean=None, y_orig_std=None):
    """Plot prediction regions with smooth boundaries.

    Layout:
      - Each Z-space method: individual panel
      - RCP + NLE: merged into one "Ellipsoid Baselines" panel
      - DistSplit + CQR + MCQR: merged into one "Rectangle Baselines" panel
      - PCP variants: individual panels

    Args:
        predictors: list of (ConformalPredictor, eval_results_dict)
        x_point: conditioning point [x_dim] tensor
        data: dict with "tsy" for fallback scatter
        grid_res: resolution of the y-grid
        grid_n_avg: MC averaging passes for stochastic scores (Diff/FM)
        smooth_sigma: Gaussian smoothing σ (in grid cells).
        dataset_name: if provided, sample true conditional for scatter/range
        nf_model: if provided, use for local y_range (real datasets)
        device: torch device
        y_true: [y_dim] true y for this x_point (plotted as ★ on each panel)
    """
    from datasets import sample_true_conditional, DATASETS
    from scipy.ndimage import gaussian_filter

    # ── Classify predictors into groups ──
    ELLIPSOID_NAMES = {"RCP", "NLE"}
    RECTANGLE_NAMES = {"DistSplit", "CQR", "MCQR", "EMCQR"}

    solo_preds = []        # individual panels (Z-space + PCP)
    ellipsoid_preds = []   # merged into one panel
    rectangle_preds = []   # merged into one panel

    for cp, res in predictors:
        name = cp.name
        if name in ELLIPSOID_NAMES:
            ellipsoid_preds.append((cp, res))
        elif name in RECTANGLE_NAMES:
            rectangle_preds.append((cp, res))
        else:
            solo_preds.append((cp, res))

    # Total panels: solo + (1 if ellipsoid) + (1 if rectangle) + (1 if true density)
    is_synthetic = (dataset_name is not None and dataset_name in DATASETS)
    n_panels = len(solo_preds)
    if ellipsoid_preds:
        n_panels += 1
    if rectangle_preds:
        n_panels += 1
    if is_synthetic:
        n_panels += 1

    if n_panels == 0:
        return

    # Get conditional samples for this x_point
    cond_samples = None
    if dataset_name is not None:
        cond_samples = sample_true_conditional(dataset_name, x_point, n=3000)
        # If Y was normalized in experiment, normalize samples to match
        if cond_samples is not None and y_orig_mean is not None and y_orig_std is not None:
            cond_samples = (cond_samples - y_orig_mean) / (y_orig_std + 1e-8)

    # Fallback: NF probe for real datasets
    if cond_samples is None and nf_model is not None:
        try:
            torch.manual_seed(42)
            nf_model.to(device).eval()
            with torch.no_grad():
                xi = x_point.unsqueeze(0).to(device)
                nf_samp = nf_model.sample(xi, 2000)
                yd = data["tsy"].shape[1]
                cond_samples = nf_samp.cpu().numpy().reshape(-1, yd)
        except Exception:
            pass

    if cond_samples is not None:
        scatter_y = cond_samples
    else:
        test_y = data["tsy"]
        scatter_y = test_y.numpy() if isinstance(test_y, torch.Tensor) else test_y

    yd = scatter_y.shape[1]
    if yd != 2:
        print(f"  [plot_regions] y_dim={yd} != 2, skipping contour plot.")
        return

    fig, axes = plt.subplots(1, n_panels,
                             figsize=(min(4.8 * n_panels, 38), 4.2),
                             squeeze=False)
    axes = axes[0]

    # y_range from conditional distribution (zoomed in)
    if y_range is None:
        margin = 0.25
        y1_min, y1_max = scatter_y[:, 0].min(), scatter_y[:, 0].max()
        y2_min, y2_max = scatter_y[:, 1].min(), scatter_y[:, 1].max()
        max_span = max(y1_max - y1_min, y2_max - y2_min)
        y1_c = (y1_min + y1_max) / 2
        y2_c = (y2_min + y2_max) / 2
        half = max_span / 2 * (1 + margin)
        y_range = ((y1_c - half, y1_c + half),
                   (y2_c - half, y2_c + half))

    y1_grid = np.linspace(y_range[0][0], y_range[0][1], grid_res)
    y2_grid = np.linspace(y_range[1][0], y_range[1][1], grid_res)
    Y1, Y2 = np.meshgrid(y1_grid, y2_grid)
    y_grid = torch.tensor(
        np.stack([Y1.ravel(), Y2.ravel()], axis=1), dtype=torch.float32
    )

    # Color palettes
    solo_fill =  ["#f5c6c6", "#b3d9f7", "#b3f0cc", "#d5b3f0",
                  "#fce3a8", "#f5d6c6", "#c6e8f5", "#d9f5c6",
                  "#f0d5b3", "#c6f5f5", "#f5c6e8", "#d9d5f0"]
    solo_edge =  ["#c0392b", "#2471a3", "#1e8449", "#7d3c98",
                  "#d68910", "#a04010", "#1a5276", "#27ae60",
                  "#b7950b", "#117a65", "#a93226", "#6c3483"]
    # Ellipsoid group colors
    ellip_fills = ["#c6e8f5", "#d5b3f0"]
    ellip_edges = ["#1a5276", "#7d3c98"]
    ellip_styles = ["-", "--"]
    # Rectangle group colors
    rect_fills  = ["#b3f0cc", "#fce3a8", "#f5c6c6"]
    rect_edges  = ["#1e8449", "#d68910", "#c0392b"]
    rect_styles = ["-", "--", ":"]

    def _draw_one(ax, cp, res, fc, ec, ls="-"):
        """Draw a single method's region on an axes."""
        _STOCH = ("Diff", "FM", "DDPM")
        is_stochastic = any(tag in cp.name for tag in _STOCH)
        kwargs = {"n_avg": grid_n_avg} if is_stochastic else {}
        inside, scores = cp.predict_grid(x_point, y_grid, **kwargs)
        score_grid = scores.reshape(grid_res, grid_res)

        tau = res["tau"]
        # Only smooth stochastic scores (Diff/FM have MC noise);
        # NF scores are deterministic — smoothing creates fake disconnections
        if smooth_sigma > 0 and is_stochastic:
            score_smooth = gaussian_filter(score_grid, sigma=smooth_sigma)
        else:
            score_smooth = score_grid
        region = (score_smooth <= tau).astype(float)

        ax.contourf(Y1, Y2, region, levels=[0.5, 1.5], colors=[fc], alpha=0.45)
        ax.contour(Y1, Y2, score_smooth, levels=[tau],
                   colors=[ec], linewidths=1.8, linestyles=[ls])

    def _format_ax(ax, idx, y_range):
        # Only show scatter for synthetic data (true conditional samples)
        if is_synthetic:
            n_show = min(500, len(scatter_y))
            ax.scatter(scatter_y[:n_show, 0], scatter_y[:n_show, 1],
                       s=4, c="#555555", alpha=0.25, edgecolors="none", zorder=5)
        # Plot true y as a star
        if y_true is not None:
            yt = y_true.numpy() if isinstance(y_true, torch.Tensor) else y_true
            ax.scatter(yt[0], yt[1], marker="*", s=120, c="#e74c3c",
                       edgecolors="k", linewidths=0.6, zorder=10)
        ax.set_xlabel("$y_1$", fontsize=10)
        if idx == 0:
            ax.set_ylabel("$y_2$", fontsize=10)
        ax.set_xlim(y_range[0])
        ax.set_ylim(y_range[1])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.15, linewidth=0.5)

    # ── Panel 0 (synthetic only): True Density heatmap ──
    panel_idx = 0
    if is_synthetic:
        ax = axes[panel_idx]
        # Sample many points from true conditional for KDE
        dense_samples = sample_true_conditional(dataset_name, x_point,
                                                n=20000, seed=99999)
        # Normalize to match model space
        if dense_samples is not None and y_orig_mean is not None and y_orig_std is not None:
            dense_samples = (dense_samples - y_orig_mean) / (y_orig_std + 1e-8)
        # 2D histogram on the grid, then smooth
        h, _, _ = np.histogram2d(dense_samples[:, 0], dense_samples[:, 1],
                                 bins=[y1_grid, y2_grid])
        h = h.T  # (y2, y1) layout to match imshow
        kde_sigma = max(1.5, grid_res / 80)
        h_smooth = gaussian_filter(h.astype(float), sigma=kde_sigma)
        # Pad to grid_res (histogram has bins-1 cells)
        h_plot = np.zeros((grid_res, grid_res))
        h_plot[:h_smooth.shape[0], :h_smooth.shape[1]] = h_smooth

        ax.imshow(h_plot, origin="lower", aspect="equal",
                  extent=[y_range[0][0], y_range[0][1],
                          y_range[1][0], y_range[1][1]],
                  cmap="magma", interpolation="bilinear")
        ax.set_title("True Density", fontsize=10, fontweight="bold",
                     color="#2c3e50")
        ax.set_xlabel("$y_1$", fontsize=10)
        ax.set_ylabel("$y_2$", fontsize=10)
        ax.set_xlim(y_range[0])
        ax.set_ylim(y_range[1])
        ax.grid(False)
        panel_idx += 1

    # ── Draw solo panels ──
    for i, (cp, res) in enumerate(solo_preds):
        ax = axes[panel_idx]
        fc = solo_fill[i % len(solo_fill)]
        ec = solo_edge[i % len(solo_edge)]

        _draw_one(ax, cp, res, fc, ec)
        _format_ax(ax, panel_idx, y_range)

        cov = res["coverage"]
        vol = res.get("volume", None)
        subtitle = f"cov={cov:.1%}  τ={res['tau']:.2f}"
        if vol is not None and vol > 0:
            subtitle += f"  vol={vol:.1f}"
        ax.set_title(f"{cp.name}\n{subtitle}", fontsize=10, color=ec)
        panel_idx += 1

    # ── Draw merged ellipsoid panel (RCP + NLE) ──
    if ellipsoid_preds:
        ax = axes[panel_idx]
        legend_items = []
        for j, (cp, res) in enumerate(ellipsoid_preds):
            fc = ellip_fills[j % len(ellip_fills)]
            ec = ellip_edges[j % len(ellip_edges)]
            ls = ellip_styles[j % len(ellip_styles)]
            _draw_one(ax, cp, res, fc, ec, ls=ls)
            cov = res["coverage"]
            vol = res.get("volume", None)
            lab = f"{cp.name} cov={cov:.1%}"
            if vol is not None and vol > 0:
                lab += f" vol={vol:.1f}"
            legend_items.append(plt.Line2D([0], [0], color=ec, ls=ls, lw=1.8,
                                           label=lab))
        _format_ax(ax, panel_idx, y_range)
        ax.legend(handles=legend_items, fontsize=8, loc="upper right")
        ax.set_title("Ellipsoid Baselines", fontsize=10, color="#333333")
        panel_idx += 1

    # ── Draw merged rectangle panel (DistSplit + CQR + MCQR) ──
    if rectangle_preds:
        ax = axes[panel_idx]
        legend_items = []
        for j, (cp, res) in enumerate(rectangle_preds):
            fc = rect_fills[j % len(rect_fills)]
            ec = rect_edges[j % len(rect_edges)]
            ls = rect_styles[j % len(rect_styles)]
            _draw_one(ax, cp, res, fc, ec, ls=ls)
            cov = res["coverage"]
            vol = res.get("volume", None)
            lab = f"{cp.name} cov={cov:.1%}"
            if vol is not None and vol > 0:
                lab += f" vol={vol:.1f}"
            legend_items.append(plt.Line2D([0], [0], color=ec, ls=ls, lw=1.8,
                                           label=lab))
        _format_ax(ax, panel_idx, y_range)
        ax.legend(handles=legend_items, fontsize=8, loc="upper right")
        ax.set_title("Rectangle Baselines", fontsize=10, color="#333333")
        panel_idx += 1

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_comparison_bars(all_results, title=None, save_path=None):
    """Bar chart comparing coverage, threshold, and volume across methods."""
    names = list(all_results.keys())
    n = len(names)

    coverages = [all_results[m]["coverage"] for m in names]
    taus = [all_results[m]["tau"] for m in names]
    volumes = [all_results[m].get("volume", 0) for m in names]
    has_volume = any(v > 0 for v in volumes)

    n_panels = 3 if has_volume else 2
    fig_w = max(5.5 * n_panels, 0.9 * n * n_panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 5))
    if n_panels == 1:
        axes = [axes]
    base_colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12",
                   "#1abc9c", "#e67e22", "#34495e", "#c0392b", "#2980b9",
                   "#27ae60", "#8e44ad", "#d35400", "#16a085", "#7f8c8d"]
    colors = [base_colors[i % len(base_colors)] for i in range(n)]
    rot = 35 if n > 6 else 25
    fs_tick = 8 if n > 8 else 9
    x_pos = np.arange(n)

    # Coverage
    ax = axes[0]
    bars = ax.bar(x_pos, coverages, color=colors, alpha=0.8, edgecolor="black")
    ax.axhline(0.9, color="black", linestyle="--", linewidth=1, label="1-α=0.9")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=rot, ha="right", fontsize=fs_tick)
    ax.set_ylabel("Coverage")
    ax.set_title("Marginal Coverage")
    ax.legend()
    ax.set_ylim(0.7, 1.0)
    for bar, val in zip(bars, coverages):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    # Threshold
    ax = axes[1]
    bars = ax.bar(x_pos, taus, color=colors, alpha=0.8, edgecolor="black")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=rot, ha="right", fontsize=fs_tick)
    ax.set_ylabel("Threshold τ")
    ax.set_title("Calibrated Threshold")
    for bar, val in zip(bars, taus):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(taus),
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    # Volume
    if has_volume:
        ax = axes[2]
        bars = ax.bar(x_pos, volumes, color=colors, alpha=0.8, edgecolor="black")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=rot, ha="right", fontsize=fs_tick)
        ax.set_ylabel("Avg Volume")
        ax.set_title("Average Prediction Region Volume")
        max_vol = max(v for v in volumes if v > 0) if any(v > 0 for v in volumes) else 1
        for bar, val in zip(bars, volumes):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.01 * max_vol,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_score_distributions(all_results, alpha=0.1, title=None, save_path=None):
    """Plot score distributions for each method (test set scores)."""
    names = [m for m in all_results if "scores" in all_results[m]]
    n = len(names)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    axes = axes[0]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]

    for idx, name in enumerate(names):
        ax = axes[idx]
        res = all_results[name]
        scores = res["scores"]
        tau = res["tau"]

        ax.hist(scores, bins=50, density=True, alpha=0.6,
                color=colors[idx % len(colors)], edgecolor="black", linewidth=0.5)
        ax.axvline(tau, color="red", linestyle="--", linewidth=2,
                   label=f"τ = {tau:.2f}")
        frac_below = (scores <= tau).mean()
        ax.set_title(f"{name}\n{frac_below:.1%} below τ", fontsize=11)
        ax.set_xlabel("Score")
        ax.legend(fontsize=9)

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


@torch.no_grad()
def plot_sample_quality(models, x_point, dataset_name, device="cpu",
                        n_samples=1000, save_path=None,
                        diff_cfg_scale=1.0, diff_cfg_mode="none",
                        diff_sample_steps=100,
                        fm_cfg_scale=1.0, fm_cfg_mode="none",
                        fm_solver="midpoint", fm_sample_steps=100,
                        y_orig_mean=None, y_orig_std=None):
    """Diagnostic: true p(Y|X=x) vs model-generated samples.

    Shows: True, NF, Diff (no CFG), Diff (dyn), FM (euler), FM (midpoint), FM (mid+dyn)
    """
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    from datasets import sample_true_conditional

    true_y = sample_true_conditional(dataset_name, x_point, n=n_samples)
    # Normalize to match model space
    if true_y is not None and y_orig_mean is not None and y_orig_std is not None:
        true_y = (true_y - y_orig_mean) / (y_orig_std + 1e-8)

    model_samples = {}
    x_dev = x_point.to(device)

    for name, model in models.items():
        model.to(device).eval()
        if name == "NF":
            samples = model.sample(x_dev.unsqueeze(0).expand(n_samples, -1),
                                   n_samples=1).squeeze()
            model_samples[name] = samples.cpu().numpy()
            # Use NF samples as reference if no true conditional
            if true_y is None:
                true_y = model_samples[name]
            if true_y is None:
                true_y = model_samples[name]
        elif name == "Diffusion":
            x_batch = x_dev.unsqueeze(0).expand(n_samples, -1)
            s1 = model.sample_ddim(x_batch, n_steps=diff_sample_steps,
                                   cfg_scale=1.0, cfg_mode="none")
            model_samples["Diff (no CFG)"] = s1.cpu().numpy()
            if diff_cfg_scale > 1.0:
                s2 = model.sample_ddim(x_batch, n_steps=diff_sample_steps,
                                       cfg_scale=diff_cfg_scale,
                                       cfg_mode=diff_cfg_mode)
                model_samples[f"Diff ({diff_cfg_mode} {diff_cfg_scale})"] = s2.cpu().numpy()
        elif name == "FM":
            x_batch = x_dev.unsqueeze(0).expand(n_samples, -1)
            # Euler baseline
            s1 = model.sample(x_batch, n_steps=fm_sample_steps,
                              cfg_scale=1.0, cfg_mode="none", solver="euler")
            model_samples["FM (euler)"] = s1.cpu().numpy()
            # Midpoint (recommended)
            s2 = model.sample(x_batch, n_steps=fm_sample_steps,
                              cfg_scale=1.0, cfg_mode="none", solver="midpoint")
            model_samples["FM (midpoint)"] = s2.cpu().numpy()
            if fm_cfg_scale > 1.0:
                s3 = model.sample(x_batch, n_steps=fm_sample_steps,
                                  cfg_scale=fm_cfg_scale,
                                  cfg_mode=fm_cfg_mode, solver="midpoint")
                model_samples[f"FM (mid+{fm_cfg_mode}{fm_cfg_scale})"] = s3.cpu().numpy()

    # Common y_range from true samples
    if true_y.ndim < 2 or true_y.shape[1] != 2:
        print(f"  [plot_sample_quality] y_dim={true_y.shape[-1]} != 2, skipping.")
        return None
    margin = 0.3
    y1_min, y1_max = true_y[:, 0].min(), true_y[:, 0].max()
    y2_min, y2_max = true_y[:, 1].min(), true_y[:, 1].max()
    max_span = max(y1_max - y1_min, y2_max - y2_min)
    y1_c = (y1_min + y1_max) / 2
    y2_c = (y2_min + y2_max) / 2
    half = max_span / 2 * (1 + margin)
    xlim = (y1_c - half, y1_c + half)
    ylim = (y2_c - half, y2_c + half)

    n_cols = 1 + len(model_samples)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 4.2))
    if n_cols == 1:
        axes = [axes]

    palette = {
        "True":            ("#2c3e50", "#95a5a6"),
        "NF":              ("#c0392b", "#f5c6c6"),
    }
    # Dynamic palette for other names
    extra_colors = [("#27ae60", "#b3f0cc"), ("#1a7a40", "#80d99e"),
                    ("#8e44ad", "#d5b3f0"), ("#6c2d82", "#b088d0"),
                    ("#4a1a60", "#9060b0"), ("#d68910", "#fce3a8")]

    # Check if we have true samples or NF fallback
    _has_true = sample_true_conditional(dataset_name, x_point, n=2) is not None

    ax = axes[0]
    ec, _ = palette["True"]
    ax.scatter(true_y[:, 0], true_y[:, 1], s=3, c=ec, alpha=0.4, edgecolors="none")
    if _has_true:
        ax.set_title(f"True $p(Y|X=x)$\n({n_samples} samples)", fontsize=11, color=ec)
    else:
        ax.set_title(f"NF reference $p(Y|X=x)$\n({n_samples} samples)", fontsize=11, color=ec)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.15)
    ax.set_xlabel("$y_1$"); ax.set_ylabel("$y_2$")

    for idx, (name, samples) in enumerate(model_samples.items()):
        ax = axes[idx + 1]
        if name in palette:
            ec, _ = palette[name]
        else:
            ec, _ = extra_colors[idx % len(extra_colors)]

        ax.scatter(samples[:, 0], samples[:, 1], s=3, c=ec, alpha=0.4,
                   edgecolors="none")
        in_range = ((samples[:, 0] >= xlim[0]) & (samples[:, 0] <= xlim[1]) &
                    (samples[:, 1] >= ylim[0]) & (samples[:, 1] <= ylim[1]))
        frac_in = in_range.mean()
        ax.set_title(f"{name} samples\n({frac_in:.0%} in range)", fontsize=11, color=ec)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.15)
        ax.set_xlabel("$y_1$")

    x_str = ", ".join(f"{v:.1f}" for v in x_point.tolist())
    fig.suptitle(f"Sample Quality Diagnostic — x = ({x_str})", fontsize=13, y=1.03)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig

# ═══════════════════════════════════════════════════════════════════
# Results Table (matplotlib table with color-coded rows)
# ═══════════════════════════════════════════════════════════════════

def plot_results_table(all_results, alpha=0.1, title=None, save_path=None,
                       time_dict=None):
    """Render a publication-quality results table as a figure.

    Rows are color-coded:
      - Green:  coverage ≥ 1-α AND among lowest volumes
      - Red:    coverage < 1-α
      - Yellow: coverage ≥ 1-α but high volume
      - White:  default

    Args:
        all_results: dict {method_name: {coverage, tau, volume, ...}}
        alpha: significance level (default 0.1)
        title: figure title
        save_path: if provided, save figure
        time_dict: optional dict {method_name: time_seconds}
    """
    target_cov = 1 - alpha
    names = list(all_results.keys())
    n = len(names)

    # Gather data
    coverages = [all_results[m]["coverage"] for m in names]
    volumes = [all_results[m].get("volume", float("nan")) for m in names]
    valid_vols = [v for v in volumes if not np.isnan(v) and v > 0]
    ref_vol = min(valid_vols) if valid_vols else 1.0

    # Build table data
    col_labels = ["Method", "Coverage", "Volume", "Rel. Vol", "Time(s)"]
    cell_text = []
    cell_colors = []

    for i, name in enumerate(names):
        cov = coverages[i]
        vol = volumes[i]
        rel_vol = vol / ref_vol if (ref_vol > 0 and not np.isnan(vol) and vol > 0) else float("nan")
        t = time_dict.get(name, float("nan")) if time_dict else float("nan")

        row = [
            name,
            f"{cov:.3f}",
            f"{vol:.2f}" if not np.isnan(vol) else "—",
            f"{rel_vol:.2f}x" if not np.isnan(rel_vol) else "—",
            f"{t:.1f}" if not np.isnan(t) else "—",
        ]
        cell_text.append(row)

        # Color logic
        if cov < target_cov - 0.005:
            # Under-coverage → red
            cell_colors.append(["#f8d7da"] * 5)
        elif not np.isnan(rel_vol) and rel_vol <= 1.5:
            # Good coverage + low volume → green
            cell_colors.append(["#d4edda"] * 5)
        elif not np.isnan(rel_vol) and rel_vol <= 3.0:
            # Good coverage + moderate volume → light yellow
            cell_colors.append(["#fff3cd"] * 5)
        else:
            cell_colors.append(["#ffffff"] * 5)

    # Create figure
    fig_h = max(2.0, 0.4 * n + 1.2)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        colColours=["#d6eaf8"] * 5,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    # Bold header
    for j in range(5):
        table[0, j].set_text_props(fontweight="bold")

    # Bold best methods
    if valid_vols:
        best_vol = min(valid_vols)
        for i, name in enumerate(names):
            vol = volumes[i]
            cov = coverages[i]
            if (cov >= target_cov - 0.005 and not np.isnan(vol)
                    and vol <= best_vol * 1.05):
                for j in range(5):
                    table[i + 1, j].set_text_props(fontweight="bold")

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=20)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════
# Pareto Frontier Plot (Coverage vs Volume)
# ═══════════════════════════════════════════════════════════════════

def plot_pareto(all_results, alpha=0.1, title=None, save_path=None):
    """Scatter plot of coverage vs volume with Pareto frontier highlighted.

    Args:
        all_results: dict {method_name: {coverage, tau, volume, ...}}
        alpha: significance level
        title: figure title
        save_path: if provided, save figure
    """
    target_cov = 1 - alpha
    names = list(all_results.keys())

    # Filter methods with valid volume
    pts = []
    for name in names:
        cov = all_results[name]["coverage"]
        vol = all_results[name].get("volume", float("nan"))
        if not np.isnan(vol) and vol > 0:
            pts.append((name, cov, vol))

    if not pts:
        return None

    # Separate into valid coverage and under-coverage
    valid = [(n, c, v) for n, c, v in pts if c >= target_cov - 0.005]
    invalid = [(n, c, v) for n, c, v in pts if c < target_cov - 0.005]

    # Find Pareto frontier among valid methods
    # Pareto optimal: no other method has both higher coverage AND lower volume
    pareto = []
    for n, c, v in valid:
        dominated = False
        for n2, c2, v2 in valid:
            if n2 != n and c2 >= c and v2 <= v and (c2 > c or v2 < v):
                dominated = True
                break
        if not dominated:
            pareto.append((n, c, v))

    # Sort Pareto points by volume for line
    pareto.sort(key=lambda x: x[2])

    # Color scheme
    method_colors = {
        "NF-Ball": "#e74c3c", "NF-NLL": "#c0392b",
        "Diff-Denoise": "#3498db", "FM-Path": "#2ecc71",
        "Diff-Quantile": "#5dade2", "FM-Quantile": "#58d68d",
        "RCP": "#9b59b6", "NLE": "#8e44ad",
        "PCP-Diff": "#f39c12", "DistSplit": "#d35400",
        "CQR": "#e67e22", "MCQR": "#c0392b",
    }
    default_color = "#7f8c8d"

    fig, ax = plt.subplots(figsize=(9, 6))

    # Plot invalid (under-coverage) as gray X
    for name, cov, vol in invalid:
        ax.scatter(vol, cov, marker="x", s=80, c="#cccccc", zorder=2)
        ax.annotate(name, (vol, cov), fontsize=7, color="#999999",
                    textcoords="offset points", xytext=(5, 3))

    # Plot valid methods
    for name, cov, vol in valid:
        color = method_colors.get(name, default_color)
        is_pareto = any(n == name for n, _, _ in pareto)
        marker = "★" if is_pareto else "o"
        size = 120 if is_pareto else 60
        edge = "black" if is_pareto else "none"
        ax.scatter(vol, cov, s=size, c=color, edgecolors=edge,
                   linewidths=1.5 if is_pareto else 0, zorder=3, marker="o")
        ax.annotate(name, (vol, cov), fontsize=8, color=color,
                    textcoords="offset points", xytext=(5, 5),
                    fontweight="bold" if is_pareto else "normal")

    # Draw Pareto frontier line
    if len(pareto) > 1:
        pv = [v for _, _, v in pareto]
        pc = [c for _, c, _ in pareto]
        ax.plot(pv, pc, "k--", alpha=0.5, linewidth=1.5, label="Pareto frontier")

    # Target coverage line
    ax.axhline(target_cov, color="red", linestyle=":", alpha=0.6,
               label=f"Target coverage (1-α={target_cov})")

    ax.set_xlabel("Volume", fontsize=12)
    ax.set_ylabel("Coverage", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.2)

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════
# Repeat Summary: Violin + Strip Plot
# ═══════════════════════════════════════════════════════════════════

def plot_repeats_violin(all_repeats, alpha=0.1, title=None, save_path=None):
    """Violin + jittered strip plot for multi-repeat experiments.

    Two panels: Coverage (left) and Volume (right).
    Each method gets a violin showing the distribution of repeat results,
    with individual points overlaid and median marked.

    Args:
        all_repeats: dict {method_name: {"coverage": [list], "volume": [list]}}
        alpha: significance level (for target coverage line)
        title: figure title
        save_path: if provided, save figure
    """
    names = list(all_repeats.keys())
    n = len(names)
    if n == 0:
        return None

    target_cov = 1 - alpha

    # Colors (consistent with other plots)
    base_colors = [
        "#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12",
        "#1abc9c", "#e67e22", "#34495e", "#c0392b", "#2980b9",
        "#27ae60", "#8e44ad", "#d35400", "#16a085", "#7f8c8d",
    ]
    colors = [base_colors[i % len(base_colors)] for i in range(n)]

    fig, axes = plt.subplots(1, 2, figsize=(max(7, 0.9 * n + 3), 5))

    for panel_idx, (metric, ylabel) in enumerate([
        ("coverage", "Coverage"), ("volume", "Volume")
    ]):
        ax = axes[panel_idx]
        data_list = []
        positions = []
        valid_names = []
        valid_colors = []

        for i, name in enumerate(names):
            vals = all_repeats[name][metric]
            vals_clean = [v for v in vals if not np.isnan(v)]
            if len(vals_clean) >= 2:
                data_list.append(vals_clean)
                positions.append(i)
                valid_names.append(name)
                valid_colors.append(colors[i])

        if not data_list:
            ax.set_title(ylabel)
            continue

        # Violin
        parts = ax.violinplot(
            data_list, positions=positions,
            showmeans=False, showmedians=False, showextrema=False,
        )
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(valid_colors[i])
            pc.set_edgecolor("black")
            pc.set_linewidth(0.8)
            pc.set_alpha(0.55)

        # Median line
        for i, vals in enumerate(data_list):
            med = np.median(vals)
            pos = positions[i]
            ax.hlines(med, pos - 0.25, pos + 0.25,
                      colors="black", linewidths=2.0, zorder=4)

        # Jittered strip (individual repeat results)
        rng = np.random.RandomState(42)
        for i, vals in enumerate(data_list):
            pos = positions[i]
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(
                pos + jitter, vals,
                s=18, c=valid_colors[i], edgecolors="white",
                linewidths=0.5, alpha=0.75, zorder=5,
            )

        # Target coverage line
        if metric == "coverage":
            ax.axhline(target_cov, color="red", linestyle="--",
                       linewidth=1.2, alpha=0.7, label=f"1-α={target_cov}")
            ax.legend(fontsize=9, loc="lower left")
            ax.set_ylim(max(0.6, min(min(v) for v in data_list) - 0.05), 1.02)

        ax.set_xticks(range(n))
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, axis="y", alpha=0.2, linewidth=0.5)
        ax.set_title(ylabel, fontsize=12, fontweight="bold")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════
# Taxi Map: Prediction regions overlaid on NYC map via Folium
# ═══════════════════════════════════════════════════════════════════

def plot_taxi_map(pred_list, x_point, y_orig_mean, y_orig_std,
                  all_results, x_orig_mean=None, x_orig_std=None,
                  y_range=None, y_true=None,
                  train_x=None, train_y=None,
                  kde_k=200, kde_bandwidth=0.005,
                  grid_res=150, grid_n_avg=3,
                  smooth_sigma=1.5, save_dir=None, prefix="taxi"):
    """Draw prediction regions on NYC map (per-method PNGs).

    Args:
        pred_list: list of (predictor_with_predict_grid, eval_results_dict)
        x_point: [x_dim] tensor, conditioning point (normalized)
        y_orig_mean/std: [2] numpy, Y normalization params
        all_results: dict
        x_orig_mean/std: [2] numpy, X normalization params (for pickup marker)
        y_range: ((y1_lo, y1_hi), (y2_lo, y2_hi)) in normalized space, or None
        y_true: [y_dim] tensor, true Y for this test point (normalized)
        train_x/train_y: [n, dim] tensors, training data (for KDE baseline)
        kde_k: number of neighbors for KDE
        kde_bandwidth: KDE bandwidth in original coordinate units
    """
    from scipy.ndimage import gaussian_filter

    if not pred_list:
        print("  [taxi_map] No methods to plot.")
        return []

    # ── Group predictors ──
    ELLIPSOID_NAMES = {"RCP", "NLE"}
    RECTANGLE_NAMES = {"DistSplit", "CQR", "MCQR", "EMCQR"}

    solo_preds = []
    ellipsoid_preds = []
    rectangle_preds = []

    for cp, res in pred_list:
        name = cp.name
        if name in ELLIPSOID_NAMES:
            ellipsoid_preds.append((cp, res))
        elif name in RECTANGLE_NAMES:
            rectangle_preds.append((cp, res))
        else:
            solo_preds.append((cp, res))

    n_panels = len(solo_preds)
    if ellipsoid_preds:
        n_panels += 1
    if rectangle_preds:
        n_panels += 1

    # ── Denormalization helpers ──
    y_mean = np.asarray(y_orig_mean, dtype=np.float64)
    y_std = np.asarray(y_orig_std, dtype=np.float64)

    def denorm(y_norm_arr):
        return y_norm_arr * y_std + y_mean

    # NYC taxi: Y[:,0]=dropoff_lon≈-73.97, Y[:,1]=dropoff_lat≈40.75
    # Column with larger absolute mean is longitude
    lat_idx, lon_idx = 0, 1
    if abs(y_mean[0]) > abs(y_mean[1]):
        lat_idx, lon_idx = 1, 0

    # Map center: hardcode NYC midtown
    center_lat, center_lon = 40.7428, -73.9660

    # Pickup marker: denormalize x_point to get real pickup lat/lon
    if x_orig_mean is not None and x_orig_std is not None:
        x_np = x_point.cpu().numpy().astype(np.float64)
        x_raw = x_np * np.asarray(x_orig_std) + np.asarray(x_orig_mean)
        pickup_lat = float(x_raw[lat_idx])
        pickup_lon = float(x_raw[lon_idx])
    else:
        pickup_lat, pickup_lon = center_lat, center_lon

    # True dropoff marker
    dropoff_lat, dropoff_lon = None, None
    if y_true is not None:
        yt_np = y_true.cpu().numpy().astype(np.float64) if hasattr(y_true, 'cpu') else np.asarray(y_true, dtype=np.float64)
        yt_orig = yt_np * y_std + y_mean
        dropoff_lat = float(yt_orig[lat_idx])
        dropoff_lon = float(yt_orig[lon_idx])

    # ── Grid in normalized space ──
    if y_range is not None:
        y1_grid = np.linspace(y_range[0][0], y_range[0][1], grid_res)
        y2_grid = np.linspace(y_range[1][0], y_range[1][1], grid_res)
    else:
        margin = 3.0
        y1_grid = np.linspace(-margin, margin, grid_res)
        y2_grid = np.linspace(-margin, margin, grid_res)
    Y1, Y2 = np.meshgrid(y1_grid, y2_grid)
    y_grid = torch.tensor(
        np.stack([Y1.ravel(), Y2.ravel()], axis=1), dtype=torch.float32
    )

    # Grid corners in original coords for PNG extent
    corner_lo = denorm(np.array([y1_grid[0], y2_grid[0]]))
    corner_hi = denorm(np.array([y1_grid[-1], y2_grid[-1]]))

    def _compute_contour(cp, res):
        _STOCHASTIC_TAGS = ("Diff", "FM", "DDPM")
        is_stochastic = any(tag in cp.name for tag in _STOCHASTIC_TAGS)
        kwargs = {"n_avg": grid_n_avg} if is_stochastic else {}
        _, scores = cp.predict_grid(x_point, y_grid, **kwargs)
        score_grid = scores.reshape(grid_res, grid_res)
        tau = res["tau"]
        if smooth_sigma > 0 and is_stochastic:
            score_grid = gaussian_filter(score_grid, sigma=smooth_sigma)
        return score_grid, tau

    # ════════════════════════════════════════════════════════════════
    # Shared: map tiles + drawing helpers
    # ════════════════════════════════════════════════════════════════
    has_tiles = False
    tile_img = tile_extent = None
    try:
        import contextily as cx
        # We'll use cx.add_basemap() directly on each axes instead of
        # pre-fetching tiles, because bounds2img returns Web Mercator extent
        # while our axes are in lon/lat (EPSG:4326).
        has_tiles = True
        _cx = cx
    except ImportError:
        _cx = None
        print("  [taxi_map] contextily not installed; plain background. "
              "pip install contextily")
    except Exception as e:
        _cx = None
        print(f"  [taxi_map] contextily error: {e}")

    region_color = "#ff0000"

    def _draw_bg(ax):
        if has_tiles:
            try:
                _cx.add_basemap(ax, crs="EPSG:4326",
                                source=_cx.providers.OpenStreetMap.Mapnik,
                                attribution=False)
            except Exception:
                ax.set_facecolor("#e8e8e8")
        else:
            ax.set_facecolor("#e8e8e8")

    def _draw_pin(ax):
        ax.plot(pickup_lon, pickup_lat, marker="o", ms=8,
                mfc="cyan", mec="black", mew=1.5, zorder=20)
        if dropoff_lat is not None:
            ax.plot(dropoff_lon, dropoff_lat, marker="*", ms=10,
                    mfc="red", mec="black", mew=0.8, zorder=20)

    def _draw_region(ax, score_grid, tau, color=region_color,
                     ls="-", lw=3.5):
        Y1o = Y1 * y_std[0] + y_mean[0]
        Y2o = Y2 * y_std[1] + y_mean[1]
        Xp = Y1o if lon_idx == 0 else Y2o
        Yp = Y2o if lon_idx == 0 else Y1o
        try:
            ax.contour(Xp, Yp, score_grid, levels=[tau],
                       colors=[color], linewidths=lw, linestyles=[ls],
                       zorder=6)
        except Exception:
            pass

    def _fmt(ax):
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        lon_lo = min(corner_lo[lon_idx], corner_hi[lon_idx])
        lon_hi = max(corner_lo[lon_idx], corner_hi[lon_idx])
        lat_lo = min(corner_lo[lat_idx], corner_hi[lat_idx])
        lat_hi = max(corner_lo[lat_idx], corner_hi[lat_idx])
        ax.set_xlim(lon_lo, lon_hi)
        ax.set_ylim(lat_lo, lat_hi)

    saved = []

    # ════════════════════════════════════════════════════════════════
    # KDE precomputation (shared by Part 1 combined + Part 2 individual)
    # ════════════════════════════════════════════════════════════════
    _kde_data = None  # will hold precomputed KDE results
    if train_x is not None and train_y is not None:
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.neighbors import KernelDensity as skKDE

        x_np = x_point.cpu().numpy().astype(np.float64)
        tx_np = train_x.cpu().numpy().astype(np.float64)
        ty_np = train_y.cpu().numpy().astype(np.float64)

        knn = KNeighborsRegressor(n_neighbors=kde_k)
        knn.fit(tx_np, ty_np)
        _, indices = knn.kneighbors(x_np.reshape(1, -1), return_distance=True)
        neighbor_y = ty_np[indices[0]]  # [k, 2] normalized

        # Denormalize → [lat, lon]
        neighbor_y_orig = neighbor_y * y_std + y_mean
        neighbor_y_latlon = np.column_stack([
            neighbor_y_orig[:, lat_idx], neighbor_y_orig[:, lon_idx]
        ])

        kde_obj = skKDE(bandwidth=kde_bandwidth, kernel='gaussian')
        kde_obj.fit(neighbor_y_latlon)

        # Grid in original [lat, lon] — use same extent as other methods
        kde_res = 200
        # Map extent from _fmt
        map_lon_lo = min(corner_lo[lon_idx], corner_hi[lon_idx])
        map_lon_hi = max(corner_lo[lon_idx], corner_hi[lon_idx])
        map_lat_lo = min(corner_lo[lat_idx], corner_hi[lat_idx])
        map_lat_hi = max(corner_lo[lat_idx], corner_hi[lat_idx])
        # Expand KDE grid to cover the full map extent
        lat_min_k = min(neighbor_y_latlon[:, 0].min() - 0.01, map_lat_lo)
        lat_max_k = max(neighbor_y_latlon[:, 0].max() + 0.01, map_lat_hi)
        lon_min_k = min(neighbor_y_latlon[:, 1].min() - 0.01, map_lon_lo)
        lon_max_k = max(neighbor_y_latlon[:, 1].max() + 0.01, map_lon_hi)
        lat_grid = np.linspace(lat_min_k, lat_max_k, kde_res)
        lon_grid = np.linspace(lon_min_k, lon_max_k, kde_res)
        LON_G, LAT_G = np.meshgrid(lon_grid, lat_grid)
        kde_grid = np.c_[LAT_G.ravel(), LON_G.ravel()]
        log_density = kde_obj.score_samples(kde_grid)
        density = np.exp(log_density).reshape(LAT_G.shape)

        # 90% HPD level set
        sorted_d = np.sort(density.ravel())[::-1]
        cum_d = np.cumsum(sorted_d)
        threshold_idx = np.searchsorted(cum_d, 0.9 * cum_d[-1])
        contour_level = sorted_d[threshold_idx]

        _kde_data = {
            "latlon": neighbor_y_latlon,
            "LON_G": LON_G, "LAT_G": LAT_G,
            "density": density, "level": contour_level,
            "lon_lim": (lon_min_k, lon_max_k),
            "lat_lim": (lat_min_k, lat_max_k),
        }

    def _draw_kde_on_ax(ax):
        """Draw KDE contour on the given axes."""
        if _kde_data is None:
            return
        d = _kde_data
        ax.contour(d["LON_G"], d["LAT_G"], d["density"],
                   levels=[d["level"]],
                   colors=['red'], linewidths=3.5, zorder=6)
        _draw_pin(ax)
        _fmt(ax)
        if has_tiles:
            try:
                _cx.add_basemap(ax, crs="EPSG:4326",
                                source=_cx.providers.OpenStreetMap.Mapnik,
                                attribution=False)
            except Exception:
                ax.set_facecolor("#e8e8e8")

    # ════════════════════════════════════════════════════════════════
    # Part 1: Combined subplot grid PNG
    # ════════════════════════════════════════════════════════════════
    has_kde = _kde_data is not None
    total_panels = (1 if has_kde else 0) + n_panels
    if total_panels > 0:
        ncols = min(total_panels, 4)
        nrows = (total_panels + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(4.5 * ncols, 4.5 * nrows),
                                 squeeze=False)
        axes_flat = axes.ravel()

        pi = 0
        # KDE as first panel
        if has_kde:
            _draw_kde_on_ax(axes_flat[pi]); pi += 1

        for cp, res in solo_preds:
            ax = axes_flat[pi]
            sg, tau = _compute_contour(cp, res)
            _draw_region(ax, sg, tau)
            _draw_pin(ax); _fmt(ax); _draw_bg(ax); pi += 1

        if ellipsoid_preds:
            ax = axes_flat[pi]
            ellip_styles = ["-", "--"]
            for i, (cp, res) in enumerate(ellipsoid_preds):
                sg, tau = _compute_contour(cp, res)
                _draw_region(ax, sg, tau, ls=ellip_styles[i % 2])
            _draw_pin(ax); _fmt(ax); _draw_bg(ax); pi += 1

        if rectangle_preds:
            ax = axes_flat[pi]
            for i, (cp, res) in enumerate(rectangle_preds):
                sg, tau = _compute_contour(cp, res)
                _draw_region(ax, sg, tau)
            _draw_pin(ax); _fmt(ax); _draw_bg(ax); pi += 1

        for j in range(pi, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.tight_layout(pad=0.5)
        if save_dir:
            fpath = os.path.join(save_dir, f"{prefix}_taxi_map.png")
            fig.savefig(fpath, dpi=180, bbox_inches="tight")
            saved.append(fpath)
            print(f"  Saved: {fpath}")
        plt.close(fig)

    # ════════════════════════════════════════════════════════════════
    # Part 2: Per-method individual PNG maps
    # ════════════════════════════════════════════════════════════════

    # --- KDE (first position) ---
    if has_kde:
        fig, ax = plt.subplots(1, 1, figsize=(6, 7))
        _draw_kde_on_ax(ax)
        if save_dir:
            fpath = os.path.join(save_dir, f"{prefix}_kde_map.png")
            fig.savefig(fpath, dpi=200, bbox_inches="tight")
            saved.append(fpath)
            print(f"  Saved: {fpath}")
        plt.close(fig)

    # --- Solo methods: one PNG each ---
    for cp, res in solo_preds:
        name = cp.name
        score_grid, tau = _compute_contour(cp, res)

        fig, ax = plt.subplots(1, 1, figsize=(6, 7))
        _draw_region(ax, score_grid, tau)
        _draw_pin(ax)
        _fmt(ax)
        _draw_bg(ax)

        if save_dir:
            safe_name = name.replace("-", "_").replace(" ", "_").lower()
            fpath = os.path.join(save_dir, f"{prefix}_{safe_name}_map.png")
            fig.savefig(fpath, dpi=200, bbox_inches="tight")
            saved.append(fpath)
            print(f"  Saved: {fpath}")
        plt.close(fig)

    # --- Ellipsoid methods (RCP, NLE): merged into one PNG, solid/dashed ---
    if ellipsoid_preds:
        ellip_styles = ["-", "--"]
        fig, ax = plt.subplots(1, 1, figsize=(6, 7))
        for i, (cp, res) in enumerate(ellipsoid_preds):
            sg, tau = _compute_contour(cp, res)
            _draw_region(ax, sg, tau, ls=ellip_styles[i % 2])
        _draw_pin(ax)
        _fmt(ax)
        _draw_bg(ax)

        if save_dir:
            names = "_".join(cp.name.replace("-","_").lower()
                             for cp, _ in ellipsoid_preds)
            fpath = os.path.join(save_dir, f"{prefix}_{names}_map.png")
            fig.savefig(fpath, dpi=200, bbox_inches="tight")
            saved.append(fpath)
            print(f"  Saved: {fpath}")
        plt.close(fig)

    # --- Rectangle methods: one PNG each ---
    for cp, res in rectangle_preds:
        name = cp.name
        score_grid, tau = _compute_contour(cp, res)

        fig, ax = plt.subplots(1, 1, figsize=(6, 7))
        _draw_region(ax, score_grid, tau)
        _draw_pin(ax)
        _fmt(ax)
        _draw_bg(ax)

        if save_dir:
            safe_name = name.replace("-", "_").replace(" ", "_").lower()
            fpath = os.path.join(save_dir, f"{prefix}_{safe_name}_map.png")
            fig.savefig(fpath, dpi=200, bbox_inches="tight")
            saved.append(fpath)
            print(f"  Saved: {fpath}")
        plt.close(fig)

    return saved


def plot_hurricane_map(predictors, x_point, y_orig_mean, y_orig_std,
                       x_orig_mean, x_orig_std,
                       all_results, grid_res=150, grid_n_avg=3,
                       smooth_sigma=1.5, save_dir=None, prefix="hurricane",
                       x_train=None, y_train=None):
    """Draw hurricane prediction regions on interactive maps using Folium.

    Generates:
      1. Per-method maps for Z-space methods (NF-Ball, NF-NLL, Diff-Denoise, etc.)
      2. A KDE baseline map (kNN neighbors + KDE 90% density contour, like taxi)
      3. A combined map with all layers + KDE reference

    Args:
        predictors: dict {name: ConformalPredictor} (Z-space only, must have predict_grid)
        x_point: [x_dim] tensor, conditioning point (normalized)
        y_orig_mean/std: [2] numpy, Y normalization params (dlat, dlon)
        x_orig_mean/std: [8] numpy, X normalization params
        all_results: dict {name: {coverage, tau, volume, ...}}
        x_train, y_train: training data tensors (normalized) for KDE
    """
    try:
        import folium
    except ImportError:
        print("  [hurricane_map] folium not installed. pip install folium")
        return

    from scipy.ndimage import gaussian_filter

    # Recover current position from normalized x_point
    x_np = x_point.cpu().numpy().astype(np.float64)
    x_raw = x_np * x_orig_std + x_orig_mean
    current_lat = float(x_raw[0])
    current_lon = float(x_raw[1])
    wind_kt = float(x_raw[4]) if len(x_raw) > 4 else 0

    # Grid in normalized displacement space
    margin = 3.5
    y1_grid = np.linspace(-margin, margin, grid_res)
    y2_grid = np.linspace(-margin, margin, grid_res)
    Y1, Y2 = np.meshgrid(y1_grid, y2_grid)
    y_grid = torch.tensor(
        np.stack([Y1.ravel(), Y2.ravel()], axis=1), dtype=torch.float32
    )

    def denorm_disp(dy_norm):
        return dy_norm * y_orig_std + y_orig_mean

    def disp_to_latlon(disp_pts):
        return [[current_lat + float(pt[0]), current_lon + float(pt[1])]
                for pt in disp_pts]

    # ── KDE reference region from kNN ──
    kde_latlons_list = []
    if x_train is not None and y_train is not None:
        try:
            from sklearn.neighbors import KNeighborsRegressor, KernelDensity

            x_train_np = x_train.cpu().numpy()
            y_train_np = y_train.cpu().numpy()
            x_query = x_point.cpu().numpy().reshape(1, -1)

            k_neighbors = min(200, len(x_train_np))
            knn = KNeighborsRegressor(n_neighbors=k_neighbors)
            knn.fit(x_train_np, y_train_np)
            _, indices = knn.kneighbors(x_query, return_distance=True)
            nn_y = y_train_np[indices[0]]  # [k, 2] normalized displacements

            # KDE on normalized displacements
            kde = KernelDensity(bandwidth=0.15, kernel='gaussian')
            kde.fit(nn_y)

            grid_pts = np.stack([Y1.ravel(), Y2.ravel()], axis=1)
            log_density = kde.score_samples(grid_pts)
            density = np.exp(log_density).reshape(Y1.shape)

            # 90% density contour
            sorted_d = np.sort(density.ravel())[::-1]
            cum_d = np.cumsum(sorted_d)
            idx_90 = np.searchsorted(cum_d, 0.9 * cum_d[-1])
            level_90 = sorted_d[min(idx_90, len(sorted_d) - 1)]

            fig_tmp, ax_tmp = plt.subplots()
            cs = ax_tmp.contour(Y1, Y2, density, levels=[level_90])
            plt.close(fig_tmp)

            for pc in cs.collections:
                for path in pc.get_paths():
                    vn = path.vertices
                    vd = denorm_disp(vn)
                    if len(vd) >= 3:
                        kde_latlons_list.append(disp_to_latlon(vd))
            print(f"  [KDE] Found {len(kde_latlons_list)} contour(s)")
        except Exception as e:
            print(f"  [hurricane_map] KDE failed: {e}")

    # Method colors
    method_colors = {
        "NF-Ball": "red", "NF-NLL": "darkred",
        "Diff-Denoise": "blue", "FM-Path": "green",
    }

    saved_files = []

    # ── KDE-only map ──
    if save_dir and kde_latlons_list:
        kde_map = folium.Map(
            location=[current_lat, current_lon],
            zoom_start=6, width='900px', height='700px',
        )
        folium.CircleMarker(
            [current_lat, current_lon],
            radius=8, color="black", fill=True, fill_color="yellow",
            fill_opacity=0.9, popup="Current Position",
        ).add_to(kde_map)
        for kde_ll in kde_latlons_list:
            folium.Polygon(
                locations=kde_ll,
                color="purple", weight=3, opacity=0.8,
                fill=True, fill_color="purple", fill_opacity=0.15,
                popup="KDE 90% (unsafe)",
            ).add_to(kde_map)
        title_html = ('<div style="font-size:11px;background:white;padding:3px;'
                      'border:1px solid purple;border-radius:3px;">'
                      f'<b>KDE 90% (unsafe)</b><br>'
                      f'pos=({current_lat:.1f}, {current_lon:.1f}), '
                      f'wind={wind_kt:.0f}kt</div>')
        folium.Marker(
            [current_lat + 0.5, current_lon],
            icon=folium.DivIcon(html=title_html),
        ).add_to(kde_map)
        fpath = os.path.join(save_dir, f"{prefix}_kde.html")
        kde_map.save(fpath)
        saved_files.append(fpath)
        print(f"  Saved: {fpath}")

    # ── Per-method maps ──
    for name, cp in predictors.items():
        if name not in all_results:
            continue
        res = all_results[name]
        tau = res["tau"]

        is_stochastic = any(tag in name for tag in ("Diff", "FM", "DDPM"))
        kwargs = {"n_avg": grid_n_avg} if is_stochastic else {}
        _, scores = cp.predict_grid(x_point, y_grid, **kwargs)
        score_grid = scores.reshape(grid_res, grid_res)

        if smooth_sigma > 0 and is_stochastic:
            score_grid = gaussian_filter(score_grid, sigma=smooth_sigma)

        fig_tmp, ax_tmp = plt.subplots()
        contour_set = ax_tmp.contour(Y1, Y2, score_grid, levels=[tau])
        plt.close(fig_tmp)

        color = method_colors.get(name, "red")
        cov = res["coverage"]
        vol = res.get("volume", float("nan"))

        m = folium.Map(
            location=[current_lat, current_lon],
            zoom_start=6, width='900px', height='700px',
        )
        folium.CircleMarker(
            [current_lat, current_lon],
            radius=8, color="black", fill=True, fill_color="yellow",
            fill_opacity=0.9, popup="Current Position",
        ).add_to(m)

        # KDE reference (gray dashed)
        for kde_ll in kde_latlons_list:
            folium.Polygon(
                locations=kde_ll,
                color="gray", weight=2, dash_array="5 5",
                opacity=0.5, fill=True, fill_color="gray", fill_opacity=0.05,
                popup="KDE 90% (unsafe ref)",
            ).add_to(m)

        # Method region
        for pc in contour_set.collections:
            for path in pc.get_paths():
                vn = path.vertices
                vd = denorm_disp(vn)
                if len(vd) < 3:
                    continue
                folium.Polygon(
                    locations=disp_to_latlon(vd),
                    color=color, weight=3, opacity=0.8,
                    fill=True, fill_color=color, fill_opacity=0.15,
                    popup=f"{name}: cov={cov:.3f}, vol={vol:.5f}",
                ).add_to(m)

        title_text = (f"<b>{name}</b><br>cov={cov:.3f}, vol={vol:.5f}<br>"
                      f"pos=({current_lat:.1f}, {current_lon:.1f}), "
                      f"wind={wind_kt:.0f}kt")
        folium.Marker(
            [current_lat + 0.5, current_lon],
            icon=folium.DivIcon(html=f'<div style="font-size:11px;background:white;'
                                     f'padding:3px;border:1px solid {color};'
                                     f'border-radius:3px;white-space:nowrap;">'
                                     f'{title_text}</div>'),
        ).add_to(m)

        if save_dir:
            fname = f"{prefix}_{name.replace('-', '_').lower()}.html"
            fpath = os.path.join(save_dir, fname)
            m.save(fpath)
            saved_files.append(fpath)
            print(f"  Saved: {fpath}")

    # ── Combined map with toggleable layers ──
    if save_dir and len(predictors) > 0:
        combined = folium.Map(
            location=[current_lat, current_lon],
            zoom_start=6, width='1000px', height='800px',
        )
        folium.CircleMarker(
            [current_lat, current_lon],
            radius=8, color="black", fill=True, fill_color="yellow",
            fill_opacity=0.9, popup="Current Position",
        ).add_to(combined)

        # KDE layer
        if kde_latlons_list:
            kde_fg = folium.FeatureGroup(name="KDE 90% (unsafe)")
            for kde_ll in kde_latlons_list:
                folium.Polygon(
                    locations=kde_ll,
                    color="purple", weight=2, dash_array="5 5",
                    opacity=0.6, fill=True, fill_color="purple", fill_opacity=0.08,
                ).add_to(kde_fg)
            kde_fg.add_to(combined)

        # Z-space layers
        for name, cp in predictors.items():
            if name not in all_results:
                continue
            res = all_results[name]
            tau = res["tau"]
            color = method_colors.get(name, "red")

            is_stochastic = any(tag in name for tag in ("Diff", "FM", "DDPM"))
            kwargs = {"n_avg": grid_n_avg} if is_stochastic else {}
            _, scores = cp.predict_grid(x_point, y_grid, **kwargs)
            score_grid = scores.reshape(grid_res, grid_res)
            if smooth_sigma > 0 and is_stochastic:
                score_grid = gaussian_filter(score_grid, sigma=smooth_sigma)

            fig_tmp, ax_tmp = plt.subplots()
            contour_set = ax_tmp.contour(Y1, Y2, score_grid, levels=[tau])
            plt.close(fig_tmp)

            fg = folium.FeatureGroup(
                name=f"{name} (cov={res['coverage']:.3f})")
            for pc in contour_set.collections:
                for path in pc.get_paths():
                    vn = path.vertices
                    vd = denorm_disp(vn)
                    if len(vd) < 3:
                        continue
                    folium.Polygon(
                        locations=disp_to_latlon(vd),
                        color=color, weight=2, opacity=0.7,
                        fill=True, fill_color=color, fill_opacity=0.1,
                    ).add_to(fg)
            fg.add_to(combined)

        folium.LayerControl(collapsed=False).add_to(combined)
        fpath = os.path.join(save_dir, f"{prefix}_combined.html")
        combined.save(fpath)
        saved_files.append(fpath)
        print(f"  Saved: {fpath}")

    return saved_files