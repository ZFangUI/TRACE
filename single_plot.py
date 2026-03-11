#!/usr/bin/env python3
"""Re-draw prediction region plots from saved checkpoints.

Usage:
    python single_plot.py --exp_dir ./experiments/twomoons_s0_single --x_index 750
    python single_plot.py --exp_dir ./experiments/pinwheel_s0_single --x_index 500 --device cuda

Loads models + baselines + results from checkpoint, draws clean region plots:
  - Only methods in ORDER (renamed, filtered)
  - No cov/tau/vol in titles, no axis labels/ticks
  - True Density panel on the left
  - Combined figure + individual per-method figures → replot/ subdirectory
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import gaussian_filter


# ══════════════════════════════════════════════════════════════════
# Display configuration — edit these to change appearance
# ══════════════════════════════════════════════════════════════════

RENAME = {
    "NF-Ball": "CONTRA",
    "NF-NLL": "JAPAN",
    "Diff-Denoise": "TRACE-Diff",
    "FM-Path": "TRACE-FM",
}

EXCLUDE = {"DistSplit", "Diff-ODE-Ball", "FM-ODE-Ball", "CQR"}

ORDER = ["TRACE-Diff", "TRACE-FM", "CONTRA", "JAPAN", "PCP-Diff", "MCQR", "RCP", "NLE"]

# Per-method colors: (fill, edge)
METHOD_STYLE = {
    "TRACE-Diff":     ("#b3d9f7", "#2471a3"),
    "TRACE-FM":       ("#f5c6c6", "#c0392b"),
    "CONTRA":   ("#d5b3f0", "#7d3c98"),
    "JAPAN":    ("#9394e7", "#1a5276"),
    "PCP-Diff": ("#b1ce46", "#1e8449"),
    "MCQR":     ("#c497b2", "#d68910"),
    "RCP":      ("#f1ba7e", "#a04010"),
    "NLE":      ("#c6f5f5", "#117a65"),
}
FALLBACK_STYLE = ("#d9d9d9", "#555555")


# ══════════════════════════════════════════════════════════════════
# Auto-infer model architecture from state_dict keys
# ══════════════════════════════════════════════════════════════════

def _infer_nf_arch(state_dict):
    keys = list(state_dict.keys())
    has_param_net = any("param_net" in k for k in keys)
    flow_type = "nsf" if has_param_net else "realnvp"

    layer_ids = set()
    for k in keys:
        if k.startswith("flow.layers."):
            parts = k.split(".")
            if len(parts) > 2:
                try:
                    layer_ids.add(int(parts[2]))
                except ValueError:
                    pass
    n_layers = max(layer_ids) + 1 if layer_ids else 8

    has_blocks = any("cond_net.blocks" in k for k in keys)
    cond_net_type = "resnet" if has_blocks else "mlp"

    hidden_dim = 256
    for k in keys:
        if "cond_net.net.0.weight" in k or "cond_net.proj_in.weight" in k:
            hidden_dim = state_dict[k].shape[0]
            break

    cond_dim = 128
    for k in keys:
        if "cond_net.net.4.weight" in k or "cond_net.proj_out.1.weight" in k:
            cond_dim = state_dict[k].shape[0]
            break

    n_bins = 8
    if flow_type == "nsf":
        for k in keys:
            if "flow.layers.0.param_net.4.weight" in k:
                out_size = state_dict[k].shape[0]
                for dim_guess in [2, 1, 3, 4]:
                    if out_size % dim_guess == 0:
                        per_dim = out_size // dim_guess
                        if (per_dim - 1) % 3 == 0:
                            n_bins = (per_dim - 1) // 3
                            break
                break

    return {"flow_type": flow_type, "n_layers": n_layers, "n_bins": n_bins,
            "hidden_dim": hidden_dim, "cond_dim": cond_dim,
            "cond_net_type": cond_net_type,
            "tail_bound": 3.0, "s_clamp": 3.0}


def _infer_diff_arch(state_dict):
    keys = list(state_dict.keys())
    block_ids = set()
    for k in keys:
        if "net.blocks." in k:
            parts = k.split(".")
            idx = parts.index("blocks") + 1
            if idx < len(parts):
                try:
                    block_ids.add(int(parts[idx]))
                except ValueError:
                    pass
    n_blocks = max(block_ids) + 1 if block_ids else 8

    hidden_dim = 256
    for k in keys:
        if "net.blocks.0.fc1.weight" in k:
            hidden_dim = state_dict[k].shape[0]
            break

    T = 200
    for k in keys:
        if "alphas_bar" in k:
            T = state_dict[k].shape[0]
            break

    return {"n_blocks": n_blocks, "hidden_dim": hidden_dim, "T": T}


def _infer_fm_arch(state_dict):
    keys = list(state_dict.keys())
    block_ids = set()
    for k in keys:
        if "net.blocks." in k:
            parts = k.split(".")
            idx = parts.index("blocks") + 1
            if idx < len(parts):
                try:
                    block_ids.add(int(parts[idx]))
                except ValueError:
                    pass
    n_blocks = max(block_ids) + 1 if block_ids else 8

    hidden_dim = 256
    for k in keys:
        if "net.blocks.0.fc1.weight" in k:
            hidden_dim = state_dict[k].shape[0]
            break

    return {"n_blocks": n_blocks, "hidden_dim": hidden_dim}


def _infer_dataset_from_dir(exp_dir):
    """Infer dataset name from directory name, e.g. pinwheel_s0_single -> pinwheel."""
    base = os.path.basename(exp_dir.rstrip("/"))
    # Strip _single suffix, then _s{seed}
    name = base.replace("_single", "")
    parts = name.rsplit("_s", 1)
    return parts[0] if parts else name


# ══════════════════════════════════════════════════════════════════
# Loading
# ══════════════════════════════════════════════════════════════════

def load_checkpoint(exp_dir, rep_idx=0):
    """Load models, baselines, data, results from a repeat directory."""
    rep_dir = os.path.join(exp_dir, f"repeat_{rep_idx:03d}")
    if not os.path.isdir(rep_dir):
        rep_dir = exp_dir

    # Results
    results_path = os.path.join(rep_dir, "results.json")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"No results.json in {rep_dir}")
    with open(results_path) as f:
        raw_results = json.load(f)
    raw_results.pop("_time_seconds", None)

    # Data split
    save_data = torch.load(os.path.join(rep_dir, "data_split.pt"),
                           map_location="cpu", weights_only=False)
    y_orig_mean = save_data.pop("_y_orig_mean", None)
    y_orig_std = save_data.pop("_y_orig_std", None)
    save_data.pop("_y_std_prod", None)
    save_data.pop("_vol_dims", None)
    x_orig_mean = save_data.pop("_x_orig_mean", None)
    x_orig_std = save_data.pop("_x_orig_std", None)
    data = save_data

    xd = int(data["trx"].shape[1])
    yd = int(data["try"].shape[1])
    y_mean = data["try"].mean(0)
    y_std = data["try"].std(0)

    # Config (from config.json or defaults + dir name inference)
    cfg = _load_config(exp_dir)

    # ── Load generative models (auto-infer architecture) ──
    models = {}

    nf_path = os.path.join(rep_dir, "nf_model.pt")
    if os.path.exists(nf_path):
        from models.nf import NFModel
        sd = torch.load(nf_path, map_location="cpu", weights_only=True)
        arch = _infer_nf_arch(sd)
        print(f"  NF: {arch['flow_type']}, {arch['n_layers']} layers, "
              f"bins={arch['n_bins']}, cond={arch['cond_net_type']}")
        nf = NFModel(xd, yd, arch["cond_dim"], arch["hidden_dim"],
                     arch["n_layers"], arch["s_clamp"],
                     flow_type=arch["flow_type"],
                     n_bins=arch["n_bins"], tail_bound=arch["tail_bound"],
                     cond_net_type=arch["cond_net_type"])
        nf.load_state_dict(sd)
        nf.eval()
        models["nf"] = nf

    diff_path = os.path.join(rep_dir, "diff_model.pt")
    if os.path.exists(diff_path):
        from models.diffusion import ConditionalDDPM
        sd = torch.load(diff_path, map_location="cpu", weights_only=True)
        arch = _infer_diff_arch(sd)
        print(f"  Diff: {arch['n_blocks']} blocks, hidden={arch['hidden_dim']}, T={arch['T']}")
        diff = ConditionalDDPM(
            yd, xd, T=arch["T"], hidden_dim=arch["hidden_dim"],
            n_blocks=arch["n_blocks"],
            schedule=cfg.get("diff_schedule", "cosine"),
            beta_min=cfg.get("diff_beta_min", 1e-4),
            beta_max=cfg.get("diff_beta_max", 0.02),
            cfg_drop_prob=cfg.get("diff_cfg_drop_prob", 0.15))
        diff.set_normalization(y_mean, y_std)
        diff.load_state_dict(sd)
        diff.eval()
        models["diff"] = diff

    fm_path = os.path.join(rep_dir, "fm_model.pt")
    if os.path.exists(fm_path):
        from models.flow_matching import ConditionalFlowMatching
        sd = torch.load(fm_path, map_location="cpu", weights_only=True)
        arch = _infer_fm_arch(sd)
        print(f"  FM: {arch['n_blocks']} blocks, hidden={arch['hidden_dim']}")
        fm = ConditionalFlowMatching(
            yd, xd, hidden_dim=arch["hidden_dim"],
            n_blocks=arch["n_blocks"],
            sigma_min=cfg.get("fm_sigma_min", 1e-4),
            cfg_drop_prob=cfg.get("fm_cfg_drop_prob", 0.15))
        fm.set_normalization(y_mean, y_std)
        fm.load_state_dict(sd)
        fm.eval()
        models["fm"] = fm

    # ── Load saved baselines ──
    baselines = {}
    bl_path = os.path.join(rep_dir, "baselines.pt")
    if os.path.exists(bl_path):
        baselines = torch.load(bl_path, map_location="cpu", weights_only=False)
        print(f"  Baselines loaded: {list(baselines.keys())}")
    else:
        print(f"  No baselines.pt found (baseline methods will be skipped)")

    return data, models, baselines, raw_results, cfg, y_orig_mean, y_orig_std, x_orig_mean, x_orig_std


def _load_config(exp_dir):
    cfg_path = os.path.join(exp_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            return json.load(f)
    # Fallback: defaults + infer dataset from dir name
    from config import Config
    d = Config().to_dict()
    d["dataset"] = _infer_dataset_from_dir(exp_dir)
    return d


# ══════════════════════════════════════════════════════════════════
# Score / predictor building
# ══════════════════════════════════════════════════════════════════

class _PredWrap:
    """Wrapper: score_fn + tau → predict_grid."""
    def __init__(self, name, score_fn, tau):
        self.name = name
        self.score_fn = score_fn
        self.tau = tau

    def predict_grid(self, x_pt, y_grid, **kw):
        sc = self.score_fn.compute_on_grid(x_pt, y_grid, **kw)
        return sc <= self.tau, sc


class _BaselineWrap:
    """Wrapper for baseline grid function."""
    def __init__(self, name, grid_fn, tau):
        self.name = name
        self._fn = grid_fn
        self.tau = tau

    def predict_grid(self, x_pt, y_grid, **kw):
        sc = self._fn(x_pt, y_grid)
        return sc <= self.tau, sc


def build_predictors(models, baselines, raw_results, cfg, x_point, device):
    """Build all predictors (Z-space + baselines) with their tau values."""
    predictors = {}
    seed = cfg.get("seed", 0)

    # ── Z-space methods ──
    if "nf" in models:
        from scores.nf_scores import NFBallScore, NFNLLScore
        for orig, cls in [("NF-Ball", NFBallScore), ("NF-NLL", NFNLLScore)]:
            if orig in EXCLUDE or orig not in raw_results:
                continue
            display = RENAME.get(orig, orig)
            if display in EXCLUDE:
                continue
            sfn = cls(models["nf"], device)
            predictors[display] = _PredWrap(display, sfn, raw_results[orig]["tau"])

    if "diff" in models:
        from scores.diffusion_scores import DiffusionDenoiseScore
        orig = "Diff-Denoise"
        if orig not in EXCLUDE and orig in raw_results:
            display = RENAME.get(orig, orig)
            if display not in EXCLUDE:
                sfn = DiffusionDenoiseScore(
                    models["diff"], device,
                    n_timesteps=cfg.get("diff_score_timesteps", 15),
                    n_repeats=cfg.get("diff_score_repeats", 8),
                    seed=seed)
                predictors[display] = _PredWrap(display, sfn,
                                                 raw_results[orig]["tau"])

    if "fm" in models:
        from scores.fm_scores import FMPathScore
        orig = "FM-Path"
        if orig not in EXCLUDE and orig in raw_results:
            display = RENAME.get(orig, orig)
            if display not in EXCLUDE:
                sfn = FMPathScore(
                    models["fm"], device,
                    n_timesteps=cfg.get("fm_score_timesteps", 15),
                    n_repeats=cfg.get("fm_score_repeats", 8),
                    seed=seed)
                predictors[display] = _PredWrap(display, sfn,
                                                 raw_results[orig]["tau"])

    # ── Baseline methods (loaded from baselines.pt) ──
    if not baselines or "nf" not in models:
        return predictors

    from baselines import sample_ys_nf, sample_ys_diff

    # NF samples at x_point (needed by most baselines)
    nf_samp = sample_ys_nf(models["nf"], x_point.unsqueeze(0),
                            n_samples=cfg.get("n_samples_baseline", 1000),
                            device=device)[0]
    yp = nf_samp.mean(dim=0)

    # RCP, NLE
    for bname in ["RCP", "NLE"]:
        if bname in EXCLUDE or bname not in baselines or bname not in raw_results:
            continue
        display = RENAME.get(bname, bname)
        if display in EXCLUDE:
            continue
        bobj = baselines[bname]
        gfn = lambda xp, yg, b=bobj, y=yp: b.compute_on_grid(xp, yg, y)
        predictors[display] = _BaselineWrap(display, gfn,
                                             raw_results[bname]["tau"])

    # MCQR — binary score (0 inside, 2 outside), use tau=0.5
    if "MCQR" not in EXCLUDE and "MCQR" in baselines and "MCQR" in raw_results:
        display = RENAME.get("MCQR", "MCQR")
        if display not in EXCLUDE:
            bobj = baselines["MCQR"]
            gfn = lambda xp, yg, b=bobj, s=nf_samp: b.compute_on_grid(xp, yg, s)
            predictors[display] = _BaselineWrap(display, gfn, 0.5)

    # DistSplit — binary score, tau=0.5 (usually excluded)
    if "DistSplit" not in EXCLUDE and "DistSplit" in baselines and "DistSplit" in raw_results:
        display = RENAME.get("DistSplit", "DistSplit")
        if display not in EXCLUDE:
            bobj = baselines["DistSplit"]
            gfn = lambda xp, yg, b=bobj, s=nf_samp: b.compute_on_grid(xp, yg, s)
            predictors[display] = _BaselineWrap(display, gfn, 0.5)

    # PCP variants
    for key in baselines:
        if not key.startswith("PCP"):
            continue
        if key in EXCLUDE or key not in raw_results:
            continue
        display = RENAME.get(key, key)
        if display in EXCLUDE:
            continue
        pcp_obj, _ = baselines[key]
        # Get samples from the appropriate model
        gname = key.split("-")[1]  # "Diff", "NF", "FM"
        try:
            if gname == "Diff" and "diff" in models:
                pdens = sample_ys_diff(
                    models["diff"], x_point.unsqueeze(0),
                    n_samples=cfg.get("pcp_n_samples", 500),
                    device=device,
                    n_steps=cfg.get("diff_sample_steps", 100))[0]
            elif gname == "NF" and "nf" in models:
                pdens = sample_ys_nf(
                    models["nf"], x_point.unsqueeze(0),
                    n_samples=cfg.get("pcp_n_samples", 500),
                    device=device)[0]
            else:
                continue
            gfn = lambda xp, yg, b=pcp_obj, s=pdens: b.compute_on_grid(xp, yg, s)
            predictors[display] = _BaselineWrap(display, gfn,
                                                 raw_results[key]["tau"])
        except Exception as e:
            print(f"  Warning: {key} grid failed: {e}")

    return predictors


# ══════════════════════════════════════════════════════════════════
# Drawing helpers
# ══════════════════════════════════════════════════════════════════

def compute_grid_scores(pred, x_point, y_grid, grid_res,
                        is_stochastic=False, grid_n_avg=3, smooth_sigma=1.5):
    kwargs = {"n_avg": grid_n_avg} if is_stochastic else {}
    _, scores = pred.predict_grid(x_point, y_grid, **kwargs)
    score_grid = scores.reshape(grid_res, grid_res)
    if smooth_sigma > 0 and is_stochastic:
        score_grid = gaussian_filter(score_grid, sigma=smooth_sigma)
    region = (score_grid <= pred.tau).astype(float)
    return region, score_grid


def draw_single_panel(ax, region, score_grid, Y1, Y2, tau,
                      scatter_y, fc, ec, y_true=None, ls="-"):
    ax.contourf(Y1, Y2, region, levels=[0.5, 1.5], colors=[fc], alpha=0.45)
    ax.contour(Y1, Y2, score_grid, levels=[tau],
               colors=[ec], linewidths=2.0, linestyles=[ls])
    if scatter_y is not None:
        n_show = min(400, len(scatter_y))
        ax.scatter(scatter_y[:n_show, 0], scatter_y[:n_show, 1],
                   s=3, c="#888888", alpha=0.2, edgecolors="none", zorder=5)
    if y_true is not None:
        yt = y_true.numpy() if isinstance(y_true, torch.Tensor) else y_true
        ax.scatter(yt[0], yt[1], marker="*", s=150, c="#e74c3c",
                   edgecolors="k", linewidths=0.6, zorder=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")


def draw_density_panel(ax, dataset_name, x_point, y_range, grid_res,
                       y_orig_mean=None, y_orig_std=None):
    from datasets import sample_true_conditional
    y1_grid = np.linspace(y_range[0][0], y_range[0][1], grid_res)
    y2_grid = np.linspace(y_range[1][0], y_range[1][1], grid_res)

    dense = sample_true_conditional(dataset_name, x_point, n=20000, seed=99999)
    if dense is not None and y_orig_mean is not None and y_orig_std is not None:
        dense = (dense - y_orig_mean) / (y_orig_std + 1e-8)

    h, _, _ = np.histogram2d(dense[:, 0], dense[:, 1],
                             bins=[y1_grid, y2_grid])
    h = h.T
    kde_sigma = max(1.5, grid_res / 80)
    h_smooth = gaussian_filter(h.astype(float), sigma=kde_sigma)
    h_plot = np.zeros((grid_res, grid_res))
    h_plot[:h_smooth.shape[0], :h_smooth.shape[1]] = h_smooth

    ax.imshow(h_plot, origin="lower", aspect="equal",
              extent=[y_range[0][0], y_range[0][1],
                      y_range[1][0], y_range[1][1]],
              cmap="magma", interpolation="bilinear")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title("True Density", fontsize=11, fontweight="bold",
                 color="#2c3e50", pad=6)


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def make_plots(exp_dir, x_index=None, rep_idx=0, grid_res=200,
               grid_n_avg=3, smooth_sigma=1.5, device="cpu", dpi=200):

    print(f"Loading from {exp_dir} (repeat {rep_idx}) ...")
    data, models, baselines, raw_results, cfg, y_orig_mean, y_orig_std, \
        x_orig_mean, x_orig_std = load_checkpoint(exp_dir, rep_idx)

    dataset_name = cfg.get("dataset", _infer_dataset_from_dir(exp_dir))
    n_test = len(data["tsx"])
    if x_index is None:
        x_index = n_test // 2
    x_point = data["tsx"][x_index]
    y_true = data["tsy"][x_index]

    print(f"  Dataset: {dataset_name}, x_index={x_index}, n_test={n_test}")

    # Build all predictors
    print("Building predictors ...")
    predictors = build_predictors(models, baselines, raw_results,
                                   cfg, x_point, device)

    # Filter and order
    ordered_names = [n for n in ORDER if n in predictors]
    print(f"  Methods: {ordered_names}")
    if not ordered_names:
        print("No methods available!")
        return

    # ── Prepare scatter data and grid ──
    from datasets import sample_true_conditional, DATASETS
    is_synthetic = dataset_name in DATASETS

    cond_samples = None
    if is_synthetic:
        cond_samples = sample_true_conditional(dataset_name, x_point, n=3000)
        if cond_samples is not None and y_orig_mean is not None:
            cond_samples = (cond_samples - y_orig_mean) / (y_orig_std + 1e-8)

    if cond_samples is None and "nf" in models:
        torch.manual_seed(42)
        models["nf"].to(device).eval()
        with torch.no_grad():
            xi = x_point.unsqueeze(0).to(device)
            cond_samples = models["nf"].sample(xi, 2000).cpu().numpy().reshape(-1, 2)

    # Also sample from Diff/FM to ensure y_range covers all methods' regions
    all_samples = [cond_samples] if cond_samples is not None else []
    if "diff" in models:
        try:
            from baselines import sample_ys_diff
            models["diff"].to(device).eval()
            diff_samp = sample_ys_diff(
                models["diff"], x_point.unsqueeze(0),
                n_samples=1000, device=device).cpu().numpy().reshape(-1, 2)
            all_samples.append(diff_samp)
        except Exception:
            pass
    if "fm" in models:
        try:
            from baselines import sample_ys_fm
            models["fm"].to(device).eval()
            fm_samp = sample_ys_fm(
                models["fm"], x_point.unsqueeze(0),
                n_samples=1000, device=device).cpu().numpy().reshape(-1, 2)
            all_samples.append(fm_samp)
        except Exception:
            pass

    if all_samples:
        scatter_y = np.concatenate(all_samples, axis=0) if len(all_samples) > 1 else all_samples[0]
    else:
        scatter_y = data["tsy"].numpy()

    margin = 0.25
    y1_lo, y1_hi = scatter_y[:, 0].min(), scatter_y[:, 0].max()
    y2_lo, y2_hi = scatter_y[:, 1].min(), scatter_y[:, 1].max()
    span = max(y1_hi - y1_lo, y2_hi - y2_lo)
    y1_c, y2_c = (y1_lo + y1_hi) / 2, (y2_lo + y2_hi) / 2
    half = span / 2 * (1 + margin)
    y_range = ((y1_c - half, y1_c + half), (y2_c - half, y2_c + half))

    y1_grid = np.linspace(y_range[0][0], y_range[0][1], grid_res)
    y2_grid = np.linspace(y_range[1][0], y_range[1][1], grid_res)
    Y1, Y2 = np.meshgrid(y1_grid, y2_grid)
    y_grid = torch.tensor(
        np.stack([Y1.ravel(), Y2.ravel()], axis=1), dtype=torch.float32)

    # ── Compute grid scores ──
    print("Computing grid scores ...")
    regions = {}
    score_grids = {}
    for name in ordered_names:
        pred = predictors[name]
        is_stoch = any(tag in name for tag in ["TRACE-Diff", "TRACE-FM", "Diff"])
        region, sg = compute_grid_scores(
            pred, x_point, y_grid, grid_res,
            is_stochastic=is_stoch, grid_n_avg=grid_n_avg,
            smooth_sigma=smooth_sigma)
        regions[name] = region
        score_grids[name] = sg
        print(f"  {name}: done")

    # ── Output ──
    out_dir = os.path.join(exp_dir, "replot")
    os.makedirs(out_dir, exist_ok=True)

    # ── Combined figure ──
    ELLIPSOID_NAMES = {"RCP", "NLE"}
    ellip_in_order = [n for n in ordered_names if n in ELLIPSOID_NAMES]
    non_ellip_in_order = [n for n in ordered_names if n not in ELLIPSOID_NAMES]
    # One panel for merged ellipsoids (if any), plus one per non-ellipsoid
    n_panels = len(non_ellip_in_order) + (1 if ellip_in_order else 0) + (1 if is_synthetic else 0)
    pw, ph = 3.2, 3.0
    fig, axes = plt.subplots(1, n_panels, figsize=(pw * n_panels + 0.5, ph + 0.3))
    if n_panels == 1:
        axes = [axes]

    pi = 0
    if is_synthetic:
        draw_density_panel(axes[pi], dataset_name, x_point,
                           y_range, grid_res, y_orig_mean, y_orig_std)
        axes[pi].set_xlim(y_range[0])
        axes[pi].set_ylim(y_range[1])
        pi += 1

    ellipsoid_done = False
    for name in ordered_names:
        if name in ELLIPSOID_NAMES:
            if ellipsoid_done:
                continue
            # Merged RCP+NLE panel
            ax = axes[pi]
            ellip_styles = ["-", "--"]
            handles = []
            for i, en in enumerate(ellip_in_order):
                fc, ec = METHOD_STYLE.get(en, FALLBACK_STYLE)
                draw_single_panel(ax, regions[en], score_grids[en],
                                  Y1, Y2, predictors[en].tau,
                                  scatter_y if i == 0 else None,
                                  fc, ec, y_true=y_true if i == 0 else None,
                                  ls=ellip_styles[i % 2])
                handles.append(plt.Line2D([0], [0], color=ec, ls=ellip_styles[i % 2],
                                          lw=2, label=en))
            ax.legend(handles=handles, loc="upper right", fontsize=7,
                      framealpha=0.8, edgecolor="gray")
            ax.set_xlim(y_range[0])
            ax.set_ylim(y_range[1])
            ax.set_title("Ellipse", fontsize=11, fontweight="bold", pad=6)
            pi += 1
            ellipsoid_done = True
        else:
            ax = axes[pi]
            fc, ec = METHOD_STYLE.get(name, FALLBACK_STYLE)
            draw_single_panel(ax, regions[name], score_grids[name],
                              Y1, Y2, predictors[name].tau,
                              scatter_y, fc, ec, y_true=y_true)
            ax.set_xlim(y_range[0])
            ax.set_ylim(y_range[1])
            ax.set_title(name, fontsize=11, fontweight="bold", color=ec, pad=6)
            pi += 1

    plt.subplots_adjust(wspace=0.05)
    cpath = os.path.join(out_dir, f"{dataset_name}_regions_x{x_index}_combined.png")
    fig.savefig(cpath, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {cpath}")

    # ── Individual figures ──
    ELLIPSOID_NAMES = {"RCP", "NLE"}
    ellipsoid_done = False
    for name in ordered_names:
        if name in ELLIPSOID_NAMES:
            if ellipsoid_done:
                continue
            # Draw merged RCP+NLE
            ellip_names = [n for n in ordered_names if n in ELLIPSOID_NAMES]
            ellip_styles = ["-", "--"]
            fig_s, ax_s = plt.subplots(1, 1, figsize=(pw + 0.2, ph + 0.2))
            handles = []
            for i, en in enumerate(ellip_names):
                fc, ec = METHOD_STYLE.get(en, FALLBACK_STYLE)
                draw_single_panel(ax_s, regions[en], score_grids[en],
                                  Y1, Y2, predictors[en].tau,
                                  scatter_y if i == 0 else None,
                                  fc, ec, y_true=y_true if i == 0 else None,
                                  ls=ellip_styles[i % 2])
                handles.append(plt.Line2D([0], [0], color=ec, ls=ellip_styles[i % 2],
                                          lw=2, label=en))
            ax_s.legend(handles=handles, loc="upper right", fontsize=8,
                        framealpha=0.8, edgecolor="gray")
            ax_s.set_xlim(y_range[0])
            ax_s.set_ylim(y_range[1])
            ax_s.set_title("Ellipse", fontsize=12, fontweight="bold", pad=6)
            fname = f"{dataset_name}_{'_'.join(n.lower() for n in ellip_names)}_x{x_index}.png"
            fpath = os.path.join(out_dir, fname)
            fig_s.savefig(fpath, dpi=dpi, bbox_inches="tight",
                          facecolor="white", edgecolor="none")
            plt.close(fig_s)
            print(f"  Saved: {fpath}")
            ellipsoid_done = True
            continue

        fig_s, ax_s = plt.subplots(1, 1, figsize=(pw + 0.2, ph + 0.2))
        fc, ec = METHOD_STYLE.get(name, FALLBACK_STYLE)
        draw_single_panel(ax_s, regions[name], score_grids[name],
                          Y1, Y2, predictors[name].tau,
                          scatter_y, fc, ec, y_true=y_true)
        ax_s.set_xlim(y_range[0])
        ax_s.set_ylim(y_range[1])
        ax_s.set_title(name, fontsize=12, fontweight="bold", color=ec, pad=6)
        fname = f"{dataset_name}_{name.replace('-', '_').lower()}_x{x_index}.png"
        fpath = os.path.join(out_dir, fname)
        fig_s.savefig(fpath, dpi=dpi, bbox_inches="tight",
                      facecolor="white", edgecolor="none")
        plt.close(fig_s)
        print(f"  Saved: {fpath}")

    if is_synthetic:
        fig_d, ax_d = plt.subplots(1, 1, figsize=(pw + 0.2, ph + 0.2))
        draw_density_panel(ax_d, dataset_name, x_point,
                           y_range, grid_res, y_orig_mean, y_orig_std)
        ax_d.set_xlim(y_range[0])
        ax_d.set_ylim(y_range[1])
        fpath = os.path.join(out_dir, f"{dataset_name}_true_density_x{x_index}.png")
        fig_d.savefig(fpath, dpi=dpi, bbox_inches="tight",
                      facecolor="white", edgecolor="none")
        plt.close(fig_d)
        print(f"  Saved: {fpath}")

    # ── Taxi / Hurricane maps ──
    yd = data["try"].shape[1]
    if dataset_name == "taxi" and y_orig_mean is not None and yd == 2:
        print("\nGenerating taxi maps ...")
        pred_list = []
        reverse_rename = {v: k for k, v in RENAME.items()}
        for name in ordered_names:
            orig = reverse_rename.get(name, name)
            if orig in raw_results:
                pred_list.append((predictors[name], raw_results[orig]))
        from plotting import plot_taxi_map
        plot_taxi_map(
            pred_list, x_point, y_orig_mean, y_orig_std,
            raw_results,
            x_orig_mean=x_orig_mean, x_orig_std=x_orig_std,
            y_range=y_range, y_true=y_true,
            train_x=data["trx"], train_y=data["try"],
            grid_res=grid_res, grid_n_avg=grid_n_avg,
            smooth_sigma=smooth_sigma,
            save_dir=out_dir, prefix=f"{dataset_name}_x{x_index}")

    if dataset_name == "hurricane" and y_orig_mean is not None and x_orig_mean is not None:
        print("\nGenerating hurricane maps ...")
        all_preds = {}
        display_results = {}
        reverse_rename = {v: k for k, v in RENAME.items()}
        for name in ordered_names:
            all_preds[name] = predictors[name]
            orig = reverse_rename.get(name, name)
            if orig in raw_results:
                display_results[name] = raw_results[orig]
        from plotting import plot_hurricane_map
        plot_hurricane_map(
            all_preds, x_point, y_orig_mean, y_orig_std,
            x_orig_mean, x_orig_std,
            display_results,
            grid_res=grid_res, grid_n_avg=grid_n_avg,
            smooth_sigma=smooth_sigma,
            save_dir=out_dir, prefix=dataset_name)

    print(f"\nDone → {out_dir}/")


def main():
    p = argparse.ArgumentParser(description="Re-draw region plots from checkpoint")
    p.add_argument("--exp_dir", type=str, required=True)
    p.add_argument("--x_index", type=int, default=None)
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--grid_res", type=int, default=200)
    p.add_argument("--grid_n_avg", type=int, default=3)
    p.add_argument("--smooth_sigma", type=float, default=1.5)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    make_plots(args.exp_dir, x_index=args.x_index, rep_idx=args.rep,
               grid_res=args.grid_res, grid_n_avg=args.grid_n_avg,
               smooth_sigma=args.smooth_sigma, device=args.device,
               dpi=args.dpi)


if __name__ == "__main__":
    main()