#!/usr/bin/env python3
"""Ablation: MC budget (|T| × R) vs Volume for FM-Path and Diff-Denoise.

Two modes:
  (A) --load_from: Load pre-trained models from existing experiment directory
      (e.g. experiments/pinwheel_s0 with repeat_000..repeat_019).
      Skips training entirely, only sweeps score + calibration + volume.

  (B) Default: Train models from scratch per repeat (original behavior).

Usage:
    # Load from existing 20-repeat experiment (recommended)
    python ablation_mc_budget.py --datasets pinwheel,twomoons \
        --load_from experiments/pinwheel_s0,experiments/twomoons_s0 \
        --device cuda

    # Train from scratch (synthetic datasets)
    python ablation_mc_budget.py --datasets pinwheel,twomoons --n_repeats 5 --device cuda

    # Train from scratch (with taxi real dataset)
    python ablation_mc_budget.py --datasets pinwheel,twomoons,taxi \
        --n_repeats 5 --device cuda

    # Replot only
    python ablation_mc_budget.py --plot_only

    # Replot with O(1/B) reference line on volume subplot
    python ablation_mc_budget.py --plot_only --vol_ref

Output:
    experiments/ablation_mc_budget/ablation_mc_budget_{dataset}.png/pdf
    experiments/ablation_mc_budget/ablation_mc_budget_{dataset}.json
    experiments/ablation_mc_budget/ablation_mc_budget_combined.png/pdf
"""

import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from config import Config
from conformal import conformal_quantile
from volume import mc_volume_grid
from training import train_diffusion, train_flow_matching, clear_gpu
from splitting import split_data
from datasets import DATASETS, REAL_DATASETS

# ── Synthetic dataset names (Y normalization done externally) ──
SYNTHETIC_DATASETS = {
    "spiral", "ring", "mixture_gaussian", "moon", "heterogeneous",
    "banana", "funnel", "pinwheel", "checkerboard", "twomoons",
}

# ── Default taxi CSV path ──
DEFAULT_TAXI_CSV = "/home/zfang11/Data/nyc.csv"

# ── Budget grid ──
BUDGET_GRID = [
    # (timesteps, repeats)  -> total budget
    (2,   5),     # 10
    (4,   5),     # 20
    (5,   8),     # 40
    (6,  10),     # 60
    (10,  8),     # 80
    (10, 10),     # 100
    (15,  8),     # 120
    (16, 10),     # 160
    (20, 10),     # 200
]


# ======================================================================
# Helper: load dataset (synthetic or real)
# ======================================================================

def _load_dataset(dataset_name, n_total, seed, taxi_csv=None):
    """Load dataset, return (X, Y_normalized, y_std_prod).

    For synthetic datasets: normalize Y externally, y_std_prod from raw Y.
    For real datasets: loader normalizes internally, returns y_std_prod.
    """
    is_synthetic = dataset_name in SYNTHETIC_DATASETS

    if is_synthetic:
        X, Y = DATASETS[dataset_name](n_total, seed)
        _y_mean = Y.mean(dim=0)
        _y_std = Y.std(dim=0).clamp(min=1e-8)
        y_std_prod = float(_y_std.prod())
        Y = (Y - _y_mean) / _y_std
        return X, Y, y_std_prod

    # Real dataset
    if dataset_name not in REAL_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Available: {list(DATASETS.keys()) + list(REAL_DATASETS.keys())}")

    kwargs = {"seed": seed}
    if dataset_name == "taxi":
        kwargs["csv_path"] = taxi_csv or DEFAULT_TAXI_CSV
    if dataset_name in ("taxi",):
        kwargs["n_subset"] = n_total

    result = REAL_DATASETS[dataset_name](**kwargs)
    # result is tuple: (X, Y, y_std_prod, vol_dims, ...)
    X, Y = result[0], result[1]
    y_std_prod = float(result[2])
    return X, Y, y_std_prod


# ======================================================================
# Core: run budget sweep for one (dataset, repeat) given models + data
# ======================================================================

def run_budget_sweep(diff_model, fm_model, data, cfg, seed, device,
                     results, rep_label="", y_std_prod=1.0):
    """Sweep BUDGET_GRID for one repeat. Mutates `results` dict in-place.

    y_std_prod: product of per-dimension Y stds, used to rescale volume
                from normalized space back to original Y-space.
    """
    n_vol = min(cfg.n_vol, len(data["tsx"]))
    x_vol = data["tsx"][:n_vol]

    from scores.diffusion_scores import DiffusionDenoiseScore
    from scores.fm_scores import FMPathScore

    for ts, reps in BUDGET_GRID:
        budget = ts * reps
        bk = str(budget)
        print(f"  {rep_label}Budget {budget} (T={ts}, R={reps}) ...")

        # --- Diff-Denoise ---
        diff_score = DiffusionDenoiseScore(
            diff_model, device,
            n_timesteps=ts, n_repeats=reps, seed=seed)

        cal_s_diff = diff_score.compute(data["cx"], data["cy"])
        tau_diff = conformal_quantile(cal_s_diff, cfg.alpha)
        test_s_diff = diff_score.compute(data["tsx"], data["tsy"])
        cov_diff = float((test_s_diff <= tau_diff).mean())

        vol_diff, _ = mc_volume_grid(
            diff_score, tau_diff, x_vol, data["try"], device,
            n_mc=cfg.n_mc, margin=cfg.vol_margin,
            gen_model=diff_model, n_probe=cfg.vol_n_probe,
            verbose_name=f"Diff(T={ts},R={reps})", seed=seed)
        vol_diff *= y_std_prod  # rescale to original Y-space

        results["Diff-Denoise"].setdefault(bk, {"volume": [], "coverage": [], "score_std": []})
        results["Diff-Denoise"][bk]["volume"].append(vol_diff)
        results["Diff-Denoise"][bk]["coverage"].append(cov_diff)
        results["Diff-Denoise"][bk]["score_std"].append(float(cal_s_diff.std()))
        print(f"    Diff: cov={cov_diff:.3f}  vol={vol_diff:.4f}  "
              f"tau={tau_diff:.4f}  score_std={cal_s_diff.std():.4f}")

        # --- FM-Path ---
        fm_score = FMPathScore(
            fm_model, device,
            n_timesteps=ts, n_repeats=reps, seed=seed)

        cal_s_fm = fm_score.compute(data["cx"], data["cy"])
        tau_fm = conformal_quantile(cal_s_fm, cfg.alpha)
        test_s_fm = fm_score.compute(data["tsx"], data["tsy"])
        cov_fm = float((test_s_fm <= tau_fm).mean())

        vol_fm, _ = mc_volume_grid(
            fm_score, tau_fm, x_vol, data["try"], device,
            n_mc=cfg.n_mc, margin=cfg.vol_margin,
            gen_model=fm_model, n_probe=cfg.vol_n_probe,
            verbose_name=f"FM(T={ts},R={reps})", seed=seed)
        vol_fm *= y_std_prod  # rescale to original Y-space

        results["FM-Path"].setdefault(bk, {"volume": [], "coverage": [], "score_std": []})
        results["FM-Path"][bk]["volume"].append(vol_fm)
        results["FM-Path"][bk]["coverage"].append(cov_fm)
        results["FM-Path"][bk]["score_std"].append(float(cal_s_fm.std()))
        print(f"    FM:   cov={cov_fm:.3f}  vol={vol_fm:.4f}  "
              f"tau={tau_fm:.4f}  score_std={cal_s_fm.std():.4f}")


# ======================================================================
# Mode A: Load pre-trained models from existing experiment directory
# ======================================================================

def run_ablation_from_checkpoints(dataset_name, exp_dir, device="cuda",
                                  outdir="./experiments/ablation_mc_budget",
                                  taxi_csv=None):
    """Load models + data from existing experiment, sweep budget grid."""
    os.makedirs(outdir, exist_ok=True)
    json_path = os.path.join(outdir, f"ablation_mc_budget_{dataset_name}.json")

    # Find all repeat directories
    repeat_dirs = sorted(glob.glob(os.path.join(exp_dir, "repeat_*")))
    n_repeats = len(repeat_dirs)
    if n_repeats == 0:
        print(f"  ERROR: No repeat_* dirs found in {exp_dir}")
        return None
    print(f"  Found {n_repeats} repeats in {exp_dir}")

    # Load config from experiment
    config_path = os.path.join(exp_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            exp_cfg = json.load(f)
        if "config" in exp_cfg:
            exp_cfg = exp_cfg["config"]
    else:
        exp_cfg = {}

    cfg = Config(
        dataset=dataset_name,
        n_total=exp_cfg.get("n_total", 30000),
        alpha=0.1,
        n_mc=5000,
        n_vol=30,
        vol_margin=0.3,
        vol_n_probe=500,
        device=device,
    )

    # Init or resume results
    results = {
        "dataset": dataset_name,
        "budgets": [ts * reps for ts, reps in BUDGET_GRID],
        "grid": [(ts, reps) for ts, reps in BUDGET_GRID],
        "FM-Path": {},
        "Diff-Denoise": {},
    }
    start_rep = 0
    if os.path.exists(json_path):
        with open(json_path) as f:
            old = json.load(f)
        if old.get("budgets") == results["budgets"]:
            results = old
            first_bk = str(results["budgets"][0])
            if first_bk in results["Diff-Denoise"]:
                start_rep = len(results["Diff-Denoise"][first_bk].get("volume", []))
            print(f"  Resuming from repeat {start_rep + 1}")
        else:
            print(f"  Budget grid changed, starting fresh")

    from models.diffusion import ConditionalDDPM
    from models.flow_matching import ConditionalFlowMatching

    for rep_idx in range(start_rep, n_repeats):
        rep_dir = repeat_dirs[rep_idx]
        seed = rep_idx
        print(f"\n{'='*65}")
        print(f"  [{dataset_name}] REPEAT {rep_idx+1}/{n_repeats}  (loading from {rep_dir})")
        print(f"{'='*65}")

        # -- Load data split --
        data_path = os.path.join(rep_dir, "data_split.pt")
        if not os.path.exists(data_path):
            print(f"  WARNING: {data_path} not found, skipping")
            continue
        data = torch.load(data_path, map_location="cpu")

        xd = data["trx"].shape[1]
        yd = data["try"].shape[1]
        y_mean = data["try"].mean(0)
        y_std = data["try"].std(0)

        # -- Load Diff model --
        diff_path = os.path.join(rep_dir, "diff_model.pt")
        if not os.path.exists(diff_path):
            print(f"  WARNING: {diff_path} not found, skipping")
            continue
        diff_ckpt = torch.load(diff_path, map_location="cpu")
        sd = diff_ckpt if isinstance(diff_ckpt, dict) and any("blocks.0.linear1.weight" in k for k in diff_ckpt) else \
             diff_ckpt.get("model_state_dict", diff_ckpt)
        hidden_dim = sd[[k for k in sd if "linear1.weight" in k][0]].shape[0]
        n_blocks = len([k for k in sd if k.endswith(".linear1.weight") and "blocks." in k])

        diff_model = ConditionalDDPM(
            yd, xd, T=sd["alpha_bar"].shape[0] if "alpha_bar" in sd else exp_cfg.get("diff_T", 1000),
            hidden_dim=hidden_dim, n_blocks=n_blocks,
            schedule=exp_cfg.get("diff_schedule", "cosine"),
            cfg_drop_prob=exp_cfg.get("diff_cfg_drop_prob", 0.0))
        diff_model.set_normalization(y_mean, y_std)
        diff_model.load_state_dict(sd)
        diff_model.eval()

        # -- Load FM model --
        fm_path = os.path.join(rep_dir, "fm_model.pt")
        if not os.path.exists(fm_path):
            print(f"  WARNING: {fm_path} not found, skipping")
            continue
        fm_ckpt = torch.load(fm_path, map_location="cpu")
        fm_sd = fm_ckpt if isinstance(fm_ckpt, dict) and any("blocks.0.linear1.weight" in k for k in fm_ckpt) else \
                fm_ckpt.get("model_state_dict", fm_ckpt)
        fm_hidden = fm_sd[[k for k in fm_sd if "linear1.weight" in k][0]].shape[0]
        fm_nblocks = len([k for k in fm_sd if k.endswith(".linear1.weight") and "blocks." in k])

        fm_model = ConditionalFlowMatching(
            yd, xd, hidden_dim=fm_hidden, n_blocks=fm_nblocks,
            sigma_min=exp_cfg.get("fm_sigma_min", 1e-4),
            cfg_drop_prob=exp_cfg.get("fm_cfg_drop_prob", 0.0))
        fm_model.set_normalization(y_mean, y_std)
        fm_model.load_state_dict(fm_sd)
        fm_model.eval()

        # -- Compute y_std_prod for volume rescaling --
        if "y_orig_std" in data:
            _y_std_prod = float(np.prod(data["y_orig_std"].numpy()))
        elif "_y_std_prod" in data:
            _y_std_prod = float(data["_y_std_prod"])
        elif "y_std_prod" in data:
            _y_std_prod = float(data["y_std_prod"])
        else:
            _, _, _y_std_prod = _load_dataset(
                dataset_name, exp_cfg.get("n_total", 30000), rep_idx,
                taxi_csv=taxi_csv)
            print(f"    y_std_prod recomputed = {_y_std_prod:.4f}")

        run_budget_sweep(diff_model, fm_model, data, cfg, seed, device,
                         results, rep_label=f"[rep {rep_idx}] ",
                         y_std_prod=_y_std_prod)

        diff_model.cpu()
        fm_model.cpu()
        clear_gpu()

        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  [{dataset_name}] Repeat {rep_idx+1} done. Saved -> {json_path}")

    return results


# ======================================================================
# Mode B: Train from scratch (original behavior)
# ======================================================================

def run_ablation_train(dataset_name, n_repeats=5, device="cuda",
                       outdir="./experiments/ablation_mc_budget",
                       taxi_csv=None):
    """Train models from scratch, then sweep budget grid."""
    os.makedirs(outdir, exist_ok=True)
    json_path = os.path.join(outdir, f"ablation_mc_budget_{dataset_name}.json")

    results = {
        "dataset": dataset_name,
        "budgets": [ts * reps for ts, reps in BUDGET_GRID],
        "grid": [(ts, reps) for ts, reps in BUDGET_GRID],
        "FM-Path": {},
        "Diff-Denoise": {},
    }

    start_rep = 0
    if os.path.exists(json_path):
        with open(json_path) as f:
            old = json.load(f)
        if old.get("budgets") == results["budgets"]:
            results = old
            first_bk = str(results["budgets"][0])
            if first_bk in results["Diff-Denoise"]:
                start_rep = len(results["Diff-Denoise"][first_bk].get("volume", []))
            print(f"  Resuming from repeat {start_rep + 1}")
        else:
            print(f"  Budget grid changed, starting fresh")

    cfg = Config(
        dataset=dataset_name,
        n_total=30000,
        diff_epochs=2000, diff_patience=0,
        fm_epochs=2000, fm_patience=0,
        alpha=0.1, n_mc=5000, n_vol=30,
        vol_margin=0.3, vol_n_probe=500,
        device=device,
    )

    from models.diffusion import ConditionalDDPM
    from models.flow_matching import ConditionalFlowMatching

    for rep in range(start_rep, n_repeats):
        seed = rep
        print(f"\n{'='*65}")
        print(f"  [{dataset_name}] REPEAT {rep+1}/{n_repeats}  (seed={seed})")
        print(f"{'='*65}")

        # -- Data --
        torch.manual_seed(seed)
        np.random.seed(seed)

        X, Y, _y_std_prod = _load_dataset(dataset_name, cfg.n_total, seed,
                                           taxi_csv=taxi_csv)

        data = split_data(X, Y, seed)
        xd, yd = X.shape[1], Y.shape[1]
        y_mean = data["try"].mean(0)
        y_std = data["try"].std(0)

        # -- Train Diff --
        print("\n  Training Diffusion ...")
        diff_model = ConditionalDDPM(
            yd, xd, T=cfg.diff_T, hidden_dim=cfg.hidden_dim,
            n_blocks=cfg.diff_n_blocks,
            schedule=cfg.diff_schedule,
            beta_min=cfg.diff_beta_min, beta_max=cfg.diff_beta_max,
            cfg_drop_prob=cfg.diff_cfg_drop_prob)
        diff_model.set_normalization(y_mean, y_std)
        train_diffusion(diff_model, data["trx"], data["try"],
                        epochs=cfg.diff_epochs, batch_size=cfg.batch_size,
                        lr=cfg.diff_lr, weight_decay=cfg.weight_decay,
                        grad_clip=cfg.grad_clip, ema_decay=cfg.ema_decay,
                        patience=cfg.diff_patience, device=device)

        # -- Train FM --
        print("  Training FM ...")
        fm_model = ConditionalFlowMatching(
            yd, xd, hidden_dim=cfg.hidden_dim, n_blocks=cfg.fm_n_blocks,
            sigma_min=cfg.fm_sigma_min,
            cfg_drop_prob=cfg.fm_cfg_drop_prob)
        fm_model.set_normalization(y_mean, y_std)
        train_flow_matching(fm_model, data["trx"], data["try"],
                            epochs=cfg.fm_epochs, batch_size=cfg.batch_size,
                            lr=cfg.fm_lr, weight_decay=cfg.weight_decay,
                            grad_clip=cfg.grad_clip, ema_decay=cfg.ema_decay,
                            patience=cfg.fm_patience, device=device)

        # -- Sweep --
        run_budget_sweep(diff_model, fm_model, data, cfg, seed, device,
                         results, rep_label="",
                         y_std_prod=_y_std_prod)

        diff_model.cpu()
        fm_model.cpu()
        clear_gpu()

        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  [{dataset_name}] Repeat {rep+1} done. Saved.")

    return results


# ======================================================================
# Plotting
# ======================================================================

def _setup_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',
        'font.size': 9,
        'axes.linewidth': 0.6,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'lines.linewidth': 1.4,
        'lines.markersize': 5,
    })

METHODS_STYLE = [
    ("Diff-Denoise", "#2471a3", "s", "TRACE-Diff"),
    ("FM-Path",      "#c0392b", "o", "TRACE-FM"),
]


def _extract_stats(results, method, metric):
    """Extract mean and std arrays for a given method and metric."""
    budgets = results["budgets"]
    means, stds = [], []
    for bk in [str(b) for b in budgets]:
        vals = results[method].get(bk, {}).get(metric, [])
        means.append(np.mean(vals) if vals else np.nan)
        stds.append(np.std(vals) if len(vals) > 1 else 0)
    return np.array(means), np.array(stds)


def plot_ablation_single(ds_name, results, save_dir, show_vol_ref=False):
    """Plot volume and score_std vs budget for one dataset."""
    _setup_style()
    budgets = results["budgets"]

    fig, (ax_vol, ax_std) = plt.subplots(2, 1, figsize=(4.5, 5.0), sharex=True)

    for method, color, marker, label in METHODS_STYLE:
        vol_m, vol_s = _extract_stats(results, method, "volume")
        std_m, std_s = _extract_stats(results, method, "score_std")

        ax_vol.plot(budgets, vol_m, marker=marker, color=color,
                    label=label, zorder=3)
        if vol_s.max() > 0:
            ax_vol.fill_between(budgets, vol_m - vol_s, vol_m + vol_s,
                                color=color, alpha=0.15, zorder=1)

        ax_std.plot(budgets, std_m, marker=marker, color=color,
                    label=label, zorder=3)
        if std_s.max() > 0:
            ax_std.fill_between(budgets, std_m - std_s, std_m + std_s,
                                color=color, alpha=0.15, zorder=1)

    # Optional reference line O(1/sqrt(B)) on volume
    if show_vol_ref:
        b_arr_vol = np.array(budgets, dtype=float)
        diff_vol0 = results["Diff-Denoise"].get(str(budgets[0]), {}).get("volume", [])
        if diff_vol0:
            c_vol = np.mean(diff_vol0) * np.sqrt(budgets[0])
            ref_vol = c_vol / np.sqrt(b_arr_vol)
            ax_vol.plot(b_arr_vol, ref_vol, color='#888888', ls='--', lw=0.9,
                        alpha=0.7, label='$O(1/\\sqrt{B})$', zorder=1)

    # Reference line O(1/sqrt(B)) on score std
    b_arr = np.array(budgets, dtype=float)
    diff_std0_vals = results["Diff-Denoise"].get(str(budgets[0]), {}).get("score_std", [])
    if diff_std0_vals:
        c = np.mean(diff_std0_vals) * np.sqrt(budgets[0])
        ref = c / np.sqrt(b_arr)
        ax_std.plot(b_arr, ref, color='#888888', ls='--', lw=0.9,
                    alpha=0.7, label='$O(1/\\sqrt{B})$', zorder=1)

    ax_vol.set_ylabel('Volume')
    ax_vol.set_title(ds_name.capitalize(), fontweight='bold', pad=6)
    ax_vol.legend(loc='upper right', framealpha=0.8, edgecolor='none')
    ax_vol.grid(True, alpha=0.25, linewidth=0.4)
    ax_vol.tick_params(direction='in')

    ax_std.set_ylabel('Score Std Dev')
    ax_std.set_xlabel('MC budget ($|\\mathcal{T}| \\times R$)')
    ax_std.legend(loc='upper right', framealpha=0.8, edgecolor='none')
    ax_std.grid(True, alpha=0.25, linewidth=0.4)
    ax_std.tick_params(direction='in')

    plt.tight_layout(h_pad=1.0)
    for ext in ['png', 'pdf']:
        path = os.path.join(save_dir, f"ablation_mc_budget_{ds_name}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: ablation_mc_budget_{ds_name}.png/pdf")


def plot_ablation_combined(all_results, save_dir, show_vol_ref=False):
    """Plot all datasets side by side (volume top, score_std bottom)."""
    _setup_style()
    n_ds = len(all_results)
    fig, axes = plt.subplots(2, n_ds, figsize=(4.0 * n_ds, 5.0))
    if n_ds == 1:
        axes = axes.reshape(2, 1)

    for i, (ds_name, results) in enumerate(all_results.items()):
        ax_vol = axes[0, i]
        ax_std = axes[1, i]
        budgets = results["budgets"]
        b_arr = np.array(budgets, dtype=float)

        for method, color, marker, label in METHODS_STYLE:
            vol_m, vol_s = _extract_stats(results, method, "volume")
            std_m, _ = _extract_stats(results, method, "score_std")

            ax_vol.plot(budgets, vol_m, marker=marker, color=color,
                        label=label, zorder=3)
            if vol_s.max() > 0:
                ax_vol.fill_between(budgets, vol_m - vol_s, vol_m + vol_s,
                                    color=color, alpha=0.15, zorder=1)
            ax_std.plot(budgets, std_m, marker=marker, color=color,
                        label=label, zorder=3)

        if show_vol_ref:
            diff_vol0 = results["Diff-Denoise"].get(str(budgets[0]), {}).get("volume", [])
            if diff_vol0:
                c_vol = np.mean(diff_vol0) * np.sqrt(budgets[0])
                ax_vol.plot(b_arr, c_vol / np.sqrt(b_arr), color='#888888', ls='--',
                            lw=0.9, alpha=0.7, label='$O(1/\\sqrt{B})$', zorder=1)

        diff_std0 = results["Diff-Denoise"].get(str(budgets[0]), {}).get("score_std", [])
        if diff_std0:
            c = np.mean(diff_std0) * np.sqrt(budgets[0])
            ax_std.plot(b_arr, c / np.sqrt(b_arr), color='#888888', ls='--',
                        lw=0.9, alpha=0.7, label='$O(1/\\sqrt{B})$', zorder=1)

        ax_vol.set_title(ds_name.capitalize(), fontweight='bold', pad=6)
        ax_vol.grid(True, alpha=0.25, linewidth=0.4)
        ax_vol.tick_params(direction='in')
        ax_std.grid(True, alpha=0.25, linewidth=0.4)
        ax_std.tick_params(direction='in')
        ax_std.set_xlabel('MC budget ($|\\mathcal{T}| \\times R$)')

        if i == 0:
            ax_vol.set_ylabel('Volume')
            ax_std.set_ylabel('Score Std Dev')
        ax_vol.legend(loc='upper right', framealpha=0.8, edgecolor='none')
        ax_std.legend(loc='upper right', framealpha=0.8, edgecolor='none')

    plt.tight_layout(h_pad=1.0, w_pad=1.5)
    for ext in ['png', 'pdf']:
        path = os.path.join(save_dir, f"ablation_mc_budget_combined.{ext}")
        fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: ablation_mc_budget_combined.png/pdf")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MC Budget ablation: volume vs timesteps x repeats")
    parser.add_argument("--datasets", type=str, default="pinwheel,twomoons",
                        help="Comma-separated dataset names (synthetic or real)")
    parser.add_argument("--load_from", type=str, default=None,
                        help="Comma-separated experiment dirs to load models from "
                             "(one per dataset, same order as --datasets)")
    parser.add_argument("--n_repeats", type=int, default=5,
                        help="Number of repeats (only used without --load_from)")
    parser.add_argument("--taxi_csv", type=str, default=DEFAULT_TAXI_CSV,
                        help="Path to NYC taxi CSV file")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--outdir", type=str,
                        default="./experiments/ablation_mc_budget")
    parser.add_argument("--plot_only", action="store_true",
                        help="Only replot from existing JSON files")
    parser.add_argument("--vol_ref", action="store_true",
                        help="Show O(1/B) reference line on volume subplot")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset_names = [s.strip() for s in args.datasets.split(",")]

    # Parse load_from directories
    load_dirs = None
    if args.load_from:
        load_dirs = [s.strip() for s in args.load_from.split(",")]
        if len(load_dirs) == 1 and len(dataset_names) > 1:
            base = load_dirs[0]
            load_dirs = [os.path.join(base, f"{ds}_s0") for ds in dataset_names]
        assert len(load_dirs) == len(dataset_names), \
            f"--load_from has {len(load_dirs)} dirs but --datasets has {len(dataset_names)}"

    all_results = {}
    for idx, ds_name in enumerate(dataset_names):
        json_path = os.path.join(args.outdir, f"ablation_mc_budget_{ds_name}.json")

        if args.plot_only:
            if not os.path.exists(json_path):
                print(f"  No results for {ds_name}, skipping")
                continue
            with open(json_path) as f:
                all_results[ds_name] = json.load(f)
        elif load_dirs:
            all_results[ds_name] = run_ablation_from_checkpoints(
                ds_name, load_dirs[idx],
                device=device, outdir=args.outdir,
                taxi_csv=args.taxi_csv)
        else:
            all_results[ds_name] = run_ablation_train(
                ds_name, n_repeats=args.n_repeats,
                device=device, outdir=args.outdir,
                taxi_csv=args.taxi_csv)

    # Plot
    if all_results:
        for ds_name, results in all_results.items():
            if results:
                plot_ablation_single(ds_name, results, args.outdir,
                                     show_vol_ref=args.vol_ref)
        if len(all_results) > 1:
            plot_ablation_combined(
                {k: v for k, v in all_results.items() if v},
                args.outdir, show_vol_ref=args.vol_ref)


if __name__ == "__main__":
    main()