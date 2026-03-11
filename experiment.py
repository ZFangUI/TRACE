"""
Experiment runner for Generative Conformal Prediction.

Z-space methods:  NF-Ball, NF-NLL, Diff-Denoise, FM-Path
Y-space baselines: RCP, NLE, PCP-{NF,Diff,FM}, DistSplit, CQR, MCQR
"""

import os
import time
import json
import numpy as np
import torch

from config import Config
from datasets import DATASETS, REAL_DATASETS, PLOT_DATASETS
from splitting import split_data
from models.nf import NFModel
from models.diffusion import ConditionalDDPM
from models.flow_matching import ConditionalFlowMatching
from scores import (NFBallScore, NFNLLScore, DiffusionDenoiseScore, FMPathScore)
from conformal import ConformalPredictor
from training import train_nf, train_diffusion, train_flow_matching, clear_gpu
from volume import mc_volume_grid
from baselines import (
    sample_ys_nf, predict_mean_nf, sample_ys_diff, sample_ys_fm,
    RCP, NLE, PCP, DistSplit, CQR, MCQR,
)
from plotting import (
    plot_prediction_regions_with_eval,
    plot_comparison_bars,
    plot_score_distributions,
    plot_sample_quality,
    plot_results_table,
    plot_pareto,
    plot_repeats_violin,
    plot_taxi_map,
    plot_hurricane_map,
)

ZSPACE_NAMES = ["NF-Ball", "NF-NLL", "Diff-Denoise", "FM-Path"]

BASELINE_NAMES = ["RCP", "NLE", "PCP-NF", "PCP-Diff", "PCP-FM",
                  "DistSplit", "CQR", "MCQR"]

ALL_METHODS = ZSPACE_NAMES + BASELINE_NAMES

# Method → which model(s) it depends on
METHOD_MODEL_MAP = {
    "NF-Ball": {"NF"}, "NF-NLL": {"NF"},
    "Diff-Denoise": {"Diff"},
    "FM-Path": {"FM"},
    "RCP": {"NF"}, "NLE": {"NF"},
    "DistSplit": {"NF"}, "CQR": {"NF"}, "MCQR": {"NF"},
    "PCP-NF": {"NF"}, "PCP-Diff": {"Diff"}, "PCP-FM": {"FM"},
}

ALL_METHOD_NAMES = list(METHOD_MODEL_MAP.keys())


def _resolve_methods(cfg):
    """Return the set of method names to run based on cfg.methods and cfg.baselines.

    Priority:
      1. If cfg.methods is set (not None), use exactly those methods.
         cfg.baselines is ignored in this case.
      2. Otherwise, run all Z-space methods, plus baselines if cfg.baselines is True.
    """
    if cfg.methods is not None:
        # Validate
        unknown = set(cfg.methods) - set(ALL_METHODS)
        if unknown:
            raise ValueError(
                f"Unknown method(s): {unknown}. "
                f"Available: {ALL_METHODS}")
        return list(cfg.methods)
    # Default: all Z-space + optionally all baselines
    methods = list(ZSPACE_NAMES)
    if cfg.baselines:
        # Determine PCP variants from config
        pcp_names = [f"PCP-{g.strip()}" for g in cfg.pcp_models.split(",")]
        baseline_list = ["RCP", "NLE"] + pcp_names + ["DistSplit", "CQR", "MCQR"]
        methods += baseline_list
    return methods


def _methods_to_models(method_names):
    """Given a list of method names, return set of models needed."""
    models = set()
    for m in method_names:
        models |= METHOD_MODEL_MAP.get(m, set())
    return models


def _repeat_dir(outdir, rep_idx):
    """Directory for a single repeat's checkpoint."""
    return os.path.join(outdir, f"repeat_{rep_idx:03d}")


def _save_checkpoint(rep_dir, nf_model, diff_model, fm_model,
                     data, y_orig_mean, y_orig_std, y_std_prod, vol_dims,
                     x_orig_mean=None, x_orig_std=None):
    """Save models + data split for one repeat."""
    os.makedirs(rep_dir, exist_ok=True)
    if nf_model is not None:
        torch.save(nf_model.state_dict(), os.path.join(rep_dir, "nf_model.pt"))
    if diff_model is not None:
        torch.save(diff_model.state_dict(), os.path.join(rep_dir, "diff_model.pt"))
    if fm_model is not None:
        torch.save(fm_model.state_dict(), os.path.join(rep_dir, "fm_model.pt"))
    # Save data split + normalization info
    save_data = {k: v for k, v in data.items()}
    save_data["_y_orig_mean"] = y_orig_mean
    save_data["_y_orig_std"] = y_orig_std
    save_data["_y_std_prod"] = y_std_prod
    save_data["_vol_dims"] = vol_dims
    save_data["_x_orig_mean"] = x_orig_mean
    save_data["_x_orig_std"] = x_orig_std
    torch.save(save_data, os.path.join(rep_dir, "data_split.pt"))


def _load_checkpoint(rep_dir, cfg, device):
    """Load models + data split from checkpoint.

    Returns: (nf_model, diff_model, fm_model, data, y_orig_mean, y_orig_std,
              y_std_prod, vol_dims, x_orig_mean, x_orig_std)
    """
    save_data = torch.load(os.path.join(rep_dir, "data_split.pt"),
                           map_location="cpu", weights_only=False)
    y_orig_mean = save_data.pop("_y_orig_mean", None)
    y_orig_std = save_data.pop("_y_orig_std", None)
    y_std_prod = save_data.pop("_y_std_prod", None)
    vol_dims = save_data.pop("_vol_dims", None)
    x_orig_mean = save_data.pop("_x_orig_mean", None)
    x_orig_std = save_data.pop("_x_orig_std", None)
    data = save_data

    xd = int(data["trx"].shape[1])
    yd = int(data["try"].shape[1])
    y_mean = data["try"].mean(0)
    y_std = data["try"].std(0)

    nf_model = NFModel(xd, yd, cfg.cond_dim, cfg.hidden_dim,
                        cfg.nf_n_layers, cfg.nf_s_clamp,
                        flow_type=cfg.nf_flow_type,
                        n_bins=cfg.nf_n_bins, tail_bound=cfg.nf_tail_bound,
                        cond_net_type=cfg.nf_cond_net)
    nf_model.load_state_dict(
        torch.load(os.path.join(rep_dir, "nf_model.pt"),
                   map_location="cpu", weights_only=True))

    diff_model = ConditionalDDPM(
        yd, xd, T=cfg.diff_T, hidden_dim=cfg.hidden_dim,
        n_blocks=cfg.diff_n_blocks, schedule=cfg.diff_schedule,
        beta_min=cfg.diff_beta_min, beta_max=cfg.diff_beta_max,
        cfg_drop_prob=cfg.diff_cfg_drop_prob)
    diff_model.set_normalization(y_mean, y_std)
    diff_model.load_state_dict(
        torch.load(os.path.join(rep_dir, "diff_model.pt"),
                   map_location="cpu", weights_only=True))

    fm_model = ConditionalFlowMatching(
        yd, xd, hidden_dim=cfg.hidden_dim, n_blocks=cfg.fm_n_blocks,
        sigma_min=cfg.fm_sigma_min, cfg_drop_prob=cfg.fm_cfg_drop_prob)
    fm_model.set_normalization(y_mean, y_std)
    fm_model.load_state_dict(
        torch.load(os.path.join(rep_dir, "fm_model.pt"),
                   map_location="cpu", weights_only=True))

    return (nf_model, diff_model, fm_model, data,
            y_orig_mean, y_orig_std, y_std_prod, vol_dims,
            x_orig_mean, x_orig_std)


def _save_repeat_results(rep_dir, all_results, time_dict=None,
                         train_time=None, score_time=None):
    """Save one repeat's method results to JSON (no raw scores arrays)."""
    obj = {}
    for name, r in all_results.items():
        obj[name] = {k: (round(v, 6) if isinstance(v, float) else v)
                     for k, v in r.items() if k != "scores"}
    if time_dict:
        obj["_time_seconds"] = {k: round(v, 2) for k, v in time_dict.items()}
    if train_time:
        obj["_training_seconds"] = {k: round(v, 2) for k, v in train_time.items()}
    if score_time:
        obj["_score_seconds"] = {k: round(v, 2) for k, v in score_time.items()}
    path = os.path.join(rep_dir, "results.json")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    return path


def _load_repeat_results(rep_dir):
    """Load one repeat's results. Returns (all_results, time_dict) or None."""
    path = os.path.join(rep_dir, "results.json")
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        obj = json.load(f)
    time_dict = obj.pop("_time_seconds", {})
    return obj, time_dict


def _repeat_is_complete(outdir, rep_idx):
    """Check if a repeat has saved results (checkpoint complete)."""
    rep_dir = _repeat_dir(outdir, rep_idx)
    return os.path.exists(os.path.join(rep_dir, "results.json"))


def _single_run(cfg, seed, device, outdir, verbose, rep_idx=0):
    """Single conformal run: train -> calibrate -> evaluate -> volume.

    Saves checkpoint (models + data) and per-repeat results.
    """
    rep_dir = _repeat_dir(outdir, rep_idx)
    os.makedirs(rep_dir, exist_ok=True)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ── Data ──
    y_std_prod = None
    vol_dims = None
    y_orig_mean = None
    y_orig_std = None
    x_orig_mean = None
    x_orig_std = None

    if cfg.dataset in DATASETS:
        X, Y = DATASETS[cfg.dataset](cfg.n_total, seed)
    elif cfg.dataset in REAL_DATASETS:
        kwargs = {}
        if cfg.dataset == "taxi" and cfg.taxi_csv:
            kwargs["csv_path"] = cfg.taxi_csv
        if cfg.dataset == "hurricane" and cfg.hurricane_csv:
            kwargs["csv_path"] = cfg.hurricane_csv
        result = REAL_DATASETS[cfg.dataset](**kwargs)
        X, Y, y_std_prod = result[0], result[1], result[2]
        vol_dims = result[3] if len(result) > 3 else None
        if len(result) > 5:
            y_orig_mean = result[4]
            y_orig_std = result[5]
        if len(result) > 7:
            x_orig_mean = result[6]
            x_orig_std = result[7]
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    # ── Normalize Y for synthetic datasets ──
    # Real datasets are already normalized in their loaders.
    # For synthetic datasets, normalize Y using global stats so that
    # models (especially NSF) train stably regardless of f_scale.
    if cfg.dataset in DATASETS:
        _y_mean = Y.mean(dim=0)
        _y_std = Y.std(dim=0).clamp(min=1e-8)
        Y = (Y - _y_mean) / _y_std
        y_orig_mean = _y_mean.numpy().astype(np.float64)
        y_orig_std = _y_std.numpy().astype(np.float64)
        y_std_prod = float(np.prod(y_orig_std))

    data = split_data(X, Y, seed)
    xd, yd = X.shape[1], Y.shape[1]
    y_train = data["try"]
    y_mean = y_train.mean(dim=0)
    y_std = y_train.std(dim=0)

    if verbose:
        y_mean_str = ", ".join(f"{v:.1f}" for v in y_mean.tolist())
        y_std_str = ", ".join(f"{v:.1f}" for v in y_std.tolist())
        vol_str = f"  vol_dims={vol_dims}" if vol_dims else ""
        print(f"\n  Data: x_dim={xd}  y_dim={yd}  "
              f"Train={len(data['trx'])}  Cal={len(data['cx'])}  "
              f"Test={len(data['tsx'])}{vol_str}")
        print(f"  Y stats: mean=[{y_mean_str}], std=[{y_std_str}]")

    # ── Resolve which methods & models to run ──
    active_methods = _resolve_methods(cfg)
    models_needed = _methods_to_models(active_methods)
    active_zspace = [m for m in active_methods if m in ZSPACE_NAMES]
    active_baselines = [m for m in active_methods if m in BASELINE_NAMES]

    if verbose:
        print(f"  Active methods: {active_methods}")
        print(f"  Models needed: {models_needed}")

    # ── Train NF ──
    nf_model = None
    train_time = {}  # model_name → training seconds
    if "NF" in models_needed:
        if verbose:
            print(f"\n[1/3] Training NF ...")
        nf_model = NFModel(xd, yd, cfg.cond_dim, cfg.hidden_dim,
                           cfg.nf_n_layers, cfg.nf_s_clamp,
                           flow_type=cfg.nf_flow_type,
                           n_bins=cfg.nf_n_bins, tail_bound=cfg.nf_tail_bound,
                           cond_net_type=cfg.nf_cond_net)
        _t = time.time()
        train_nf(nf_model, data["trx"], data["try"],
                 epochs=cfg.nf_epochs, batch_size=cfg.batch_size,
                 lr=cfg.nf_lr, patience=cfg.nf_patience,
                 device=device, verbose=verbose)
        train_time["NF"] = time.time() - _t
    elif verbose:
        print(f"\n[1/3] Skipping NF (not needed)")

    # ── Train Diffusion ──
    diff_model = None
    if "Diff" in models_needed:
        if verbose:
            print(f"\n[2/3] Training Diffusion ...")
        diff_model = ConditionalDDPM(
            yd, xd, T=cfg.diff_T, hidden_dim=cfg.hidden_dim,
            n_blocks=cfg.diff_n_blocks, schedule=cfg.diff_schedule,
            beta_min=cfg.diff_beta_min, beta_max=cfg.diff_beta_max,
            cfg_drop_prob=cfg.diff_cfg_drop_prob)
        diff_model.set_normalization(y_mean, y_std)
        _t = time.time()
        train_diffusion(diff_model, data["trx"], data["try"],
                        epochs=cfg.diff_epochs, batch_size=cfg.batch_size,
                        lr=cfg.diff_lr, weight_decay=cfg.weight_decay,
                        grad_clip=cfg.grad_clip, ema_decay=cfg.ema_decay,
                        patience=cfg.diff_patience,
                        device=device, verbose=verbose)
        train_time["Diff"] = time.time() - _t
    elif verbose:
        print(f"\n[2/3] Skipping Diffusion (not needed)")

    # ── Train FM ──
    fm_model = None
    if "FM" in models_needed:
        if verbose:
            print(f"\n[3/3] Training FM ...")
        fm_model = ConditionalFlowMatching(
            yd, xd, hidden_dim=cfg.hidden_dim, n_blocks=cfg.fm_n_blocks,
            sigma_min=cfg.fm_sigma_min, cfg_drop_prob=cfg.fm_cfg_drop_prob)
        fm_model.set_normalization(y_mean, y_std)
        _t = time.time()
        train_flow_matching(fm_model, data["trx"], data["try"],
                            epochs=cfg.fm_epochs, batch_size=cfg.batch_size,
                            lr=cfg.fm_lr, weight_decay=cfg.weight_decay,
                            grad_clip=cfg.grad_clip, ema_decay=cfg.ema_decay,
                            patience=cfg.fm_patience,
                            device=device, verbose=verbose)
        train_time["FM"] = time.time() - _t
    elif verbose:
        print(f"\n[3/3] Skipping FM (not needed)")

    # ── Save checkpoint (models + data split) ──
    _save_checkpoint(rep_dir, nf_model, diff_model, fm_model,
                     data, y_orig_mean, y_orig_std, y_std_prod, vol_dims,
                     x_orig_mean, x_orig_std)
    if verbose:
        print(f"  Checkpoint saved to {rep_dir}")

    # ── Z-space: score -> calibrate -> evaluate ──
    all_score_builders = {
        "NF-Ball":      lambda: NFBallScore(nf_model, device),
        "NF-NLL":       lambda: NFNLLScore(nf_model, device),
        "Diff-Denoise": lambda: DiffusionDenoiseScore(
            diff_model, device,
            n_timesteps=cfg.diff_score_timesteps,
            n_repeats=cfg.diff_score_repeats,
            seed=cfg.seed),
        "FM-Path":      lambda: FMPathScore(
            fm_model, device,
            n_timesteps=cfg.fm_score_timesteps,
            n_repeats=cfg.fm_score_repeats,
            seed=cfg.seed),
    }

    score_fns = {}
    for name in active_zspace:
        if name in all_score_builders:
            score_fns[name] = all_score_builders[name]()

    if score_fns and verbose:
        print(f"\n[C] Calibrating Z-space methods (alpha={cfg.alpha}) ...")
    predictors = {}
    score_time = {}  # method_name → score computation seconds (cal + test)
    for name, sfn in score_fns.items():
        cp = ConformalPredictor(sfn, alpha=cfg.alpha)
        _t = time.time()
        cp.calibrate(data["cx"], data["cy"])
        cp.evaluate(data["tsx"], data["tsy"])
        score_time[name] = time.time() - _t
        predictors[name] = cp
        if verbose:
            print(f"  {name:20s}  tau = {cp.tau:.4f}  score_time = {score_time[name]:.2f}s")

    all_results = {}
    for name, cp in predictors.items():
        all_results[name] = cp.evaluate(data["tsx"], data["tsy"])

    # ── Timing-only mode: save timings and stop ──
    if getattr(cfg, "timing_only", False):
        timing_results = {
            "_training_seconds": {k: round(v, 2) for k, v in train_time.items()},
            "_score_seconds": {k: round(v, 2) for k, v in score_time.items()},
        }
        # Include basic results (coverage, tau) but no volume
        for name, r in all_results.items():
            timing_results[name] = {k: (round(v, 6) if isinstance(v, float) else v)
                                     for k, v in r.items() if k != "scores"}
        path = os.path.join(rep_dir, "timing.json")
        with open(path, "w") as f:
            json.dump(timing_results, f, indent=2, default=str)

        if verbose:
            print(f"\n  === Timing Results ===")
            print(f"  Training:  {train_time}")
            print(f"  Score:     {score_time}")
            print(f"  Saved to {path}")

        # Cleanup
        if nf_model is not None:
            nf_model.cpu()
        if diff_model is not None:
            diff_model.cpu()
        if fm_model is not None:
            fm_model.cpu()
        clear_gpu()

        return (all_results, score_fns, predictors, {},
                data, nf_model, diff_model, fm_model,
                vol_dims, y_std_prod, {}, {**train_time, **score_time},
                y_orig_mean, y_orig_std, x_orig_mean, x_orig_std)

    # ── Z-space MC Volume ──
    # ── MC Volume (unified: all methods use mc_volume_grid in Y-space) ──
    if active_zspace and verbose:
        print(f"\n[V] Computing MC volumes (Y-space, unified) ...")
    n_vol = min(cfg.n_vol, len(data["tsx"]))
    x_vol = data["tsx"][:n_vol]
    time_dict = {}  # method_name → time in seconds

    _method_gen_model = {
        "NF-Ball": nf_model, "NF-NLL": nf_model,
        "Diff-Denoise": diff_model,
        "FM-Path": fm_model,
    }

    for sn in predictors:
        _t = time.time()
        vol, _ = mc_volume_grid(score_fns[sn], predictors[sn].tau,
                                x_vol, data["try"], device,
                                n_mc=cfg.n_mc, margin=cfg.vol_margin,
                                dataset_name=cfg.dataset,
                                gen_model=_method_gen_model.get(sn),
                                n_probe=cfg.vol_n_probe,
                                verbose_name=sn, seed=cfg.seed,
                                y_orig_mean=y_orig_mean, y_orig_std=y_orig_std)
        all_results[sn]["volume"] = vol
        time_dict[sn] = time.time() - _t

    # Release GPU after Z-space score + volume
    if nf_model is not None:
        nf_model.cpu()
    if diff_model is not None:
        diff_model.cpu()
    if fm_model is not None:
        fm_model.cpu()
    clear_gpu()

    # ── Y-space Baselines ──
    baseline_predictors = {}
    nf_samples_cache = {}

    if active_baselines:
        if verbose:
            print(f"\n[B] Running Y-space baselines: {active_baselines} ...")

        # Pre-compute NF samples if any baseline needs them
        # (all current baselines depend on NF samples)
        nf_den_cal = nf_den_test = nf_den_train = None
        yp_train = yp_cal = yp_test = None
        needs_nf_samples = bool(
            set(active_baselines) &
            {"RCP", "NLE", "DistSplit", "CQR", "MCQR", "PCP-NF"})
        if needs_nf_samples:
            if verbose:
                print(f"  Sampling from NF (n={cfg.n_samples_baseline}) ...")
            nf_den_cal = sample_ys_nf(nf_model, data["cx"],
                                       n_samples=cfg.n_samples_baseline,
                                       device=device)
            nf_den_test = sample_ys_nf(nf_model, data["tsx"],
                                        n_samples=cfg.n_samples_baseline,
                                        device=device)
            nf_den_train = sample_ys_nf(nf_model, data["trx"],
                                         n_samples=cfg.n_samples_baseline,
                                         device=device)
            yp_train = nf_den_train.mean(dim=1)
            yp_cal = nf_den_cal.mean(dim=1)
            yp_test = nf_den_test.mean(dim=1)
            nf_samples_cache["nf_den_test"] = nf_den_test
            nf_samples_cache["yp_test"] = yp_test

        # RCP
        if "RCP" in active_baselines:
            if verbose:
                print(f"  RCP ...", end="")
            _t = time.time()
            rcp = RCP(alpha=cfg.alpha, vol_dims=vol_dims)
            rcp.calibrate(data["cy"], yp_cal, data["try"], yp_train)
            all_results["RCP"] = rcp.evaluate(data["tsy"], yp_test)
            baseline_predictors["RCP"] = rcp
            time_dict["RCP"] = time.time() - _t
            if verbose:
                print(f"  cov={all_results['RCP']['coverage']:.3f}  "
                      f"vol={all_results['RCP']['volume']:.5f}")

        # NLE
        if "NLE" in active_baselines:
            if verbose:
                print(f"  NLE ...", end="")
            _t = time.time()
            nle = NLE(alpha=cfg.alpha, lam=cfg.nle_lambda,
                      k_frac=cfg.nle_k_frac, vol_dims=vol_dims)
            nle.calibrate(data["cx"], data["cy"], yp_cal,
                          data["trx"], data["try"], yp_train)
            all_results["NLE"] = nle.evaluate(data["tsx"], data["tsy"], yp_test)
            baseline_predictors["NLE"] = nle
            time_dict["NLE"] = time.time() - _t
            if verbose:
                print(f"  cov={all_results['NLE']['coverage']:.3f}  "
                      f"vol={all_results['NLE']['volume']:.5f}")

        # PCP variants
        pcp_names = [s.strip() for s in cfg.pcp_models.split(",")]
        for gname in pcp_names:
            key = f"PCP-{gname}"
            if key not in active_baselines:
                continue
            if verbose:
                print(f"  {key} ...", end="")
            _t = time.time()
            try:
                if gname == "NF":
                    pc = sample_ys_nf(nf_model, data["cx"],
                                      n_samples=cfg.pcp_n_samples, device=device)
                    pt = sample_ys_nf(nf_model, data["tsx"],
                                      n_samples=cfg.pcp_n_samples, device=device)
                elif gname == "Diff":
                    pc = sample_ys_diff(diff_model, data["cx"],
                                        n_samples=cfg.pcp_n_samples, device=device,
                                        n_steps=cfg.diff_sample_steps,
                                        cfg_scale=cfg.diff_cfg_scale,
                                        cfg_mode=cfg.diff_cfg_mode)
                    pt = sample_ys_diff(diff_model, data["tsx"],
                                        n_samples=cfg.pcp_n_samples, device=device,
                                        n_steps=cfg.diff_sample_steps,
                                        cfg_scale=cfg.diff_cfg_scale,
                                        cfg_mode=cfg.diff_cfg_mode)
                elif gname == "FM":
                    pc = sample_ys_fm(fm_model, data["cx"],
                                      n_samples=cfg.pcp_n_samples, device=device,
                                      n_steps=cfg.fm_sample_steps,
                                      cfg_scale=cfg.fm_cfg_scale,
                                      cfg_mode=cfg.fm_cfg_mode,
                                      solver=cfg.fm_solver)
                    pt = sample_ys_fm(fm_model, data["tsx"],
                                      n_samples=cfg.pcp_n_samples, device=device,
                                      n_steps=cfg.fm_sample_steps,
                                      cfg_scale=cfg.fm_cfg_scale,
                                      cfg_mode=cfg.fm_cfg_mode,
                                      solver=cfg.fm_solver)
                else:
                    if verbose:
                        print(f" [WARN] unknown model {gname}")
                    continue
            except Exception as e:
                if verbose:
                    print(f" [ERR] {e}")
                continue

            pcp = PCP(alpha=cfg.alpha, vol_dims=vol_dims, gen_name=gname)
            pcp.calibrate(data["cy"], pc)
            all_results[key] = pcp.evaluate(data["tsy"], pt)
            baseline_predictors[key] = (pcp, pt)
            time_dict[key] = time.time() - _t
            if verbose:
                print(f"  cov={all_results[key]['coverage']:.3f}  "
                      f"vol={all_results[key]['volume']:.5f}")

        # DistSplit
        if "DistSplit" in active_baselines:
            if verbose:
                print(f"  DistSplit ...", end="")
            _t = time.time()
            ds = DistSplit(alpha=cfg.alpha, vol_dims=vol_dims)
            ds.calibrate(data["cy"], nf_den_cal)
            all_results["DistSplit"] = ds.evaluate(data["tsy"], nf_den_test)
            baseline_predictors["DistSplit"] = ds
            time_dict["DistSplit"] = time.time() - _t
            if verbose:
                print(f"  cov={all_results['DistSplit']['coverage']:.3f}  "
                      f"vol={all_results['DistSplit']['volume']:.5f}")

        # CQR
        if "CQR" in active_baselines:
            if verbose:
                print(f"  CQR ...", end="")
            _t = time.time()
            cqr = CQR(alpha=cfg.alpha, vol_dims=vol_dims)
            cqr.calibrate(data["cy"], nf_den_cal)
            all_results["CQR"] = cqr.evaluate(data["tsy"], nf_den_test)
            baseline_predictors["CQR"] = cqr
            time_dict["CQR"] = time.time() - _t
            if verbose:
                print(f"  cov={all_results['CQR']['coverage']:.3f}  "
                      f"vol={all_results['CQR']['volume']:.5f}")

        # MCQR
        if "MCQR" in active_baselines:
            if verbose:
                print(f"  MCQR ...", end="")
            _t = time.time()
            mcqr = MCQR(alpha=cfg.alpha, device=device,
                         weight_epochs=cfg.mcqr_epochs, weight_lr=cfg.mcqr_lr,
                         vol_dims=vol_dims)
            mcqr.calibrate(data["cy"], nf_den_cal, data["trx"])
            all_results["MCQR"] = mcqr.evaluate(data["tsy"], nf_den_test)
            baseline_predictors["MCQR"] = mcqr
            time_dict["MCQR"] = time.time() - _t
            if verbose:
                print(f"  cov={all_results['MCQR']['coverage']:.3f}  "
                      f"vol={all_results['MCQR']['volume']:.5f}")

    # ── Scale volumes to original Y-space for real datasets ──
    if y_std_prod is not None:
        for name in all_results:
            if "volume" in all_results[name]:
                all_results[name]["volume"] *= y_std_prod

    # ── Save baseline objects for replotting ──
    if baseline_predictors:
        bl_path = os.path.join(rep_dir, "baselines.pt")
        torch.save(baseline_predictors, bl_path)
        if verbose:
            print(f"  Baselines saved to {bl_path}")

    # ── Summary table ──
    if verbose:
        print(f"\n  {'Method':20s}  {'Coverage':>8s}  {'tau':>10s}  {'Avg Vol':>10s}")
        print(f"  {'-'*52}")
        for name in all_results:
            r = all_results[name]
            print(f"  {name:20s}  {r['coverage']:8.3f}  {r['tau']:10.4f}  "
                  f"{r.get('volume', float('nan')):14.5f}")
        if train_time:
            print(f"\n  Training time:  {', '.join(f'{k}={v:.1f}s' for k, v in train_time.items())}")
        if score_time:
            print(f"  Score time:     {', '.join(f'{k}={v:.1f}s' for k, v in score_time.items())}")

    # ── Save per-repeat results ──
    _save_repeat_results(rep_dir, all_results, time_dict,
                         train_time=train_time, score_time=score_time)

    # ── Final GPU cleanup for this repeat ──
    if nf_model is not None:
        nf_model.cpu()
    if diff_model is not None:
        diff_model.cpu()
    if fm_model is not None:
        fm_model.cpu()
    clear_gpu()

    return (all_results, score_fns, predictors, baseline_predictors,
            data, nf_model, diff_model, fm_model,
            vol_dims, y_std_prod, nf_samples_cache, time_dict,
            y_orig_mean, y_orig_std, x_orig_mean, x_orig_std)


# ====================================================================
# Plotting helper: wrap baselines for region plot
# ====================================================================

def _rerun_methods_single(cfg, seed, device, outdir, verbose, rep_idx,
                          rerun_methods):
    """Rerun specific methods for one repeat: retrain needed models, re-score.

    1. Load checkpoint (data split + all models)
    2. Determine which models need retraining from rerun_methods
    3. Retrain those models (with current cfg params)
    4. Re-score + calibrate + volume for rerun_methods only
    5. Merge into existing results and save

    Args:
        rerun_methods: list of method names, e.g. ["FM-Path", "NF-Ball"]
    """
    rep_dir = _repeat_dir(outdir, rep_idx)
    if not os.path.exists(os.path.join(rep_dir, "data_split.pt")):
        raise FileNotFoundError(
            f"No checkpoint at {rep_dir}. Run full pipeline first.")

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ── Load checkpoint ──
    (nf_model, diff_model, fm_model, data,
     y_orig_mean, y_orig_std, y_std_prod, vol_dims,
     x_orig_mean, x_orig_std) = _load_checkpoint(
        rep_dir, cfg, device)

    if verbose:
        print(f"  Loaded checkpoint from {rep_dir}")

    # ── Determine which models to retrain ──
    models_needed = _methods_to_models(rerun_methods)
    if verbose:
        print(f"  Rerun methods: {rerun_methods}")
        print(f"  Models to retrain: {models_needed}")

    y_mean = data["try"].mean(0)
    y_std = data["try"].std(0)

    if "NF" in models_needed:
        if verbose:
            print(f"\n  [R] Retraining NF ...")
        xd = int(data["trx"].shape[1])
        yd = int(data["try"].shape[1])
        nf_model = NFModel(xd, yd, cfg.cond_dim, cfg.hidden_dim,
                           cfg.nf_n_layers, cfg.nf_s_clamp,
                           flow_type=cfg.nf_flow_type,
                           n_bins=cfg.nf_n_bins,
                           tail_bound=cfg.nf_tail_bound,
                           cond_net_type=cfg.nf_cond_net)
        train_nf(nf_model, data["trx"], data["try"],
                 epochs=cfg.nf_epochs, batch_size=cfg.batch_size,
                 lr=cfg.nf_lr, patience=cfg.nf_patience,
                 device=device, verbose=verbose)

    if "Diff" in models_needed:
        if verbose:
            print(f"\n  [R] Retraining Diffusion ...")
        xd = int(data["trx"].shape[1])
        yd = int(data["try"].shape[1])
        diff_model = ConditionalDDPM(
            yd, xd, T=cfg.diff_T, hidden_dim=cfg.hidden_dim,
            n_blocks=cfg.diff_n_blocks, schedule=cfg.diff_schedule,
            beta_min=cfg.diff_beta_min, beta_max=cfg.diff_beta_max,
            cfg_drop_prob=cfg.diff_cfg_drop_prob)
        diff_model.set_normalization(y_mean, y_std)
        train_diffusion(diff_model, data["trx"], data["try"],
                        epochs=cfg.diff_epochs, batch_size=cfg.batch_size,
                        lr=cfg.diff_lr, weight_decay=cfg.weight_decay,
                        grad_clip=cfg.grad_clip, ema_decay=cfg.ema_decay,
                        patience=cfg.diff_patience,
                        device=device, verbose=verbose)

    if "FM" in models_needed:
        if verbose:
            print(f"\n  [R] Retraining FM ...")
        xd = int(data["trx"].shape[1])
        yd = int(data["try"].shape[1])
        fm_model = ConditionalFlowMatching(
            yd, xd, hidden_dim=cfg.hidden_dim, n_blocks=cfg.fm_n_blocks,
            sigma_min=cfg.fm_sigma_min, cfg_drop_prob=cfg.fm_cfg_drop_prob)
        fm_model.set_normalization(y_mean, y_std)
        train_flow_matching(fm_model, data["trx"], data["try"],
                            epochs=cfg.fm_epochs, batch_size=cfg.batch_size,
                            lr=cfg.fm_lr, weight_decay=cfg.weight_decay,
                            grad_clip=cfg.grad_clip, ema_decay=cfg.ema_decay,
                            patience=cfg.fm_patience,
                            device=device, verbose=verbose)

    # ── Save retrained models back to checkpoint ──
    _save_checkpoint(rep_dir, nf_model, diff_model, fm_model,
                     data, y_orig_mean, y_orig_std, y_std_prod, vol_dims,
                     x_orig_mean, x_orig_std)

    # ── Build score fns for rerun methods only ──
    all_score_fns = {
        "NF-Ball": lambda: NFBallScore(nf_model, device),
        "NF-NLL": lambda: NFNLLScore(nf_model, device),
        "Diff-Denoise": lambda: DiffusionDenoiseScore(
            diff_model, device,
            n_timesteps=cfg.diff_score_timesteps,
            n_repeats=cfg.diff_score_repeats,
            seed=cfg.seed),
        "FM-Path": lambda: FMPathScore(
            fm_model, device,
            n_timesteps=cfg.fm_score_timesteps,
            n_repeats=cfg.fm_score_repeats,
            seed=cfg.seed),
    }

    # Z-space methods to rerun
    zspace_rerun = [m for m in rerun_methods if m in all_score_fns]
    baseline_rerun = [m for m in rerun_methods if m not in all_score_fns]

    score_fns = {}
    predictors = {}
    new_results = {}
    time_dict = {}

    # ── Z-space: score → calibrate → evaluate ──
    if zspace_rerun:
        if verbose:
            print(f"\n  [C] Calibrating rerun Z-space methods ...")
        for name in zspace_rerun:
            sfn = all_score_fns[name]()
            score_fns[name] = sfn
            cp = ConformalPredictor(sfn, alpha=cfg.alpha)
            cp.calibrate(data["cx"], data["cy"])
            predictors[name] = cp
            new_results[name] = cp.evaluate(data["tsx"], data["tsy"])
            if verbose:
                print(f"    {name:20s}  tau={cp.tau:.4f}  "
                      f"cov={new_results[name]['coverage']:.3f}")

    # ── Z-space volumes (unified: mc_volume_grid for all) ──
    n_vol = min(cfg.n_vol, len(data["tsx"]))
    x_vol = data["tsx"][:n_vol]

    _rerun_gen_model = {
        "NF-Ball": nf_model, "NF-NLL": nf_model,
        "Diff-Denoise": diff_model,
        "FM-Path": fm_model,
    }

    for sn in zspace_rerun:
        _t = time.time()
        vol, _ = mc_volume_grid(score_fns[sn], predictors[sn].tau,
                                x_vol, data["try"], device,
                                n_mc=cfg.n_mc, margin=cfg.vol_margin,
                                dataset_name=cfg.dataset,
                                gen_model=_rerun_gen_model.get(sn),
                                n_probe=cfg.vol_n_probe,
                                verbose_name=sn, seed=cfg.seed,
                                y_orig_mean=y_orig_mean, y_orig_std=y_orig_std)
        new_results[sn]["volume"] = vol
        time_dict[sn] = time.time() - _t

    # ── Baselines to rerun ──
    if baseline_rerun and cfg.baselines:
        if verbose:
            print(f"\n  [B] Rerunning baselines: {baseline_rerun} ...")

        # NF sampling (shared dependency for most baselines)
        need_nf_samples = any(
            m in ("RCP", "NLE", "DistSplit", "CQR", "MCQR") 
            for m in baseline_rerun)
        nf_den_cal = nf_den_test = nf_den_train = None
        yp_cal = yp_test = yp_train = None

        if need_nf_samples:
            nf_den_cal = sample_ys_nf(nf_model, data["cx"],
                                       n_samples=cfg.n_samples_baseline,
                                       device=device)
            nf_den_test = sample_ys_nf(nf_model, data["tsx"],
                                        n_samples=cfg.n_samples_baseline,
                                        device=device)
            nf_den_train = sample_ys_nf(nf_model, data["trx"],
                                         n_samples=cfg.n_samples_baseline,
                                         device=device)
            yp_train = nf_den_train.mean(dim=1)
            yp_cal = nf_den_cal.mean(dim=1)
            yp_test = nf_den_test.mean(dim=1)

        for bname in baseline_rerun:
            _t = time.time()
            try:
                if bname == "RCP":
                    bl = RCP(alpha=cfg.alpha, vol_dims=vol_dims)
                    bl.calibrate(data["cy"], yp_cal, data["try"], yp_train)
                    new_results[bname] = bl.evaluate(data["tsy"], yp_test)
                elif bname == "NLE":
                    bl = NLE(alpha=cfg.alpha, lam=cfg.nle_lambda,
                             k_frac=cfg.nle_k_frac, vol_dims=vol_dims)
                    bl.calibrate(data["cx"], data["cy"], yp_cal,
                                 data["trx"], data["try"], yp_train)
                    new_results[bname] = bl.evaluate(
                        data["tsx"], data["tsy"], yp_test)
                elif bname == "DistSplit":
                    bl = DistSplit(alpha=cfg.alpha, vol_dims=vol_dims)
                    bl.calibrate(data["cy"], nf_den_cal)
                    new_results[bname] = bl.evaluate(data["tsy"], nf_den_test)
                elif bname == "CQR":
                    bl = CQR(alpha=cfg.alpha, vol_dims=vol_dims)
                    bl.calibrate(data["cy"], nf_den_cal)
                    new_results[bname] = bl.evaluate(data["tsy"], nf_den_test)
                elif bname == "MCQR":
                    bl = MCQR(alpha=cfg.alpha, device=device,
                              weight_epochs=cfg.mcqr_epochs,
                              weight_lr=cfg.mcqr_lr, vol_dims=vol_dims)
                    bl.calibrate(data["cy"], nf_den_cal, data["trx"])
                    new_results[bname] = bl.evaluate(data["tsy"], nf_den_test)
                elif bname.startswith("PCP-"):
                    gname = bname.split("-", 1)[1]
                    if gname == "NF":
                        pc = sample_ys_nf(nf_model, data["cx"],
                                          n_samples=cfg.pcp_n_samples,
                                          device=device)
                        pt = sample_ys_nf(nf_model, data["tsx"],
                                          n_samples=cfg.pcp_n_samples,
                                          device=device)
                    elif gname == "Diff":
                        pc = sample_ys_diff(diff_model, data["cx"],
                                            n_samples=cfg.pcp_n_samples,
                                            device=device,
                                            n_steps=cfg.diff_sample_steps,
                                            cfg_scale=cfg.diff_cfg_scale,
                                            cfg_mode=cfg.diff_cfg_mode)
                        pt = sample_ys_diff(diff_model, data["tsx"],
                                            n_samples=cfg.pcp_n_samples,
                                            device=device,
                                            n_steps=cfg.diff_sample_steps,
                                            cfg_scale=cfg.diff_cfg_scale,
                                            cfg_mode=cfg.diff_cfg_mode)
                    elif gname == "FM":
                        pc = sample_ys_fm(fm_model, data["cx"],
                                          n_samples=cfg.pcp_n_samples,
                                          device=device,
                                          n_steps=cfg.fm_sample_steps,
                                          cfg_scale=cfg.fm_cfg_scale,
                                          cfg_mode=cfg.fm_cfg_mode,
                                          solver=cfg.fm_solver)
                        pt = sample_ys_fm(fm_model, data["tsx"],
                                          n_samples=cfg.pcp_n_samples,
                                          device=device,
                                          n_steps=cfg.fm_sample_steps,
                                          cfg_scale=cfg.fm_cfg_scale,
                                          cfg_mode=cfg.fm_cfg_mode,
                                          solver=cfg.fm_solver)
                    else:
                        continue
                    pcp = PCP(alpha=cfg.alpha, vol_dims=vol_dims,
                              gen_name=gname)
                    pcp.calibrate(data["cy"], pc)
                    new_results[bname] = pcp.evaluate(data["tsy"], pt)
                else:
                    if verbose:
                        print(f"    [WARN] Unknown baseline: {bname}")
                    continue
            except Exception as e:
                if verbose:
                    print(f"    [ERR] {bname}: {e}")
                continue
            time_dict[bname] = time.time() - _t
            if verbose:
                r = new_results[bname]
                print(f"    {bname:20s}  cov={r['coverage']:.3f}  "
                      f"vol={r.get('volume', float('nan')):.5f}")

    # ── Scale volumes to original Y-space ──
    if y_std_prod is not None:
        for name in new_results:
            if "volume" in new_results[name]:
                new_results[name]["volume"] *= y_std_prod

    # ── Merge with existing results and save ──
    old_results, old_time = _load_repeat_results(rep_dir)
    if old_results is None:
        old_results = {}
    if old_time is None:
        old_time = {}
    old_results.update(new_results)
    old_time.update(time_dict)
    _save_repeat_results(rep_dir, old_results, old_time)

    if verbose:
        print(f"\n  Rerun results merged and saved to {rep_dir}")
        for name in new_results:
            r = new_results[name]
            print(f"    {name:20s}  cov={r['coverage']:.3f}  "
                  f"vol={r.get('volume', float('nan')):.5f}")

    # ── GPU cleanup ──
    nf_model.cpu(); diff_model.cpu(); fm_model.cpu()
    clear_gpu()

    return new_results, old_results


class _GridWrap:
    """Lightweight wrapper so baselines work with plot_prediction_regions."""
    def __init__(self, name, tau, grid_fn):
        self.name = name
        self.tau = tau
        self._gfn = grid_fn

    def predict_grid(self, x_pt, y_grid, **kw):
        sc = self._gfn(x_pt, y_grid)
        return sc <= self.tau, sc


def _build_pred_list(cfg, predictors, baseline_predictors, all_results,
                     data, nf_model, diff_model, fm_model, device,
                     x_index=None):
    """Build list of (predictor, results) for region plot."""
    n_test = len(data["tsx"])
    if x_index is None:
        x_index = n_test // 2
    x_point = data["tsx"][x_index]

    # Z-space methods
    pred_list = [(predictors[n], all_results[n]) for n in ZSPACE_NAMES
                 if n in predictors]

    if not cfg.baselines:
        return pred_list, x_point

    # NF samples for the plot x-point
    nf_samp_pt = sample_ys_nf(nf_model, x_point.unsqueeze(0),
                               n_samples=cfg.n_samples_baseline,
                               device=device)[0]   # [S, yd]
    yp_pt = nf_samp_pt.mean(dim=0)                  # [yd]

    # RCP, NLE
    for bname in ["RCP", "NLE"]:
        if bname not in baseline_predictors:
            continue
        bobj = baseline_predictors[bname]
        res = all_results[bname]
        def _make_gfn_pred(b=bobj, yp=yp_pt):
            return lambda xp, yg: b.compute_on_grid(xp, yg, yp)
        pred_list.append((_GridWrap(bname, res["tau"], _make_gfn_pred()), res))

    # DistSplit, CQR, MCQR — binary score (0 inside, 2 outside), tau=0.5
    for bname in ["DistSplit", "CQR", "MCQR"]:
        if bname not in baseline_predictors:
            continue
        bobj = baseline_predictors[bname]
        res = all_results[bname]
        def _make_gfn_dens(b=bobj, d=nf_samp_pt):
            return lambda xp, yg: b.compute_on_grid(xp, yg, d)
        pred_list.append((_GridWrap(bname, 0.5, _make_gfn_dens()), res))

    # PCP variants
    for key in sorted(baseline_predictors.keys()):
        if not key.startswith("PCP"):
            continue
        pcp_obj, _ = baseline_predictors[key]
        res = all_results[key]
        gname = key.split("-")[1]
        try:
            if gname == "NF":
                pdens = sample_ys_nf(nf_model, x_point.unsqueeze(0),
                                      n_samples=cfg.pcp_n_samples,
                                      device=device)[0]
            elif gname == "Diff":
                pdens = sample_ys_diff(diff_model, x_point.unsqueeze(0),
                                        n_samples=cfg.pcp_n_samples,
                                        device=device,
                                        n_steps=cfg.diff_sample_steps)[0]
            elif gname == "FM":
                pdens = sample_ys_fm(fm_model, x_point.unsqueeze(0),
                                      n_samples=cfg.pcp_n_samples,
                                      device=device,
                                      n_steps=cfg.fm_sample_steps,
                                      solver=cfg.fm_solver)[0]
            else:
                continue
        except Exception:
            continue
        def _make_gfn_pcp(p=pcp_obj, d=pdens):
            return lambda xp, yg: p.compute_on_grid(xp, yg, d)
        pred_list.append((_GridWrap(key, res["tau"], _make_gfn_pcp()), res))

    return pred_list, x_point


# ====================================================================
# Main entry
# ====================================================================

def run_experiment(cfg=None, **kwargs):
    """Run experiment with optional repeats."""
    if cfg is None:
        cfg = Config(**kwargs)

    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    verbose = cfg.verbose
    train_only = getattr(cfg, "train_only", False)

    outdir = os.path.join(cfg.outdir, f"{cfg.dataset}_s{cfg.seed}")
    if cfg.n_repeats == 1:
        outdir = outdir + "_single"
    os.makedirs(outdir, exist_ok=True)

    # Save config for reproducibility / replotting
    cfg_path = os.path.join(outdir, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg.to_dict(), f, indent=2, default=str)

    # Force restart: remove existing repeat directories
    if cfg.force_restart:
        import shutil
        for item in os.listdir(outdir):
            item_path = os.path.join(outdir, item)
            if item.startswith("repeat_") and os.path.isdir(item_path):
                shutil.rmtree(item_path)
        # Also remove old results.json
        old_results_path = os.path.join(outdir, "results.json")
        if os.path.exists(old_results_path):
            os.remove(old_results_path)

    if verbose:
        active_methods = _resolve_methods(cfg)
        models_needed = _methods_to_models(active_methods)
        print(f"{'='*65}")
        print(f"  Dataset: {cfg.dataset}  |  Seed: {cfg.seed}  |  Device: {device}")
        mode = "TRAIN ONLY" if train_only else (
            "TIMING ONLY" if getattr(cfg, "timing_only", False)
            else f"FULL (n_repeats={cfg.n_repeats})")
        if cfg.methods is not None:
            methods_str = f"custom ({len(active_methods)})"
        else:
            methods_str = f"all ({len(active_methods)})"
        print(f"  Mode: {mode}  |  Methods: {methods_str}")
        print(f"  Active: {active_methods}")
        print(f"  Models to train: {sorted(models_needed)}")
        if "NF" in models_needed:
            print(f"  NF: epochs={cfg.nf_epochs}  layers={cfg.nf_n_layers}")
        if "Diff" in models_needed:
            print(f"  Diff: epochs={cfg.diff_epochs}  T={cfg.diff_T}  "
                  f"blocks={cfg.diff_n_blocks}")
        if "FM" in models_needed:
            print(f"  FM: epochs={cfg.fm_epochs}  blocks={cfg.fm_n_blocks}  "
                  f"solver={cfg.fm_solver}")
        print(f"  hidden={cfg.hidden_dim}  batch={cfg.batch_size}")
        print(f"  Patience: NF={cfg.nf_patience}  Diff={cfg.diff_patience}  FM={cfg.fm_patience}")
        baseline_methods = [m for m in active_methods if m in BASELINE_NAMES]
        if baseline_methods:
            print(f"  Baselines: {baseline_methods}")
            print(f"  NF samples: {cfg.n_samples_baseline}")
        print(f"{'='*65}")

    t0 = time.time()
    # Variables that may or may not be set depending on mode
    data = None
    nf_model = diff_model = fm_model = None
    y_std_prod = vol_dims = None
    y_orig_mean = y_orig_std = None
    x_orig_mean = x_orig_std = None
    time_dict = {}
    nf_samples_cache = {}
    predictors = {}
    baseline_predictors = {}
    all_results = {}
    score_fns = {}

    # ── Train-only mode ──
    if train_only:
        res = _single_run(cfg, cfg.seed, device, outdir, verbose, rep_idx=0)
        (_, _, _, _, data, nf_model, diff_model, fm_model,
         _, _, _, _, _, _) = res
        if cfg.dataset in DATASETS:
            if verbose:
                print(f"\n[*] Sample quality diagnostic ...")
            n_test = len(data["tsx"])
            xi_mid = n_test // 2
            models_dict = {}
            if nf_model is not None:
                models_dict["NF"] = nf_model
            if diff_model is not None:
                models_dict["Diffusion"] = diff_model
            if fm_model is not None:
                models_dict["FM"] = fm_model
            if models_dict:
                plot_sample_quality(
                    models_dict,
                    data["tsx"][xi_mid], cfg.dataset, device=device, n_samples=cfg.sample_quality_n,
                    save_path=os.path.join(outdir,
                        f"{cfg.dataset}_sample_quality.png"),
                    diff_cfg_scale=cfg.diff_cfg_scale,
                    diff_cfg_mode=cfg.diff_cfg_mode,
                    diff_sample_steps=cfg.diff_sample_steps,
                    fm_cfg_scale=cfg.fm_cfg_scale,
                    fm_cfg_mode=cfg.fm_cfg_mode,
                    fm_solver=cfg.fm_solver,
                    fm_sample_steps=cfg.fm_sample_steps,
                    y_orig_mean=y_orig_mean, y_orig_std=y_orig_std)
        print(f"\n  Train-only done in {time.time()-t0:.1f}s")
        return None

    # ── Timing-only mode ──
    if getattr(cfg, "timing_only", False):
        res = _single_run(cfg, cfg.seed, device, outdir, verbose, rep_idx=0)
        print(f"\n  Timing-only done in {time.time()-t0:.1f}s")
        return None

    # ── Full pipeline with repeats ──
    rerun_methods = getattr(cfg, "rerun", None)  # list or None
    all_repeats = {}

    for rep in range(cfg.n_repeats):
        seed = cfg.seed + rep

        if rerun_methods:
            # ── RERUN MODE: retrain specified methods ──
            if verbose:
                print(f"\n{'='*65}")
                print(f"  RERUN {rep+1}/{cfg.n_repeats}  (seed={seed})")
                print(f"  Methods: {rerun_methods}")
                print(f"{'='*65}")
            try:
                new_results, merged_results = _rerun_methods_single(
                    cfg, seed, device, outdir, verbose, rep, rerun_methods)
                all_results = merged_results
            except FileNotFoundError as e:
                print(f"  [SKIP] repeat {rep}: {e}")
                continue
        elif cfg.n_repeats > 1 and _repeat_is_complete(outdir, rep):
            # ── RESUME: skip already-done repeats (only for multi-repeat) ──
            if verbose:
                print(f"\n  Repeat {rep+1}/{cfg.n_repeats} (seed={seed}) "
                      f"already complete, loading results ...")
            rep_dir = _repeat_dir(outdir, rep)
            all_results, _ = _load_repeat_results(rep_dir)
            if all_results is None:
                all_results = {}
        else:
            # ── NORMAL: full run ──
            if verbose and cfg.n_repeats > 1:
                print(f"\n{'='*65}")
                print(f"  REPEAT {rep+1}/{cfg.n_repeats}  (seed={seed})")
                print(f"{'='*65}")

            (all_results, score_fns, predictors, baseline_predictors,
             data, nf_model, diff_model, fm_model,
             vol_dims, y_std_prod, nf_samples_cache, time_dict,
             y_orig_mean, y_orig_std, x_orig_mean, x_orig_std) = _single_run(
                cfg, seed, device, outdir, verbose, rep_idx=rep)

        # Accumulate for multi-repeat summary
        for name, r in all_results.items():
            if name not in all_repeats:
                all_repeats[name] = {"coverage": [], "volume": []}
            cov_val = r["coverage"] if isinstance(r["coverage"], float) else float(r["coverage"])
            vol_val = r.get("volume", float("nan"))
            if not isinstance(vol_val, float):
                vol_val = float(vol_val)
            all_repeats[name]["coverage"].append(cov_val)
            all_repeats[name]["volume"].append(vol_val)

    # ── Summary (multi-repeat) ──
    if verbose and cfg.n_repeats > 1:
        print(f"\n{'='*65}")
        print(f"  SUMMARY ({cfg.n_repeats} repeats)")
        print(f"  {'Method':20s}  {'Coverage':>16s}  {'Volume':>16s}")
        print(f"  {'-'*56}")
        for name in all_repeats:
            covs = all_repeats[name]["coverage"]
            vols = all_repeats[name]["volume"]
            print(f"  {name:20s}  {np.mean(covs):.3f} +/- {np.std(covs):.3f}  "
                  f"{np.nanmean(vols):.5f} +/- {np.nanstd(vols):.5f}")
        print(f"{'='*65}")

    # ── Plots ──
    do_plot = (cfg.n_repeats == 1 and cfg.dataset in PLOT_DATASETS
               and not rerun_methods and data is not None)

    # Multi-repeat: violin plot
    if cfg.n_repeats > 1 and all_repeats:
        plot_repeats_violin(
            all_repeats, alpha=cfg.alpha,
            title=f"{cfg.dataset} — {cfg.n_repeats} Repeats",
            save_path=os.path.join(outdir, f"{cfg.dataset}_repeats_violin.png"))

    # Table + Pareto: generate for single runs (works with resumed results too)
    if cfg.n_repeats == 1 and all_results:
        plot_results_table(
            all_results, alpha=cfg.alpha,
            title=f"{cfg.dataset} — Results (α={cfg.alpha})",
            save_path=os.path.join(outdir, f"{cfg.dataset}_table.png"),
            time_dict=time_dict)

        plot_pareto(
            all_results, alpha=cfg.alpha,
            title=f"{cfg.dataset} — Coverage vs Volume Pareto",
            save_path=os.path.join(outdir, f"{cfg.dataset}_pareto.png"))

    if do_plot:
        if verbose:
            print(f"\n[P] Generating plots ...")
        n_test = len(data["tsx"])

        # Sample quality (synthetic only — needs true p(Y|X) for comparison)
        is_synthetic = cfg.dataset in DATASETS
        if is_synthetic:
            xi_mid = n_test // 2
            models_dict = {}
            if nf_model is not None:
                models_dict["NF"] = nf_model
            if diff_model is not None:
                models_dict["Diffusion"] = diff_model
            if fm_model is not None:
                models_dict["FM"] = fm_model
            if models_dict:
                plot_sample_quality(
                    models_dict,
                    data["tsx"][xi_mid], cfg.dataset, device=device, n_samples=cfg.sample_quality_n,
                    save_path=os.path.join(outdir,
                        f"{cfg.dataset}_sample_quality.png"),
                    diff_cfg_scale=cfg.diff_cfg_scale,
                    diff_cfg_mode=cfg.diff_cfg_mode,
                    diff_sample_steps=cfg.diff_sample_steps,
                    fm_cfg_scale=cfg.fm_cfg_scale,
                    fm_cfg_mode=cfg.fm_cfg_mode,
                    fm_solver=cfg.fm_solver,
                    fm_sample_steps=cfg.fm_sample_steps,
                    y_orig_mean=y_orig_mean, y_orig_std=y_orig_std)

        # Prediction regions (3 diverse x points)
        for xi in [n_test // 4, n_test // 2, 3 * n_test // 4]:
            pred_list, x_point = _build_pred_list(
                cfg, predictors, baseline_predictors, all_results,
                data, nf_model, diff_model, fm_model, device,
                x_index=xi)

            plot_prediction_regions_with_eval(
                pred_list, x_point, data,
                grid_res=cfg.grid_res, grid_n_avg=cfg.grid_n_avg,
                smooth_sigma=cfg.smooth_sigma,
                title=f"{cfg.dataset} -- Regions (x idx={xi}, α={cfg.alpha})",
                save_path=os.path.join(outdir,
                    f"{cfg.dataset}_regions_x{xi}.png"),
                dataset_name=cfg.dataset,
                nf_model=nf_model, device=device,
                y_true=data["tsy"][xi],
                y_orig_mean=y_orig_mean, y_orig_std=y_orig_std)

        # Taxi map (static PNG with map tiles + prediction regions)
        if cfg.dataset == "taxi" and y_orig_mean is not None:
            n_test = len(data["tsx"])
            pred_list, map_x = _build_pred_list(
                cfg, predictors, baseline_predictors, all_results,
                data, nf_model, diff_model, fm_model, device,
                x_index=n_test // 2)
            plot_taxi_map(
                pred_list, map_x,
                y_orig_mean, y_orig_std,
                all_results,
                x_orig_mean=x_orig_mean, x_orig_std=x_orig_std,
                grid_res=cfg.grid_res, grid_n_avg=cfg.grid_n_avg,
                smooth_sigma=cfg.smooth_sigma,
                save_dir=outdir, prefix=cfg.dataset)

        # Hurricane map (interactive HTML with prediction regions on world map)
        if cfg.dataset == "hurricane" and y_orig_mean is not None:
            # Build wrapped predictors (Z-space + baselines) using _build_pred_list
            n_test = len(data["tsx"])
            pred_list, map_x = _build_pred_list(
                cfg, predictors, baseline_predictors, all_results,
                data, nf_model, diff_model, fm_model, device,
                x_index=n_test // 2)
            # Convert pred_list to dict {name: predictor}
            all_preds = {}
            for pred_obj, res in pred_list:
                all_preds[pred_obj.name] = pred_obj
            plot_hurricane_map(
                all_preds, map_x,
                y_orig_mean, y_orig_std,
                x_orig_mean, x_orig_std,
                all_results,
                grid_res=cfg.grid_res, grid_n_avg=cfg.grid_n_avg,
                smooth_sigma=cfg.smooth_sigma,
                save_dir=outdir, prefix=cfg.dataset)

        # Comparison bars (all methods)
        plot_comparison_bars(
            all_results,
            title=f"{cfg.dataset} -- Method Comparison",
            save_path=os.path.join(outdir, f"{cfg.dataset}_comparison.png"))

        # Score distributions (Z-space only)
        zspace_res = {n: all_results[n] for n in ZSPACE_NAMES
                      if n in all_results}
        if zspace_res:
            plot_score_distributions(
                zspace_res, alpha=cfg.alpha,
                title=f"{cfg.dataset} -- Score Distributions",
                save_path=os.path.join(outdir, f"{cfg.dataset}_scores.png"))

    # ── Save results ──
    elapsed = time.time() - t0

    save_obj = {
        "config": cfg.to_dict(),
        "elapsed_sec": round(elapsed, 1),
        "n_repeats": cfg.n_repeats,
        "dataset": cfg.dataset,
        "methods": list(all_results.keys()) if all_results else [],
        "last_run": {},
    }

    # Add data_info and model_info if available (normal/train mode)
    if data is not None:
        save_obj["data_info"] = {
            "x_dim": int(data["trx"].shape[1]),
            "y_dim": int(data["try"].shape[1]),
            "n_train": int(data["trx"].shape[0]),
            "n_cal": int(data["cx"].shape[0]),
            "n_test": int(data["tsx"].shape[0]),
            "y_mean": [round(float(v), 4) for v in data["try"].mean(0).tolist()],
            "y_std": [round(float(v), 4) for v in data["try"].std(0).tolist()],
        }
        def _count_params(m):
            return sum(p.numel() for p in m.parameters()) if m is not None else 0
        model_info = {}
        if nf_model is not None:
            model_info["nf_params"] = _count_params(nf_model)
        if diff_model is not None:
            model_info["diff_params"] = _count_params(diff_model)
        if fm_model is not None:
            model_info["fm_params"] = _count_params(fm_model)
        if model_info:
            save_obj["model_info"] = model_info

    if all_results:
        for name, r in all_results.items():
            save_obj["last_run"][name] = {
                k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in r.items() if k != "scores"
            }
    if y_std_prod is not None:
        save_obj["y_std_prod"] = y_std_prod
    if time_dict:
        save_obj["time_seconds"] = {k: round(v, 2) for k, v in time_dict.items()}
    if vol_dims is not None:
        save_obj["vol_dims"] = vol_dims
    if rerun_methods:
        save_obj["rerun_methods"] = rerun_methods
    if cfg.n_repeats > 1 and all_repeats:
        save_obj["repeats"] = {
            name: {
                "coverage": [round(float(v), 6) for v in d["coverage"]],
                "volume": [round(float(v), 6) for v in d["volume"]],
                "coverage_mean": round(float(np.mean(d["coverage"])), 4),
                "coverage_std": round(float(np.std(d["coverage"])), 4),
                "volume_mean": round(float(np.nanmean(d["volume"])), 4),
                "volume_std": round(float(np.nanstd(d["volume"])), 4),
            }
            for name, d in all_repeats.items()
        }

    results_path = os.path.join(outdir, "results.json")
    with open(results_path, "w") as f:
        json.dump(save_obj, f, indent=2, default=str)

    if verbose:
        print(f"\n{'='*65}")
        print(f"  Done in {elapsed:.1f}s  ->  {results_path}")
        print(f"{'='*65}")

    return all_results


# ====================================================================
# CLI
# ====================================================================

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="GC experiment runner")

    p.add_argument("--dataset", type=str, default="spiral")
    p.add_argument("--taxi_csv", type=str, default=None)
    p.add_argument("--hurricane_csv", type=str, default=None)
    p.add_argument("--n_total", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train_only", action="store_true")
    p.add_argument("--timing_only", action="store_true",
                   help="Train + score timing only, skip volume/baselines/plots")
    p.add_argument("--force_restart", action="store_true",
                   help="Delete existing checkpoints and start fresh")
    p.add_argument("--n_repeats", type=int, default=None)
    p.add_argument("--rerun", type=str, default=None,
                   help='Comma-separated methods to rerun, e.g. "FM-Path,NF-Ball"')
    p.add_argument("--no_baselines", action="store_true")
    p.add_argument("--methods", type=str, default=None,
                   help='Comma-separated methods to run, e.g. "NF-Ball,FM-Path,RCP". '
                        'Default: all methods. Available Z-space: NF-Ball, NF-NLL, '
                        'Diff-Denoise, FM-Path. '
                        'Available baselines: RCP, NLE, PCP-NF, PCP-Diff, PCP-FM, '
                        'DistSplit, CQR, MCQR')
    p.add_argument("--quiet", action="store_true")

    # Architecture
    p.add_argument("--hidden_dim", type=int, default=None)
    p.add_argument("--cond_dim", type=int, default=None)
    p.add_argument("--t_embed_dim", type=int, default=None)

    # NF
    p.add_argument("--nf_epochs", type=int, default=None)
    p.add_argument("--nf_n_layers", type=int, default=None)
    p.add_argument("--nf_lr", type=float, default=None)
    p.add_argument("--nf_s_clamp", type=float, default=None)
    p.add_argument("--nf_flow_type", type=str, default=None,
                   choices=["realnvp", "nsf"])
    p.add_argument("--nf_cond_net", type=str, default=None,
                   choices=["mlp", "resnet"])
    p.add_argument("--nf_n_bins", type=int, default=None)
    p.add_argument("--nf_tail_bound", type=float, default=None)

    # Diffusion
    p.add_argument("--diff_epochs", type=int, default=None)
    p.add_argument("--diff_T", type=int, default=None)
    p.add_argument("--diff_n_blocks", type=int, default=None)
    p.add_argument("--diff_lr", type=float, default=None)
    p.add_argument("--diff_schedule", type=str, default=None)
    p.add_argument("--diff_beta_min", type=float, default=None)
    p.add_argument("--diff_beta_max", type=float, default=None)
    p.add_argument("--diff_cfg_scale", type=float, default=None)
    p.add_argument("--diff_cfg_mode", type=str, default=None)
    p.add_argument("--diff_cfg_drop_prob", type=float, default=None)
    p.add_argument("--diff_sample_steps", type=int, default=None)
    p.add_argument("--diff_score_timesteps", type=int, default=None)
    p.add_argument("--diff_score_repeats", type=int, default=None)

    # FM
    p.add_argument("--fm_epochs", type=int, default=None)
    p.add_argument("--fm_n_blocks", type=int, default=None)
    p.add_argument("--fm_lr", type=float, default=None)
    p.add_argument("--fm_sigma_min", type=float, default=None)
    p.add_argument("--fm_cfg_scale", type=float, default=None)
    p.add_argument("--fm_cfg_mode", type=str, default=None)
    p.add_argument("--fm_cfg_drop_prob", type=float, default=None)
    p.add_argument("--fm_solver", type=str, default=None)
    p.add_argument("--fm_sample_steps", type=int, default=None)
    p.add_argument("--fm_score_timesteps", type=int, default=None)
    p.add_argument("--fm_score_repeats", type=int, default=None)

    # Mode Attraction
    p.add_argument("--ma_n_steps", type=int, default=None)
    p.add_argument("--ma_lr", type=float, default=None)
    p.add_argument("--ma_nf_lr", type=float, default=None)
    p.add_argument("--ma_grad_clip", type=float, default=None)
    p.add_argument("--ma_diff_t_star", type=float, default=None)
    p.add_argument("--ma_fm_t_star", type=float, default=None)

    # Training
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--grad_clip", type=float, default=None)
    p.add_argument("--ema_decay", type=float, default=None)
    p.add_argument("--nf_patience", type=int, default=None)
    p.add_argument("--diff_patience", type=int, default=None)
    p.add_argument("--fm_patience", type=int, default=None)

    # Conformal
    p.add_argument("--alpha", type=float, default=None)

    # MC volume
    p.add_argument("--n_mc", type=int, default=None)
    p.add_argument("--n_vol", type=int, default=None)
    p.add_argument("--vol_margin", type=float, default=None)
    p.add_argument("--vol_R_mult", type=float, default=None)
    p.add_argument("--vol_n_probe", type=int, default=None)

    # Baselines
    p.add_argument("--n_samples_baseline", type=int, default=None)
    p.add_argument("--pcp_models", type=str, default=None)
    p.add_argument("--pcp_n_samples", type=int, default=None)
    p.add_argument("--nle_lambda", type=float, default=None)
    p.add_argument("--nle_k_frac", type=float, default=None)
    p.add_argument("--mcqr_epochs", type=int, default=None)
    p.add_argument("--mcqr_lr", type=float, default=None)

    # Plotting
    p.add_argument("--grid_res", type=int, default=None)
    p.add_argument("--grid_n_avg", type=int, default=None)
    p.add_argument("--smooth_sigma", type=float, default=None)
    p.add_argument("--n_scatter", type=int, default=None)
    p.add_argument("--sample_quality_n", type=int, default=None)

    # Output
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--device", type=str, default=None)

    args = p.parse_args()

    # Build config: start from defaults, override with CLI args
    overrides = {}
    for k, v in vars(args).items():
        if k == "no_baselines":
            if v:
                overrides["baselines"] = False
        elif k == "quiet":
            if v:
                overrides["verbose"] = False
        elif k == "train_only":
            overrides["train_only"] = v
        elif k == "timing_only":
            overrides["timing_only"] = v
        elif k == "force_restart":
            if v:
                overrides["force_restart"] = True
        elif k == "rerun":
            if v is not None:
                overrides["rerun"] = [s.strip() for s in v.split(",")]
        elif k == "methods":
            if v is not None:
                overrides["methods"] = [s.strip() for s in v.split(",")]
        elif v is not None:
            overrides[k] = v

    cfg = Config(**overrides)
    run_experiment(cfg)