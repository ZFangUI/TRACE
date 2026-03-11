"""
Central configuration for conformal_generative experiments.

6 methods: NF-Ball, NF-NLL, Diff-Denoise, FM-Path, Diff-ODE-Ball, FM-ODE-Ball
"""

from dataclasses import dataclass, asdict, field
from typing import Optional


@dataclass
class Config:
    # ── Data ──
    dataset: str = "spiral"
    taxi_csv: str = None                   # path to NYC taxi CSV
    hurricane_csv: str = None              # path to IBTrACS CSV
    n_total: int = 10000
    seed: int = 0

    # ── Shared architecture ──
    hidden_dim: int = 256
    cond_dim: int = 128
    t_embed_dim: int = 64                  # sinusoidal time embedding dim

    # ── NF (RealNVP / NSF) ──
    nf_flow_type: str = "realnvp"          # "realnvp" or "nsf"
    nf_cond_net: str = "mlp"               # "mlp" or "resnet"
    nf_n_layers: int = 8
    nf_s_clamp: float = 3.0               # RealNVP: scale clamp
    nf_n_bins: int = 8                     # NSF: number of spline bins
    nf_tail_bound: float = 3.0            # NSF: spline domain [-B, B]
    nf_epochs: int = 500
    nf_lr: float = 1e-3

    # ── Diffusion (DDPM) ──
    diff_n_blocks: int = 8
    diff_T: int = 200
    diff_schedule: str = "cosine"
    diff_beta_min: float = 1e-4
    diff_beta_max: float = 0.02
    diff_cfg_drop_prob: float = 0.15
    diff_epochs: int = 2000
    diff_lr: float = 3e-4
    diff_cfg_scale: float = 1.0
    diff_cfg_mode: str = "none"
    diff_sample_steps: int = 100
    diff_score_timesteps: int = 15
    diff_score_repeats: int = 8
    diff_ode_steps: int = 50               # DDIM encode steps for ODE-Ball

    # ── Flow Matching (OT-CFM) ──
    fm_n_blocks: int = 8
    fm_sigma_min: float = 1e-4
    fm_cfg_drop_prob: float = 0.15
    fm_epochs: int = 2000
    fm_lr: float = 3e-4
    fm_cfg_scale: float = 1.0
    fm_cfg_mode: str = "none"
    fm_solver: str = "midpoint"
    fm_sample_steps: int = 100
    fm_score_timesteps: int = 15
    fm_score_repeats: int = 8
    fm_ode_steps: int = 50                 # ODE encode steps for ODE-Ball

    # ── Training (shared) ──
    batch_size: int = 512
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    nf_patience: int = 0                   # NF early stopping (0=off, NF is fast enough)
    diff_patience: int = 150               # Diff early stopping patience
    fm_patience: int = 150                 # FM early stopping patience

    # ── Mode Attraction ──
    ma_n_steps: int = 50                   # gradient ascent steps
    ma_lr: float = 0.01                    # step size (Diff/FM, normalized direction)
    ma_nf_lr: float = 0.05                 # step size (NF, raw gradient, can be larger)
    ma_grad_clip: float = 5.0              # NF gradient norm clipping
    ma_diff_t_star: float = 0.05           # Diff noise level fraction of T
    ma_fm_t_star: float = 0.95             # FM time level

    # ── Conformal ──
    alpha: float = 0.1

    # ── Baselines ──
    n_samples_baseline: int = 1000       # NF samples for RCP/NLE/DistSplit/CQR/MCQR
    pcp_models: str = "Diff"             # PCP variants: "NF", "Diff", "FM", or "NF,Diff,FM"
    pcp_n_samples: int = 500             # PCP samples per model
    nle_lambda: float = 0.9
    nle_k_frac: float = 0.05
    mcqr_epochs: int = 300
    mcqr_lr: float = 1e-3
    baselines: bool = True               # run Y-space baselines

    # ── Method selection ──
    # None = run all methods.  List of method names to run a subset, e.g.:
    #   ["NF-Ball", "FM-Path"]            → only these two Z-space methods
    #   ["NF-Ball", "RCP", "PCP-Diff"]    → mix of Z-space and baselines
    # Available Z-space:  NF-Ball, NF-NLL, Diff-Denoise, FM-Path,
    #                     Diff-ODE-Ball, FM-ODE-Ball
    # Available baselines: RCP, NLE, PCP-NF, PCP-Diff, PCP-FM,
    #                      DistSplit, CQR, MCQR
    # Models are auto-trained only when needed by the selected methods.
    methods: Optional[list] = None

    # ── Repeats (run full pipeline n_repeats times, report mean±std) ──
    n_repeats: int = 1

    # ── MC volume ──
    n_mc: int = 5000
    n_vol: int = 30
    vol_margin: float = 0.3
    vol_R_mult: float = 2.5             # NF z-space bounding ball radius multiplier
    vol_n_probe: int = 500              # samples for local bounding box estimation

    # ── Plotting ──
    grid_res: int = 200
    grid_n_avg: int = 3              # Diff/FM grid score averaging passes
    smooth_sigma: float = 1.5        # Gaussian smooth σ for stochastic region boundaries
    n_scatter: int = 300
    sample_quality_n: int = 2000     # samples per model for sample quality plot

    # ── Output ──
    outdir: str = "./experiments"
    device: Optional[str] = None
    verbose: bool = True
    train_only: bool = False
    rerun: Optional[list] = None   # e.g. ["FM-Path", "NF-Ball"] → retrain + re-score
    force_restart: bool = False    # delete existing checkpoints, start fresh
    timing_only: bool = False      # train + score timing only, skip volume/baselines/plots

    def copy(self, **overrides):
        d = asdict(self)
        d.update(overrides)
        return Config(**d)

    def to_dict(self):
        return asdict(self)


CFG = Config()