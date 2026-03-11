# Generative Conformal Prediction

Constructing conformal prediction regions for continuous response variables using generative models (Normalizing Flows, Diffusion Models, Flow Matching), with systematic comparison against existing methods.

## Problem

Given features X, construct a prediction region C(x) = {y : s(x,y) ≤ τ} such that P(Y ∈ C(X)) ≥ 1−α, while minimizing volume(C(x)).

## Methods

**Z-space methods** (ours):

| Method | Display Name | Score Function | Model |
|--------|-------------|----------------|-------|
| NF-Ball | CONTRA | s = ‖z‖² | Normalizing Flow |
| NF-NLL | JAPAN | s = 0.5‖z‖² − log\|det J\| | Normalizing Flow |
| Diff-Denoise | DDPM | s = E[‖ε − ε̂‖²] | Diffusion (DDPM) |
| FM-Path | FM | s = E[‖v − v̂‖²] | Flow Matching (OT-CFM) |

**Baselines** (Y-space):

RCP, NLE, PCP-Diff, DistSplit, CQR, MCQR

## Installation

```bash
pip install torch numpy scipy matplotlib
```

## Quick Start

### 1. Run experiment

```bash
# Single run (parameter tuning)
python experiment.py --dataset spiral --n_total 15000

# Formal experiment with multiple repeats
python experiment.py --dataset spiral --n_total 15000 --n_repeats 10
```

### 2. Draw publication-quality region plots

```bash
python single_plot.py --exp_dir ./experiments/spiral_s0_single --x_index 750
```

### 3. Draw violin plots (requires n_repeats > 1)

```bash
python replot.py --exp_dir ./experiments/spiral_s0
```

## Datasets

### Synthetic (2D → 2D)

All synthetic datasets follow Y = f(X) + ε, where f(X) is a nonlinear function and ε has shape-specific structure.

| Dataset | X dim | Noise Shape | Key Feature |
|---------|-------|-------------|-------------|
| spiral | 2 | Archimedean spiral | Curved manifold |
| ring | 2 | Concentric ring | Annular region |
| moon | 2 | Two moons | Disconnected modes |
| mixture_gaussian | 2 | Gaussian mixture | Multi-modal |
| pinwheel | 7 | 6-arm pinwheel | High-dim X + multi-modal |
| checkerboard | 8 | 4×4 grid | High-dim X + many modes |
| twomoons | 9 | Two moons | High-dim X + disconnected |

### Real

| Dataset | Description |
|---------|-------------|
| bio | Biological response (q=1) |
| energy | Building energy (q=2) |
| taxi | NYC taxi (q=2) |
| hurricane | Tropical cyclone tracks (q=2) |

## Project Structure

```
GenerativeConformal/
├── experiment.py           # Main entry: train + evaluate + save
├── config.py               # All hyperparameters (dataclass)
├── single_plot.py          # Publication-quality region plots
├── replot.py               # Violin plots from saved results
│
├── models/
│   ├── nf.py               # Normalizing Flow (RealNVP / NSF)
│   ├── diffusion.py        # Conditional DDPM (FiLM + ResBlocks)
│   └── flow_matching.py    # OT-CFM (FiLM + ResBlocks)
│
├── scores/
│   ├── nf_scores.py        # NF-Ball, NF-NLL
│   ├── diffusion_scores.py # Diff-Denoise
│   ├── fm_scores.py        # FM-Path
│   └── ...                 # ODE-Ball, density, mode attraction
│
├── baselines.py            # RCP, NLE, PCP, DistSplit, CQR, MCQR
├── conformal.py            # Conformal calibration and evaluation
├── datasets.py             # Synthetic and real data generators
├── training.py             # Model training loops
├── volume.py               # MC volume estimation
├── splitting.py            # Train/calibration/test split
└── plotting.py             # Plotting utilities (used by experiment.py)
```

## Output Structure

```
experiments/
├── spiral_s0/                    # n_repeats > 1 (formal)
│   ├── config.json
│   ├── results.json              # Aggregated mean ± std
│   ├── repeat_000/
│   │   ├── nf_model.pt
│   │   ├── diff_model.pt
│   │   ├── fm_model.pt
│   │   ├── data_split.pt
│   │   ├── baselines.pt          # Fitted baseline objects
│   │   └── results.json          # Per-repeat results
│   ├── repeat_001/
│   │   └── ...
│   └── replot/                   # Publication figures
│
├── spiral_s0_single/             # n_repeats == 1 (tuning)
│   ├── config.json
│   ├── repeat_000/
│   │   └── ...
│   └── replot/
```

## Key CLI Arguments

### experiment.py

```
--dataset           Dataset name (spiral, pinwheel, twomoons, ...)
--n_total           Total data points (default: 10000)
--n_repeats         Number of repeats (default: 1)
--seed              Random seed (default: 0)

--nf_flow_type      "realnvp" or "nsf"
--nf_cond_net       "mlp" or "resnet"
--nf_n_layers       Number of coupling layers (default: 8)
--nf_n_bins         NSF spline bins (default: 8)
--nf_lr             NF learning rate (default: 1e-3)
--nf_epochs         NF training epochs (default: 500)

--diff_n_blocks     Diffusion ResBlocks (default: 8)
--diff_epochs       Diffusion epochs (default: 2000)
--diff_score_timesteps  MC score: number of timesteps (default: 15)
--diff_score_repeats    MC score: noise repeats (default: 8)

--fm_n_blocks       FM ResBlocks (default: 8)
--fm_epochs         FM epochs (default: 2000)
--fm_score_timesteps    (default: 15)
--fm_score_repeats      (default: 8)

--methods           Comma-separated method subset (e.g. "NF-Ball,FM-Path")
--no_baselines      Skip all baselines
--device            "cpu" or "cuda"
```

### single_plot.py

```
--exp_dir           Experiment directory
--x_index           Test point index for region plot
--rep               Repeat index (default: 0)
--grid_res          Grid resolution (default: 200)
--dpi               Output DPI (default: 200)
--device            "cpu" or "cuda"
```

### replot.py

```
--exp_dir           Experiment directory (needs n_repeats > 1)
```

## Model Architecture

All three generative models share the same conditioning mechanism:

- **Conditioning network**: MLP or ResNet that maps X → conditioning vector h
- **Score/velocity network**: FiLM-conditioned ResBlocks (scale + shift injection)
- **Time embedding**: Sinusoidal positional encoding

| Model | Transform | Training Objective |
|-------|-----------|-------------------|
| NF (RealNVP/NSF) | Invertible coupling layers | Exact log-likelihood |
| Diffusion (DDPM) | Noise prediction ε̂(y_t, t, x) | E[‖ε − ε̂‖²] |
| Flow Matching (OT-CFM) | Velocity field v̂(y_t, t, x) | E[‖v − v̂‖²] |

## References

1. **CONTRA**: Fang, Tan & Huang, ICLR 2025 — NF-based conformal prediction regions
2. **DDPM**: Ho et al., 2020 — Denoising Diffusion Probabilistic Models
3. **Flow Matching**: Lipman et al., 2023 — Conditional Flow Matching
4. **RealNVP**: Dinh et al., 2017
5. **NSF**: Durkan et al., 2019 — Neural Spline Flows
