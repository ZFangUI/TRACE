#!/usr/bin/env python3
"""Replot violin charts from existing results.json files.

Usage:
    python replot.py --exp_dir ./experiments/spiral_s0
    python replot.py --exp_dir ./experiments/spiral_s0 --alpha 0.1

Reads results.json from each repeat_XXX subfolder,
applies renaming and filtering, then produces two separate figures:
  1. coverage violin
  2. volume violin
"""

import argparse
import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt


# ── Rename and filter config ──

RENAME = {
    "NF-Ball": "CONTRA",
    "NF-NLL": "JAPAN",
    "Diff-Denoise": "TRACE-Diff",
    "FM-Path": "TRACE-FM",
}

EXCLUDE = {"DistSplit", "CQR"}

# Display order
ORDER = ["TRACE-Diff", "TRACE-FM","CONTRA", "JAPAN", "PCP-Diff", "MCQR","RCP", "NLE",  ]

# Fixed method-color mapping
METHOD_COLORS = {
    "CONTRA":   "#9b59b6",  # purple
    "JAPAN":    "#9394e7",  # orange
    "TRACE-Diff":     "#3498db",  # blue
    "TRACE-FM":       "#e74c3c",  # red
    "RCP":      "#f1ba7e", 
    "NLE":      "#1abc9c",  # teal
    "PCP-Diff": "#b1ce46",  # dark orange
    "MCQR":     "#c497b2",  # gold
}
FALLBACK_COLOR = "#7f8c8d"  # gray


    # # Color palettes
    # solo_fill =  ["#f5c6c6", "#b3d9f7", "#b3f0cc", "#d5b3f0",
    #               "#fce3a8", "#f5d6c6", "#c6e8f5", "#d9f5c6",
    #               "#f0d5b3", "#c6f5f5", "#f5c6e8", "#d9d5f0"]
    # solo_edge =  ["#c0392b", "#2471a3", "#1e8449", "#7d3c98",
    #               "#d68910", "#a04010", "#1a5276", "#27ae60",
    #               "#b7950b", "#117a65", "#a93226", "#6c3483"]
    # # Ellipsoid group colors
    # ellip_fills = ["#c6e8f5", "#d5b3f0"]
    # ellip_edges = ["#1a5276", "#7d3c98"]
    # ellip_styles = ["-", "--"]
    # # Rectangle group colors
    # rect_fills  = ["#b3f0cc", "#fce3a8", "#f5c6c6"]
    # rect_edges  = ["#1e8449", "#d68910", "#c0392b"]
    # rect_styles = ["-", "--", ":"]

def load_all_repeats(exp_dir):
    """Load results from all repeat_XXX/results.json files."""
    all_repeats = {}

    # Try repeat subdirs first
    rep_dirs = sorted(glob.glob(os.path.join(exp_dir, "repeat_*")))
    if not rep_dirs:
        # Single run: results.json in exp_dir itself
        rep_dirs = [exp_dir]

    for rd in rep_dirs:
        path = os.path.join(rd, "results.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            obj = json.load(f)
        obj.pop("_time_seconds", None)

        for method, res in obj.items():
            if method.startswith("_"):
                continue
            display_name = RENAME.get(method, method)
            if method in EXCLUDE or display_name in EXCLUDE:
                continue

            if display_name not in all_repeats:
                all_repeats[display_name] = {"coverage": [], "volume": []}

            cov = res.get("coverage", float("nan"))
            vol = res.get("volume", float("nan"))
            if isinstance(cov, str):
                cov = float(cov)
            if isinstance(vol, str):
                vol = float(vol)
            all_repeats[display_name]["coverage"].append(cov)
            all_repeats[display_name]["volume"].append(vol)

    return all_repeats


def plot_violin_single(all_repeats, metric, ylabel, alpha=0.1,
                       title=None, save_path=None):
    """Single violin plot for one metric (coverage or volume)."""
    # Order methods
    names = [n for n in ORDER if n in all_repeats]
    for n in all_repeats:
        if n not in names:
            names.append(n)

    n_methods = len(names)
    if n_methods == 0:
        return None

    colors = [METHOD_COLORS.get(n, FALLBACK_COLOR) for n in names]

    fig, ax = plt.subplots(figsize=(max(6, 0.9 * n_methods + 2), 4.5))

    data_list = []
    positions = []
    valid_names = []
    valid_colors = []

    for i, name in enumerate(names):
        vals = all_repeats[name][metric]
        vals_clean = [v for v in vals if not np.isnan(v)]
        if len(vals_clean) >= 1:
            data_list.append(vals_clean)
            positions.append(i)
            valid_names.append(name)
            valid_colors.append(colors[i])

    if not data_list:
        plt.close(fig)
        return None

    # Violin (only if >= 2 data points)
    violin_data = []
    violin_pos = []
    violin_colors = []
    for i, vals in enumerate(data_list):
        if len(vals) >= 2:
            violin_data.append(vals)
            violin_pos.append(positions[i])
            violin_colors.append(valid_colors[i])

    if violin_data:
        parts = ax.violinplot(
            violin_data, positions=violin_pos,
            showmeans=False, showmedians=False, showextrema=False,
        )
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(violin_colors[i])
            pc.set_edgecolor("black")
            pc.set_linewidth(0.8)
            pc.set_alpha(0.55)

    # Median line
    for i, vals in enumerate(data_list):
        med = np.median(vals)
        pos = positions[i]
        ax.hlines(med, pos - 0.25, pos + 0.25,
                  colors="black", linewidths=2.0, zorder=4)

    # Jittered strip
    rng = np.random.RandomState(42)
    for i, vals in enumerate(data_list):
        pos = positions[i]
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            pos + jitter, vals,
            s=22, c=valid_colors[i], edgecolors="white",
            linewidths=0.5, alpha=0.8, zorder=5,
        )

    # Target coverage line
    if metric == "coverage":
        target_cov = 1 - alpha
        ax.axhline(target_cov, color="red", linestyle="--",
                    linewidth=1.2, alpha=0.7, label=f"1−α = {target_cov}")
        ax.legend(fontsize=9, loc="lower left")
        all_vals = [v for vals in data_list for v in vals]
        ax.set_ylim(max(0.6, min(all_vals) - 0.05), 1.02)

    ax.set_xticks(range(n_methods))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, axis="y", alpha=0.2, linewidth=0.5)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return fig


def main():
    parser = argparse.ArgumentParser(description="Replot violins from results.json")
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="Experiment directory (e.g. ./experiments/spiral_s0)")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--prefix", type=str, default=None,
                        help="Output filename prefix (default: inferred from exp_dir)")
    args = parser.parse_args()

    all_repeats = load_all_repeats(args.exp_dir)
    if not all_repeats:
        print(f"No results found in {args.exp_dir}")
        return

    prefix = args.prefix or os.path.basename(args.exp_dir.rstrip("/"))
    out_dir = args.exp_dir

    print(f"  Methods found: {list(all_repeats.keys())}")
    print(f"  Repeats per method: {[len(v['coverage']) for v in all_repeats.values()]}")

    plot_violin_single(
        all_repeats, "coverage", "Coverage", alpha=args.alpha,
        title=f"{prefix} — Coverage",
        save_path=os.path.join(out_dir, f"{prefix}_coverage_violin.png"))

    plot_violin_single(
        all_repeats, "volume", "Volume", alpha=args.alpha,
        title=f"{prefix} — Volume",
        save_path=os.path.join(out_dir, f"{prefix}_volume_violin.png"))


if __name__ == "__main__":
    main()