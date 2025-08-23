# visualize.py - 
#
# Ben Cavanagh
# 08-23-2025
# Description: 
#

"""
noisy_channel_viz.py
Visualize results produced by the sampling program:
- summary_stats.csv
- *_samples.csv  (per (pattern,length) raw outputs + Hamming distances)
- *_perbit.csv   (per-bit error probabilities)

Usage example:
    from noisy_channel_viz import visualize_noisy_channel
    visualize_noisy_channel("results", out_dir="viz",
                            lengths=[128,256,512],
                            patterns=["all_0s","all_1s","alternating_0_start","alternating_1_start"])
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Helpers to find/load data
# ----------------------------

def _find_files(results_dir, suffix):
    return sorted(glob.glob(os.path.join(results_dir, f"*{suffix}.csv")))

def _samples_path(results_dir, pattern, length):
    return os.path.join(results_dir, f"{pattern}_len{length}_samples.csv")

def _perbit_path(results_dir, pattern, length):
    return os.path.join(results_dir, f"{pattern}_len{length}_perbit.csv")

def _load_samples(results_dir, pattern, length):
    path = _samples_path(results_dir, pattern, length)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing samples CSV: {path}")
    return pd.read_csv(path)

def _load_perbit(results_dir, pattern, length):
    path = _perbit_path(results_dir, pattern, length)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing per-bit CSV: {path}")
    return pd.read_csv(path)

# ----------------------------
# Plotting primitives (matplotlib only; single plot per figure)
# ----------------------------

def _plot_histogram(data, title, xlabel, ylabel, outpath):
    plt.figure()
    # Choose bins sensibly for integer distances
    bins = np.arange(min(data), max(data) + 2) - 0.5
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def _plot_boxplot(groups, labels, title, ylabel, outpath):
    plt.figure()
    plt.boxplot(groups, labels=labels, showmeans=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def _plot_ecdf(data, title, xlabel, ylabel, outpath):
    plt.figure()
    x = np.sort(np.asarray(data))
    y = np.arange(1, len(x)+1) / len(x)
    plt.step(x, y, where='post')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def _plot_line(x, y, title, xlabel, ylabel, outpath):
    plt.figure()
    plt.plot(x, y, marker='o', linewidth=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def _plot_heatmap(matrix, title, xlabel, ylabel, outpath):
    plt.figure()
    plt.imshow(matrix, aspect='auto', interpolation='nearest')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

# ----------------------------
# Public API
# ----------------------------

def visualize_noisy_channel(results_dir: str,
                            out_dir: str = "viz",
                            lengths = None,
                            patterns = None,
                            make_correlation=False):
    """
    Create a suite of figures from the saved CSVs.

    Parameters
    ----------
    results_dir : str
        Directory that contains the CSV outputs from the sampling program.
    out_dir : str
        Where to save figures.
    lengths : list[int] | None
        Subset of lengths to visualize. If None, inferred from files.
    patterns : list[str] | None
        Subset of patterns to visualize. If None, uses all four known patterns.
    make_correlation : bool
        If True, compute & save error-correlation matrices from samples
        (slower; requires reading outputs and inputs).
    """
    os.makedirs(out_dir, exist_ok=True)

    # Infer defaults
    if patterns is None:
        patterns = ["all_0s", "all_1s", "alternating_0_start", "alternating_1_start"]

    if lengths is None:
        # Parse lengths from available per-bit files
        perbit_files = _find_files(results_dir, "_perbit")
        inferred = set()
        for f in perbit_files:
            # expect ..._len{L}_perbit.csv
            base = os.path.basename(f)
            try:
                L = int(base.split("_len")[1].split("_")[0])
                inferred.add(L)
            except Exception:
                pass
        lengths = sorted(inferred)
        if not lengths:
            raise RuntimeError("Could not infer lengths; supply `lengths=` or generate per-bit CSVs first.")

    # 1) Hamming distance histograms per (length, pattern)
    for L in lengths:
        for pat in patterns:
            df = _load_samples(results_dir, pat, L)
            hd = df["hamming_distance"].to_numpy()
            _plot_histogram(
                hd,
                title=f"Hamming distance distribution — {pat}, L={L}",
                xlabel="Hamming distance (bit flips)",
                ylabel="Count",
                outpath=os.path.join(out_dir, f"hist_hd_{pat}_L{L}.png"),
            )
            _plot_ecdf(
                hd,
                title=f"ECDF of Hamming distance — {pat}, L={L}",
                xlabel="Hamming distance (bit flips)",
                ylabel="Empirical CDF",
                outpath=os.path.join(out_dir, f"ecdf_hd_{pat}_L{L}.png"),
            )

    # 2) Boxplot comparing patterns at each length
    for L in lengths:
        groups, labels = [], []
        for pat in patterns:
            df = _load_samples(results_dir, pat, L)
            groups.append(df["hamming_distance"].to_numpy())
            labels.append(pat)
        _plot_boxplot(
            groups, labels,
            title=f"Hamming distance by pattern — L={L}",
            ylabel="Hamming distance (bit flips)",
            outpath=os.path.join(out_dir, f"box_hd_L{L}.png"),
        )

    # 3) Per-bit error probabilities (line) and across-pattern heatmap
    for L in lengths:
        # Line plots: one per (length, pattern)
        perbit_stack = []
        for pat in patterns:
            dfp = _load_perbit(results_dir, pat, L)
            x = dfp["bit_position"].to_numpy()
            y = dfp["error_probability"].to_numpy()
            perbit_stack.append((pat, y))
            _plot_line(
                x, y,
                title=f"Per-bit error probability — {pat}, L={L}",
                xlabel="Bit position",
                ylabel="Error probability",
                outpath=os.path.join(out_dir, f"perbit_{pat}_L{L}.png"),
            )
        # Heatmap over patterns for same length
        # rows = patterns, cols = bit positions
        matrix = np.vstack([y for _, y in perbit_stack])
        _plot_heatmap(
            matrix,
            title=f"Per-bit error probability (patterns × bits) — L={L}",
            xlabel="Bit position",
            ylabel="Pattern index (0..{}: {})".format(len(patterns)-1, ", ".join(patterns)),
            outpath=os.path.join(out_dir, f"perbit_heatmap_L{L}.png"),
        )

    # 4) Optional: error-correlation matrix (bit-error correlations) from samples
    if make_correlation:
        for L in lengths:
            for pat in patterns:
                df = _load_samples(results_dir, pat, L)
                # Build error matrix: rows = samples, cols = bit positions
                # Derive inputs/outputs as integer arrays
                outs = np.array([[int(b) for b in s] for s in df["output"].tolist()], dtype=int)
                ins  = np.array([[int(b) for b in s] for s in df["input"].tolist()], dtype=int)
                err  = (outs != ins).astype(int)

                # Compute Pearson corr across columns (bit positions)
                # Guard for constant columns (all zeros) to avoid NaNs
                # Replace NaNs with 0 (no variance => undefined correlation -> treat as 0)
                if err.shape[0] > 1:
                    corr = np.corrcoef(err, rowvar=False)
                    corr = np.nan_to_num(corr, nan=0.0)
                else:
                    corr = np.zeros((err.shape[1], err.shape[1]), dtype=float)

                # Save CSV and figure
                corr_csv = os.path.join(out_dir, f"corr_{pat}_L{L}.csv")
                pd.DataFrame(corr).to_csv(corr_csv, index=False)
                _plot_heatmap(
                    corr,
                    title=f"Error correlation (bits × bits) — {pat}, L={L}",
                    xlabel="Bit position",
                    ylabel="Bit position",
                    outpath=os.path.join(out_dir, f"corr_{pat}_L{L}.png"),
                )

    print(f"Visualization complete. Figures saved to: {os.path.abspath(out_dir)}")

visualize_noisy_channel("results")
