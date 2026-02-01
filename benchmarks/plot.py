"""
Plotting utilities for Mamba custom-op benchmarks.

Reads pytest-benchmark JSON output and generates PNG plots.

Usage:
    # From saved benchmark results:
    pytest benchmarks/ --benchmark-only --benchmark-save=run1
    python benchmarks/plot.py

    # Or specify a JSON file directly:
    python benchmarks/plot.py --input .benchmarks/Linux-CPython-.../0001_run1.json
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
OPS = ["SelectiveScan", "SelectiveScanExact", "SelectiveScanFused", "SelectiveScanFusedExact"]
OP_COLORS = {
    "SelectiveScan": "#4CAF50",
    "SelectiveScanExact": "#2196F3",
    "SelectiveScanFused": "#FF9800",
    "SelectiveScanFusedExact": "#9C27B0",
}
OP_LABELS = {
    "SelectiveScan": "Linear (fastest)",
    "SelectiveScanExact": "Exact",
    "SelectiveScanFused": "Linear + Fused",
    "SelectiveScanFusedExact": "Exact + Fused",
}
BACKEND_COLORS = {"cpp": "#E53935", "zig": "#1E88E5"}
BACKEND_STYLES = {"zig": "-", "cpp": "--"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _find_latest_json(benchmarks_dir=".benchmarks"):
    """Find the most recently modified JSON file in the .benchmarks tree."""
    pattern = os.path.join(benchmarks_dir, "**", "*.json")
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def _load_benchmark_json(path):
    """Load a pytest-benchmark JSON file and return a DataFrame."""
    with open(path) as f:
        data = json.load(f)

    rows = []
    for bench in data.get("benchmarks", []):
        stats = bench.get("stats", {})
        extra = bench.get("extra_info", {})
        rows.append({
            "name": bench.get("name", ""),
            "group": bench.get("group", ""),
            "latency_ms": stats.get("mean", 0) * 1000,
            "median_ms": stats.get("median", 0) * 1000,
            "stddev_ms": stats.get("stddev", 0) * 1000,
            "min_ms": stats.get("min", 0) * 1000,
            "op": extra.get("op", ""),
            "backend": extra.get("backend", ""),
            "scenario": extra.get("scenario", ""),
            "L": extra.get("L", 0),
            "D": extra.get("D", 0),
            "B": extra.get("B", 0),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_scaling(df, axis_col, fixed_col, fixed_val, scenario_label, output_dir):
    """Scaling plot with one subplot per backend."""
    sub = df[df["scenario"] == scenario_label]
    if sub.empty:
        print(f"No {scenario_label} data found, skipping.")
        return

    backends = sorted(sub["backend"].unique())
    n = len(backends)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6), sharey=True, squeeze=False)
    axes = axes[0]

    for ax, backend in zip(axes, backends):
        for op in OPS:
            mask = (sub["op"] == op) & (sub["backend"] == backend)
            chunk = sub[mask].sort_values(axis_col)
            if chunk.empty:
                continue
            ax.plot(chunk[axis_col], chunk["median_ms"], marker="o",
                    color=OP_COLORS[op], label=OP_LABELS[op], linewidth=2)
            ax.fill_between(
                chunk[axis_col],
                chunk["median_ms"] - chunk["stddev_ms"],
                chunk["median_ms"] + chunk["stddev_ms"],
                color=OP_COLORS[op], alpha=0.15,
            )
        ax.set_xlabel(axis_col, fontsize=12)
        ax.set_title(f"{backend} ({fixed_col}={fixed_val})", fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Median Latency (ms)", fontsize=12)
    fig.suptitle(f"Selective Scan Latency vs {axis_col}", fontsize=14, y=1.02)
    plt.tight_layout()
    fname = os.path.join(output_dir, f"scaling_{axis_col}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close()


def plot_scaling_L(df, output_dir):
    _plot_scaling(df, "L", "D", 768, "Scaling_L", output_dir)


def plot_scaling_D(df, output_dir):
    _plot_scaling(df, "D", "L", 1024, "Scaling_D", output_dir)


def plot_speedup(df, output_dir):
    """Grouped bar chart: ONNX speedup vs PyTorch for each op, by backend."""
    onnx = df[df["group"] == "compare-onnx"]
    pytorch = df[df["group"] == "compare-pytorch"]
    if onnx.empty or pytorch.empty:
        print("No compare data for speedup chart, skipping.")
        return

    pt_lookup = pytorch.groupby("op")["median_ms"].median().to_dict()

    backends = sorted(onnx["backend"].unique())
    n_ops = len(OPS)
    bar_width = 0.35
    x = np.arange(n_ops)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [2, 1]})

    # --- Top: speedup bars ---
    for i, backend in enumerate(backends):
        speedups = []
        for op in OPS:
            row = onnx[(onnx["op"] == op) & (onnx["backend"] == backend)]
            onnx_ms = row["median_ms"].values[0] if not row.empty else 0
            if onnx_ms <= 0 or op not in pt_lookup or pt_lookup[op] == 0:
                speedups.append(0)
            else:
                speedups.append(pt_lookup[op] / onnx_ms)

        offset = (i - (len(backends) - 1) / 2) * bar_width
        bars = ax1.bar(x + offset, speedups, bar_width, label=backend,
                       color=BACKEND_COLORS.get(backend, "#999"),
                       edgecolor="black", linewidth=0.5)
        for bar, s in zip(bars, speedups):
            if s > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                         f"{s:.0f}x", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels([OP_LABELS[op] for op in OPS], fontsize=10)
    ax1.set_ylabel("Speedup vs PyTorch", fontsize=12)
    ax1.set_title("ONNX Custom Op Speedup over PyTorch Reference (B=1, D=768, L=1024)", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3)

    # --- Bottom: absolute latency bars (log scale) ---
    all_backends = ["pytorch"] + backends
    n_bars = len(all_backends)
    bar_width_abs = 0.8 / n_bars

    for i, backend in enumerate(all_backends):
        latencies = []
        for op in OPS:
            if backend == "pytorch":
                latencies.append(pt_lookup.get(op, 0))
            else:
                row = onnx[(onnx["op"] == op) & (onnx["backend"] == backend)]
                latencies.append(row["median_ms"].values[0] if not row.empty else 0)

        offset = (i - (n_bars - 1) / 2) * bar_width_abs
        color = "#78909C" if backend == "pytorch" else BACKEND_COLORS.get(backend, "#999")
        ax2.bar(x + offset, latencies, bar_width_abs, label=backend, color=color,
                edgecolor="black", linewidth=0.5)

    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels([OP_LABELS[op] for op in OPS], fontsize=10)
    ax2.set_ylabel("Median Latency (ms, log)", fontsize=12)
    ax2.set_title("Absolute Latency Comparison", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(output_dir, "speedup.png")
    plt.savefig(fname, dpi=150)
    print(f"Saved {fname}")
    plt.close()


def plot_feature_cost(df, output_dir):
    """Grouped bar chart showing absolute latency of each op variant at selected L values."""
    sub = df[df["scenario"] == "Scaling_L"]
    if sub.empty:
        print("No Scaling_L data for feature cost chart, skipping.")
        return

    # Pick one backend (prefer zig for lower noise)
    backends = sorted(sub["backend"].unique())
    backend = "zig" if "zig" in backends else backends[0]
    sub = sub[sub["backend"] == backend]

    piv = sub.pivot_table(index="L", columns="op", values="median_ms").sort_index()
    if not all(op in piv.columns for op in OPS):
        print("Not all ops present for feature cost chart, skipping.")
        return

    # Pick a few representative L values
    sample_Ls = [L for L in [128, 1024, 4096] if L in piv.index]
    if not sample_Ls:
        print("No matching L values for feature cost chart, skipping.")
        return
    piv = piv.loc[sample_Ls]

    fig, ax = plt.subplots(figsize=(10, 6))

    n_groups = len(sample_Ls)
    n_bars = len(OPS)
    bar_width = 0.8 / n_bars
    x = np.arange(n_groups)

    for i, op in enumerate(OPS):
        offset = (i - (n_bars - 1) / 2) * bar_width
        vals = piv[op].values
        bars = ax.bar(x + offset, vals, bar_width, label=OP_LABELS[op],
                      color=OP_COLORS[op], edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"L={L}" for L in sample_Ls], fontsize=11)
    ax.set_ylabel("Median Latency (ms)", fontsize=12)
    ax.set_title(f"Feature Cost Breakdown by Op Variant ({backend}, D=768)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(output_dir, "feature_cost.png")
    plt.savefig(fname, dpi=150)
    print(f"Saved {fname}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot Mamba benchmark results")
    parser.add_argument("--input", default=None,
                        help="pytest-benchmark JSON file (default: latest in .benchmarks/)")
    parser.add_argument("--output-dir", default="benchmarks/results",
                        help="Directory for PNG output")
    args = parser.parse_args()

    if args.input is None:
        args.input = _find_latest_json()
        if args.input is None:
            print("No benchmark JSON found in .benchmarks/.")
            print("Run: pytest benchmarks/ --benchmark-only --benchmark-save=baseline")
            return

    if not os.path.exists(args.input):
        print(f"No data at {args.input}.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    df = _load_benchmark_json(args.input)
    print(f"Loaded {len(df)} benchmark entries from {args.input}")

    plot_scaling_L(df, args.output_dir)
    plot_scaling_D(df, args.output_dir)
    plot_speedup(df, args.output_dir)
    plot_feature_cost(df, args.output_dir)

    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
