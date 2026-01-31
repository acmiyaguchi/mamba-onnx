"""
Plotting utilities for Mamba custom-op benchmarks.

Reads the CSV produced by suite.py (scale mode) and generates PNG plots.
"""
import argparse
import os

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
BACKEND_STYLES = {"zig": "-", "cpp": "--"}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_scaling(df, axis_col, fixed_col, fixed_val, scenario_label, output_dir):
    """Generic scaling plot (used for both L and D sweeps)."""
    sub = df[df["scenario"] == scenario_label]
    if sub.empty:
        print(f"No {scenario_label} data found, skipping.")
        return

    backends = sorted(sub["backend"].unique())

    plt.figure(figsize=(12, 7))
    for op in OPS:
        for backend in backends:
            mask = (sub["op"] == op) & (sub["backend"] == backend)
            chunk = sub[mask].sort_values(axis_col)
            if chunk.empty:
                continue
            style = BACKEND_STYLES.get(backend, "-")
            label = f"{OP_LABELS[op]} ({backend})" if len(backends) > 1 else OP_LABELS[op]
            plt.plot(chunk[axis_col], chunk["latency_ms"], marker="o", linestyle=style,
                     color=OP_COLORS[op], label=label, linewidth=2)

    plt.xlabel(f"{axis_col}", fontsize=12)
    plt.ylabel("Latency (ms)", fontsize=12)
    plt.title(f"Selective Scan Latency vs {axis_col} ({fixed_col}={fixed_val})", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    fname = os.path.join(output_dir, f"scaling_{axis_col}.png")
    plt.savefig(fname, dpi=150)
    print(f"Saved {fname}")
    plt.close()


def plot_scaling_L(df, output_dir):
    _plot_scaling(df, "L", "D", 768, "Scaling_L", output_dir)


def plot_scaling_D(df, output_dir):
    _plot_scaling(df, "D", "L", 1024, "Scaling_D", output_dir)


def plot_op_comparison_bar(df, output_dir):
    """Bar chart comparing all 4 ops at L=1024, D=768."""
    sub = df[(df["scenario"] == "Scaling_L") & (df["L"] == 1024)]
    if sub.empty:
        sub = df[df["scenario"] == "Scaling_L"]
        if sub.empty:
            print("No data for op comparison bar chart, skipping.")
            return
        sub = sub[sub["L"] == sub["L"].max()]

    # Pick one backend (prefer zig)
    backends = sorted(sub["backend"].unique())
    backend = "zig" if "zig" in backends else backends[0]
    sub = sub[sub["backend"] == backend]

    latencies, labels, colors = [], [], []
    for op in OPS:
        row = sub[sub["op"] == op]
        if row.empty:
            continue
        latencies.append(row["latency_ms"].values[0])
        labels.append(OP_LABELS[op])
        colors.append(OP_COLORS[op])

    plt.figure(figsize=(10, 6))
    x = np.arange(len(labels))
    bars = plt.bar(x, latencies, color=colors, edgecolor="black", linewidth=0.5)

    for bar, lat in zip(bars, latencies):
        plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.05,
                 f"{lat:.2f}ms", ha="center", va="bottom", fontsize=10)

    L_val = int(sub["L"].values[0])
    plt.xticks(x, labels, fontsize=11)
    plt.ylabel("Latency (ms)", fontsize=12)
    plt.title(f"Op Variant Comparison (L={L_val}, D=768, backend={backend})", fontsize=14)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(output_dir, "op_comparison.png")
    plt.savefig(fname, dpi=150)
    print(f"Saved {fname}")
    plt.close()


def plot_overhead_analysis(df, output_dir):
    """Visualize overhead from exact discretization and fusion."""
    sub = df[df["scenario"] == "Scaling_L"]
    if sub.empty:
        return

    # Pick one backend
    backends = sorted(sub["backend"].unique())
    backend = "zig" if "zig" in backends else backends[0]
    sub = sub[sub["backend"] == backend]

    # Pivot to wide format: rows=L, columns=op
    piv = sub.pivot_table(index="L", columns="op", values="latency_ms").sort_index()
    if not all(op in piv.columns for op in OPS):
        print("Not all ops present for overhead analysis, skipping.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Exact vs Linear overhead
    exact_oh = (piv["SelectiveScanExact"] / piv["SelectiveScan"] - 1) * 100
    fused_exact_oh = (piv["SelectiveScanFusedExact"] / piv["SelectiveScanFused"] - 1) * 100
    ax1.plot(piv.index, exact_oh, marker="o", color="#2196F3", label="Non-fused", linewidth=2)
    ax1.plot(piv.index, fused_exact_oh, marker="s", color="#9C27B0", label="Fused", linewidth=2)
    ax1.set_xlabel("Sequence Length (L)", fontsize=12)
    ax1.set_ylabel("Overhead (%)", fontsize=12)
    ax1.set_title("Exact Discretization Overhead", fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Fusion overhead
    fusion_lin = (piv["SelectiveScanFused"] / piv["SelectiveScan"] - 1) * 100
    fusion_exact = (piv["SelectiveScanFusedExact"] / piv["SelectiveScanExact"] - 1) * 100
    ax2.plot(piv.index, fusion_lin, marker="o", color="#FF9800", label="Linear", linewidth=2)
    ax2.plot(piv.index, fusion_exact, marker="s", color="#9C27B0", label="Exact", linewidth=2)
    ax2.set_xlabel("Sequence Length (L)", fontsize=12)
    ax2.set_ylabel("Overhead (%)", fontsize=12)
    ax2.set_title("Fusion (Softplus+SiLU) Overhead", fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(output_dir, "overhead_analysis.png")
    plt.savefig(fname, dpi=150)
    print(f"Saved {fname}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot Mamba benchmark results")
    parser.add_argument("--input", default="benchmarks/results/benchmark_data.csv",
                        help="CSV from suite.py scale mode")
    parser.add_argument("--output-dir", default="benchmarks/results",
                        help="Directory for PNG output")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"No data at {args.input}. Run 'python benchmarks/suite.py --mode scale' first.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    plot_scaling_L(df, args.output_dir)
    plot_scaling_D(df, args.output_dir)
    plot_op_comparison_bar(df, args.output_dir)
    plot_overhead_analysis(df, args.output_dir)

    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
