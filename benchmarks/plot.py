"""
Plotting utilities for Mamba custom op benchmarks.

Generates visualizations for all 4 op variants:
  - SelectiveScan (linear, fastest)
  - SelectiveScanExact (exact discretization)
  - SelectiveScanFused (linear + Softplus/SiLU)
  - SelectiveScanFusedExact (exact + Softplus/SiLU)
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Op configuration
OPS = ['SelectiveScan', 'SelectiveScanExact', 'SelectiveScanFused', 'SelectiveScanFusedExact']
OP_COLORS = {
    'SelectiveScan': '#4CAF50',        # Green (fastest)
    'SelectiveScanExact': '#2196F3',   # Blue
    'SelectiveScanFused': '#FF9800',   # Orange
    'SelectiveScanFusedExact': '#9C27B0',  # Purple
}
OP_LABELS = {
    'SelectiveScan': 'Linear (fastest)',
    'SelectiveScanExact': 'Exact',
    'SelectiveScanFused': 'Linear + Fused',
    'SelectiveScanFusedExact': 'Exact + Fused',
}


def plot_scaling_L(df):
    """Plot latency vs sequence length for all 4 ops."""
    df_L = df[df['Scenario'] == 'Scaling_L']
    if df_L.empty:
        print("No Scaling_L data found.")
        return

    plt.figure(figsize=(12, 7))

    for op in OPS:
        col = f'{op}_ms'
        if col in df_L.columns:
            plt.plot(df_L['L'], df_L[col], marker='o',
                    color=OP_COLORS[op], label=OP_LABELS[op], linewidth=2)

    plt.xlabel('Sequence Length (L)', fontsize=12)
    plt.ylabel('Latency (ms)', fontsize=12)
    plt.title('Mamba Selective Scan Latency vs Sequence Length (D=768)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("benchmarks/results/scaling_L.png", dpi=150)
    print("Saved benchmarks/results/scaling_L.png")
    plt.close()


def plot_scaling_D(df):
    """Plot latency vs model dimension for all 4 ops."""
    df_D = df[df['Scenario'] == 'Scaling_D']
    if df_D.empty:
        print("No Scaling_D data found.")
        return

    plt.figure(figsize=(12, 7))

    for op in OPS:
        col = f'{op}_ms'
        if col in df_D.columns:
            plt.plot(df_D['D'], df_D[col], marker='s',
                    color=OP_COLORS[op], label=OP_LABELS[op], linewidth=2)

    plt.xlabel('Model Dimension (D)', fontsize=12)
    plt.ylabel('Latency (ms)', fontsize=12)
    plt.title('Mamba Selective Scan Latency vs Model Dimension (L=1024)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("benchmarks/results/scaling_D.png", dpi=150)
    print("Saved benchmarks/results/scaling_D.png")
    plt.close()


def plot_op_comparison_bar(df):
    """Bar chart comparing all 4 ops at a fixed configuration."""
    # Use L=1024, D=768 from Scaling_L data
    df_L = df[df['Scenario'] == 'Scaling_L']
    if df_L.empty:
        print("No data for op comparison.")
        return

    row = df_L[df_L['L'] == 1024]
    if row.empty:
        row = df_L.iloc[-1:]  # Use largest L available

    latencies = []
    labels = []
    colors = []
    for op in OPS:
        col = f'{op}_ms'
        if col in row.columns:
            latencies.append(row[col].values[0])
            labels.append(OP_LABELS[op])
            colors.append(OP_COLORS[op])

    plt.figure(figsize=(10, 6))
    x = np.arange(len(labels))
    bars = plt.bar(x, latencies, color=colors, edgecolor='black', linewidth=0.5)

    # Add value labels on bars
    for bar, lat in zip(bars, latencies):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{lat:.2f}ms', ha='center', va='bottom', fontsize=10)

    plt.xticks(x, labels, fontsize=11)
    plt.ylabel('Latency (ms)', fontsize=12)
    plt.title(f'Op Variant Comparison (L={int(row["L"].values[0])}, D=768)', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("benchmarks/results/op_comparison.png", dpi=150)
    print("Saved benchmarks/results/op_comparison.png")
    plt.close()


def plot_overhead_analysis(df):
    """Visualize overhead from exact discretization and fusion."""
    df_L = df[df['Scenario'] == 'Scaling_L']
    if df_L.empty:
        return

    # Calculate overheads at each sequence length
    if not all(f'{op}_ms' in df_L.columns for op in OPS):
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Exact vs Linear overhead
    exact_overhead = (df_L['SelectiveScanExact_ms'] / df_L['SelectiveScan_ms'] - 1) * 100
    fused_exact_overhead = (df_L['SelectiveScanFusedExact_ms'] / df_L['SelectiveScanFused_ms'] - 1) * 100

    ax1.plot(df_L['L'], exact_overhead, marker='o', color='#2196F3', label='Non-fused', linewidth=2)
    ax1.plot(df_L['L'], fused_exact_overhead, marker='s', color='#9C27B0', label='Fused', linewidth=2)
    ax1.set_xlabel('Sequence Length (L)', fontsize=12)
    ax1.set_ylabel('Overhead (%)', fontsize=12)
    ax1.set_title('Exact Discretization Overhead', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Fusion overhead
    fusion_linear_overhead = (df_L['SelectiveScanFused_ms'] / df_L['SelectiveScan_ms'] - 1) * 100
    fusion_exact_overhead = (df_L['SelectiveScanFusedExact_ms'] / df_L['SelectiveScanExact_ms'] - 1) * 100

    ax2.plot(df_L['L'], fusion_linear_overhead, marker='o', color='#FF9800', label='Linear', linewidth=2)
    ax2.plot(df_L['L'], fusion_exact_overhead, marker='s', color='#9C27B0', label='Exact', linewidth=2)
    ax2.set_xlabel('Sequence Length (L)', fontsize=12)
    ax2.set_ylabel('Overhead (%)', fontsize=12)
    ax2.set_title('Fusion (Softplus+SiLU) Overhead', fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("benchmarks/results/overhead_analysis.png", dpi=150)
    print("Saved benchmarks/results/overhead_analysis.png")
    plt.close()


def plot_results():
    """Generate all plots from benchmark data."""
    csv_path = "benchmarks/results/benchmark_data.csv"
    if not os.path.exists(csv_path):
        print(f"No data found at {csv_path}. Run 'python benchmarks/suite.py --mode scale' first.")
        return

    os.makedirs("benchmarks/results", exist_ok=True)
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Columns: {list(df.columns)}")

    plot_scaling_L(df)
    plot_scaling_D(df)
    plot_op_comparison_bar(df)
    plot_overhead_analysis(df)

    print("\nAll plots saved to benchmarks/results/")


if __name__ == "__main__":
    plot_results()
