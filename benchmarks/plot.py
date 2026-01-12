import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_results():
    if not os.path.exists("benchmarks/results/benchmark_data.csv"):
        print("No data found. Run suite.py first.")
        return

    df = pd.read_csv("benchmarks/results/benchmark_data.csv")
    
    # Plot 1: Scaling L (Latency)
    df_L = df[df['Scenario'] == 'Scaling_L']
    if not df_L.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(df_L['L'], df_L['PyTorch_Latency'] * 1000, marker='o', label='PyTorch Eager')
        plt.plot(df_L['L'], df_L['Fused_Latency'] * 1000, marker='s', label='Fused ONNX (Optimized)')
        
        # Baseline only exists for small L
        df_bl = df_L.dropna(subset=['Baseline_Latency'])
        if not df_bl.empty:
            plt.plot(df_bl['L'], df_bl['Baseline_Latency'] * 1000, marker='x', linestyle='--', label='Vanilla ONNX')

        plt.xlabel('Sequence Length (L)')
        plt.ylabel('Latency (ms) - Log Scale')
        plt.yscale('log')
        plt.title('Mamba Scan Latency vs Sequence Length (D=768)')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend()
        plt.savefig("benchmarks/results/scaling_L.png")
        print("Saved benchmarks/results/scaling_L.png")
        plt.close()

    # Plot 2: Speedup Factor (L=1024, D scaling)
    df_D = df[df['Scenario'] == 'Scaling_D']
    if not df_D.empty:
        df_D['Speedup'] = df_D['PyTorch_Latency'] / df_D['Fused_Latency']
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(df_D['D'].astype(str), df_D['Speedup'], color='#4CAF50')
        
        plt.xlabel('Model Dimension (D)')
        plt.ylabel('Speedup Factor (x)')
        plt.title('Fused Operator Speedup vs PyTorch (L=1024)')
        plt.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}x',
                     ha='center', va='bottom')
                     
        plt.savefig("benchmarks/results/speedup_D.png")
        print("Saved benchmarks/results/speedup_D.png")
        plt.close()

if __name__ == "__main__":
    plot_results()
