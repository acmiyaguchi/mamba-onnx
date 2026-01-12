"""
Benchmark suite for Mamba custom ops.

Compares all 4 op variants:
  - SelectiveScan:           6-input, linear discretization (fast)
  - SelectiveScanExact:      6-input, exact discretization (accurate)
  - SelectiveScanFused:      7-input, linear + Softplus/SiLU
  - SelectiveScanFusedExact: 7-input, exact + Softplus/SiLU
"""
import argparse
import time
import os
import torch
import torch.nn.functional as F
import onnx
import onnxruntime as ort
import numpy as np
import pandas as pd
import mamba_onnx
from mamba_onnx import register_custom_ops
import torch.nn as nn
from onnx import helper, TensorProto

# =============================================================================
# Op Definitions
# =============================================================================

OP_CONFIGS = {
    'SelectiveScan': {
        'num_inputs': 6,
        'use_fused': False,
        'use_exact': False,
        'description': '6-input, linear (fast)',
    },
    'SelectiveScanExact': {
        'num_inputs': 6,
        'use_fused': False,
        'use_exact': True,
        'description': '6-input, exact (accurate)',
    },
    'SelectiveScanFused': {
        'num_inputs': 7,
        'use_fused': True,
        'use_exact': False,
        'description': '7-input, linear + Softplus/SiLU',
    },
    'SelectiveScanFusedExact': {
        'num_inputs': 7,
        'use_fused': True,
        'use_exact': True,
        'description': '7-input, exact + Softplus/SiLU',
    },
}

# =============================================================================
# PyTorch Reference Implementation
# =============================================================================

class MambaScanReference(nn.Module):
    """PyTorch reference that can match any of the 4 op variants."""

    def __init__(self, d_model=768, d_state=16, use_fused=False, use_exact=True):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.use_fused = use_fused
        self.use_exact = use_exact

    def forward(self, u, delta, A, B, C, D, z=None):
        batch, dim, seqlen = u.shape
        dstate = self.d_state
        h = torch.zeros(batch, dim, dstate, device=u.device)
        ys = []

        for l in range(seqlen):
            u_t = u[:, :, l]
            delta_t = delta[:, :, l]

            if self.use_fused:
                delta_t = F.softplus(delta_t)

            B_t = B[:, l, :]
            C_t = C[:, l, :]

            if self.use_exact:
                A_bar = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0))
            else:
                A_bar = 1.0 + delta_t.unsqueeze(-1) * A.unsqueeze(0)

            B_bar = delta_t.unsqueeze(-1) * B_t.unsqueeze(1)
            h = A_bar * h + B_bar * u_t.unsqueeze(-1)

            C_broadcast = C_t.unsqueeze(1)
            y_t = torch.sum(C_broadcast * h, dim=-1)
            y_t = y_t + D.unsqueeze(0) * u_t

            if self.use_fused:
                y_t = y_t * F.silu(z[:, :, l])

            ys.append(y_t)

        return torch.stack(ys, dim=2)


# =============================================================================
# ONNX Model Creation
# =============================================================================

def create_onnx_model(path, op_name, D_sz, N_sz):
    """Create ONNX model for any of the 4 op variants."""
    config = OP_CONFIGS[op_name]
    use_fused = config['use_fused']

    input_infos = [
        helper.make_tensor_value_info('u', TensorProto.FLOAT, ['batch', D_sz, 'seqlen']),
        helper.make_tensor_value_info('delta', TensorProto.FLOAT, ['batch', D_sz, 'seqlen']),
        helper.make_tensor_value_info('A', TensorProto.FLOAT, [D_sz, N_sz]),
        helper.make_tensor_value_info('B', TensorProto.FLOAT, ['batch', 'seqlen', N_sz]),
        helper.make_tensor_value_info('C', TensorProto.FLOAT, ['batch', 'seqlen', N_sz]),
        helper.make_tensor_value_info('D', TensorProto.FLOAT, [D_sz]),
    ]
    inputs = ['u', 'delta', 'A', 'B', 'C', 'D']

    if use_fused:
        input_infos.append(helper.make_tensor_value_info('z', TensorProto.FLOAT, ['batch', D_sz, 'seqlen']))
        inputs.append('z')

    output_infos = [helper.make_tensor_value_info('out', TensorProto.FLOAT, ['batch', D_sz, 'seqlen'])]
    node = helper.make_node(op_name, inputs=inputs, outputs=['out'], domain='mamba')
    graph = helper.make_graph([node], f'{op_name}Benchmark', input_infos, output_infos)
    opset_imports = [helper.make_opsetid("", 17), helper.make_opsetid("mamba", 1)]
    model = helper.make_model(graph, producer_name='mamba-onnx-benchmark', opset_imports=opset_imports)
    model.ir_version = 8
    onnx.save(model, path)


# =============================================================================
# Benchmarking Engine
# =============================================================================

class BenchmarkRunner:
    def __init__(self, use_cuda=False):
        self.providers = ['CUDAExecutionProvider'] if use_cuda else ['CPUExecutionProvider']
        os.makedirs("benchmarks/results", exist_ok=True)
        os.makedirs("benchmarks/models", exist_ok=True)

    def _generate_inputs(self, B_sz, L, D_sz, N=16, use_fused=False):
        """Generate inputs for benchmarking."""
        u = torch.randn(B_sz, D_sz, L)
        A = -torch.rand(D_sz, N) - 0.5
        B = torch.randn(B_sz, L, N)
        C = torch.randn(B_sz, L, N)
        D = torch.randn(D_sz)

        if use_fused:
            # delta_raw can be any value (Softplus applied in kernel)
            delta = torch.randn(B_sz, D_sz, L) * 0.5
            z = torch.randn(B_sz, D_sz, L)
        else:
            # delta should be positive (already post-Softplus)
            delta = torch.rand(B_sz, D_sz, L) * 0.1 + 0.01
            z = None

        # Build feed dict
        feed = {
            'u': u.numpy(), 'delta': delta.numpy(), 'A': A.numpy(),
            'B': B.numpy(), 'C': C.numpy(), 'D': D.numpy()
        }
        if use_fused:
            feed['z'] = z.numpy()

        return (u, delta, A, B, C, D, z), feed

    def run_pytorch(self, inputs, use_fused, use_exact, reps=20):
        """Run PyTorch reference implementation."""
        tensors = inputs[0]
        u, delta, A, B, C, D, z = tensors

        model = MambaScanReference(u.shape[1], 16, use_fused=use_fused, use_exact=use_exact)

        # Warmup
        for _ in range(3):
            if use_fused:
                _ = model(u, delta, A, B, C, D, z)
            else:
                _ = model(u, delta, A, B, C, D)

        # Benchmark
        start = time.perf_counter()
        for _ in range(reps):
            if use_fused:
                _ = model(u, delta, A, B, C, D, z)
            else:
                _ = model(u, delta, A, B, C, D)
        end = time.perf_counter()

        return (end - start) / reps

    def run_onnx_op(self, op_name, inputs, reps=20):
        """Run a specific ONNX custom op."""
        config = OP_CONFIGS[op_name]
        tensors = inputs[0]
        feed = inputs[1]
        D, L = tensors[0].shape[1], tensors[0].shape[2]

        path = f"benchmarks/models/{op_name.lower()}.onnx"
        create_onnx_model(path, op_name, D, 16)

        sess_opt = ort.SessionOptions()
        register_custom_ops(sess_opt)
        sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(path, sess_opt, providers=self.providers)

        # Warmup
        for _ in range(3):
            sess.run(None, feed)

        # Benchmark
        start = time.perf_counter()
        for _ in range(reps):
            sess.run(None, feed)
        end = time.perf_counter()

        return (end - start) / reps

    def run_comparison(self, L=1024, D=768, reps=50):
        """Run a head-to-head comparison of all 4 ops."""
        print("\n" + "=" * 70)
        print(f"Comparing All 4 Ops (L={L}, D={D}, reps={reps})")
        print("=" * 70)

        results = []

        # Test non-fused ops (6-input)
        inputs_6, feed_6 = self._generate_inputs(1, L, D, use_fused=False)

        # Test fused ops (7-input)
        inputs_7, feed_7 = self._generate_inputs(1, L, D, use_fused=True)

        for op_name, config in OP_CONFIGS.items():
            use_fused = config['use_fused']
            use_exact = config['use_exact']

            inputs = (inputs_7, feed_7) if use_fused else (inputs_6, feed_6)

            # PyTorch reference
            lat_pt = self.run_pytorch(inputs, use_fused, use_exact, reps)

            # ONNX custom op
            lat_onnx = self.run_onnx_op(op_name, inputs, reps)

            speedup = lat_pt / lat_onnx if lat_onnx > 0 else 0

            results.append({
                'Op': op_name,
                'PyTorch (ms)': lat_pt * 1000,
                'ONNX (ms)': lat_onnx * 1000,
                'Speedup': speedup,
            })

            print(f"{op_name:25s}: PyTorch={lat_pt*1000:7.3f}ms  ONNX={lat_onnx*1000:7.3f}ms  Speedup={speedup:.2f}x")

        # Summary table
        print("\n" + "-" * 70)
        print("Op Comparison Summary:")
        print("-" * 70)

        # Find fastest
        fastest = min(results, key=lambda x: x['ONNX (ms)'])
        print(f"Fastest ONNX op: {fastest['Op']} ({fastest['ONNX (ms)']:.3f}ms)")

        # Compare fused vs non-fused (same discretization)
        ss = next(r for r in results if r['Op'] == 'SelectiveScan')
        ssf = next(r for r in results if r['Op'] == 'SelectiveScanFused')
        sse = next(r for r in results if r['Op'] == 'SelectiveScanExact')
        ssfe = next(r for r in results if r['Op'] == 'SelectiveScanFusedExact')

        print(f"\nFusion overhead (linear):  {ssf['ONNX (ms)'] - ss['ONNX (ms)']:+.3f}ms ({ssf['ONNX (ms)']/ss['ONNX (ms)']:.2f}x)")
        print(f"Fusion overhead (exact):   {ssfe['ONNX (ms)'] - sse['ONNX (ms)']:+.3f}ms ({ssfe['ONNX (ms)']/sse['ONNX (ms)']:.2f}x)")
        print(f"Exact overhead (non-fused): {sse['ONNX (ms)'] - ss['ONNX (ms)']:+.3f}ms ({sse['ONNX (ms)']/ss['ONNX (ms)']:.2f}x)")
        print(f"Exact overhead (fused):    {ssfe['ONNX (ms)'] - ssf['ONNX (ms)']:+.3f}ms ({ssfe['ONNX (ms)']/ssf['ONNX (ms)']:.2f}x)")

        return results

    def run_scaling_suite(self, reps=30):
        """Run scaling benchmarks across L and D."""
        all_results = []

        # Scenario 1: Scaling L (D=768)
        print("\n" + "=" * 70)
        print("Scaling Sequence Length (L) - D=768")
        print("=" * 70)
        D = 768

        for L in [128, 256, 512, 1024, 2048, 4096]:
            print(f"\nL={L}:")

            inputs_6, feed_6 = self._generate_inputs(1, L, D, use_fused=False)
            inputs_7, feed_7 = self._generate_inputs(1, L, D, use_fused=True)

            row = {'Scenario': 'Scaling_L', 'L': L, 'D': D}

            for op_name, config in OP_CONFIGS.items():
                use_fused = config['use_fused']
                inputs = (inputs_7, feed_7) if use_fused else (inputs_6, feed_6)

                lat = self.run_onnx_op(op_name, inputs, reps)
                row[f'{op_name}_ms'] = lat * 1000
                print(f"  {op_name:25s}: {lat*1000:7.3f}ms")

            all_results.append(row)

        # Scenario 2: Scaling D (L=1024)
        print("\n" + "=" * 70)
        print("Scaling Model Dimension (D) - L=1024")
        print("=" * 70)
        L = 1024

        for D in [256, 512, 768, 1024, 1536, 2048]:
            print(f"\nD={D}:")

            inputs_6, feed_6 = self._generate_inputs(1, L, D, use_fused=False)
            inputs_7, feed_7 = self._generate_inputs(1, L, D, use_fused=True)

            row = {'Scenario': 'Scaling_D', 'L': L, 'D': D}

            for op_name, config in OP_CONFIGS.items():
                use_fused = config['use_fused']
                inputs = (inputs_7, feed_7) if use_fused else (inputs_6, feed_6)

                lat = self.run_onnx_op(op_name, inputs, reps)
                row[f'{op_name}_ms'] = lat * 1000
                print(f"  {op_name:25s}: {lat*1000:7.3f}ms")

            all_results.append(row)

        # Save results
        df = pd.DataFrame(all_results)
        df.to_csv("benchmarks/results/benchmark_data.csv", index=False)
        print("\n" + "=" * 70)
        print("Results saved to benchmarks/results/benchmark_data.csv")

        return all_results


def main():
    parser = argparse.ArgumentParser(description='Mamba Custom Ops Benchmark Suite')
    parser.add_argument('--mode', choices=['compare', 'scale', 'all'], default='compare',
                        help='Benchmark mode: compare (head-to-head), scale (L/D scaling), all (both)')
    parser.add_argument('--L', type=int, default=1024, help='Sequence length for comparison')
    parser.add_argument('--D', type=int, default=768, help='Model dimension for comparison')
    parser.add_argument('--reps', type=int, default=50, help='Number of repetitions')
    args = parser.parse_args()

    runner = BenchmarkRunner()

    if args.mode in ('compare', 'all'):
        runner.run_comparison(L=args.L, D=args.D, reps=args.reps)

    if args.mode in ('scale', 'all'):
        runner.run_scaling_suite(reps=args.reps)


if __name__ == "__main__":
    main()
