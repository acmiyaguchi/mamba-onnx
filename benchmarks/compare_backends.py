#!/usr/bin/env python3
"""Compare C++ and Zig backends for correctness and performance."""

import time
import tempfile
import numpy as np
import onnxruntime as ort
from onnx import helper, TensorProto
import onnx
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mamba_onnx


def create_model(op_name: str, D: int, N: int, use_fused: bool) -> str:
    """Create ONNX model for testing."""
    inputs = [
        helper.make_tensor_value_info('u', TensorProto.FLOAT, ['B', D, 'L']),
        helper.make_tensor_value_info('delta', TensorProto.FLOAT, ['B', D, 'L']),
        helper.make_tensor_value_info('A', TensorProto.FLOAT, [D, N]),
        helper.make_tensor_value_info('B', TensorProto.FLOAT, ['B', 'L', N]),
        helper.make_tensor_value_info('C', TensorProto.FLOAT, ['B', 'L', N]),
        helper.make_tensor_value_info('D', TensorProto.FLOAT, [D]),
    ]
    input_names = ['u', 'delta', 'A', 'B', 'C', 'D']

    if use_fused:
        inputs.append(helper.make_tensor_value_info('z', TensorProto.FLOAT, ['B', D, 'L']))
        input_names.append('z')

    outputs = [helper.make_tensor_value_info('out', TensorProto.FLOAT, ['B', D, 'L'])]
    node = helper.make_node(op_name, inputs=input_names, outputs=['out'], domain='mamba')
    graph = helper.make_graph([node], f'{op_name}', inputs, outputs)
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("mamba", 1)]
    )
    model.ir_version = 8

    path = os.path.join(tempfile.gettempdir(), f"{op_name.lower()}_bench.onnx")
    onnx.save(model, path)
    return path


def make_inputs(B: int, D: int, L: int, N: int, use_fused: bool) -> dict:
    """Generate random inputs."""
    np.random.seed(42)
    inputs = {
        'u': np.random.randn(B, D, L).astype(np.float32),
        'delta': np.abs(np.random.randn(B, D, L).astype(np.float32)) * 0.1 + 0.01,
        'A': (-np.random.rand(D, N).astype(np.float32) - 0.5),
        'B': np.random.randn(B, L, N).astype(np.float32) * 0.1,
        'C': np.random.randn(B, L, N).astype(np.float32) * 0.1,
        'D': np.random.randn(D).astype(np.float32) * 0.1,
    }
    if use_fused:
        inputs['z'] = np.random.randn(B, D, L).astype(np.float32)
    return inputs


def benchmark_backend(backend: str, model_path: str, inputs: dict, warmup: int = 5, runs: int = 50) -> tuple:
    """Benchmark a backend, return (output, mean_ms, std_ms)."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    lib_path = mamba_onnx.get_library_path(backend)
    opts.register_custom_ops_library(lib_path)

    sess = ort.InferenceSession(model_path, opts, providers=['CPUExecutionProvider'])

    # Warmup
    for _ in range(warmup):
        sess.run(None, inputs)

    # Timed runs
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        out = sess.run(None, inputs)[0]
        times.append(time.perf_counter() - start)

    return out, np.mean(times) * 1000, np.std(times) * 1000


def compare_outputs(out_cpp: np.ndarray, out_zig: np.ndarray, op_name: str) -> bool:
    """Compare outputs for correctness."""
    diff = np.abs(out_cpp - out_zig)
    max_diff = diff.max()
    mean_diff = diff.mean()

    print(f"  Max abs diff:  {max_diff:.2e}")
    print(f"  Mean abs diff: {mean_diff:.2e}")

    # Use numpy's allclose with reasonable tolerances
    # rtol=1e-4 means 0.01% relative tolerance
    # atol=1e-5 means absolute tolerance for values near zero
    match = np.allclose(out_cpp, out_zig, rtol=1e-4, atol=1e-4)

    print(f"  Result: {'MATCH' if match else 'MISMATCH!'}")
    return match


def main():
    print("=" * 70)
    print("Backend Comparison: C++ vs Zig")
    print("=" * 70)

    backends = mamba_onnx.available_backends()
    print(f"Available backends: {backends}")

    if 'cpp' not in backends or 'zig' not in backends:
        print("ERROR: Need both cpp and zig backends. Build with: zig build both -Doptimize=ReleaseFast")
        return 1

    # Test configurations
    configs = [
        {"B": 1, "D": 768, "L": 512, "N": 16},
        {"B": 1, "D": 768, "L": 2048, "N": 16},
        {"B": 4, "D": 768, "L": 512, "N": 16},
    ]

    ops = [
        ("SelectiveScan", False),
        ("SelectiveScanExact", False),
        ("SelectiveScanFused", True),
        ("SelectiveScanFusedExact", True),
    ]

    all_match = True
    results = []

    for op_name, use_fused in ops:
        print(f"\n{'='*70}")
        print(f"Op: {op_name}")
        print("=" * 70)

        model_path = create_model(op_name, 768, 16, use_fused)

        for cfg in configs:
            print(f"\n  Config: B={cfg['B']}, D={cfg['D']}, L={cfg['L']}")
            inputs = make_inputs(**cfg, use_fused=use_fused)

            # Run both backends
            out_cpp, time_cpp, std_cpp = benchmark_backend("cpp", model_path, inputs)
            out_zig, time_zig, std_zig = benchmark_backend("zig", model_path, inputs)

            # Compare correctness
            match = compare_outputs(out_cpp, out_zig, op_name)
            all_match = all_match and match

            # Performance
            speedup = time_cpp / time_zig
            print(f"  C++ time:  {time_cpp:.3f} +/- {std_cpp:.3f} ms")
            print(f"  Zig time:  {time_zig:.3f} +/- {std_zig:.3f} ms")
            print(f"  Speedup:   {speedup:.2f}x {'(Zig faster)' if speedup > 1 else '(C++ faster)'}")

            results.append({
                "op": op_name,
                "config": f"B={cfg['B']},L={cfg['L']}",
                "cpp_ms": time_cpp,
                "zig_ms": time_zig,
                "speedup": speedup,
                "match": match,
            })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Op':<25} {'Config':<15} {'C++ (ms)':<12} {'Zig (ms)':<12} {'Speedup':<10} {'Match'}")
    print("-" * 85)
    for r in results:
        match_str = "YES" if r["match"] else "NO"
        print(f"{r['op']:<25} {r['config']:<15} {r['cpp_ms']:<12.3f} {r['zig_ms']:<12.3f} {r['speedup']:<10.2f} {match_str}")

    print("\n" + "=" * 70)
    if all_match:
        print("All outputs MATCH between C++ and Zig backends!")
    else:
        print("WARNING: Some outputs DO NOT MATCH!")
    print("=" * 70)

    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())
