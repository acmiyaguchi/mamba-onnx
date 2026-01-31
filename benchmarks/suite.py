"""
Consolidated benchmark suite for Mamba selective scan custom ops.

Modes:
  compare  -- All 4 ops x both backends + PyTorch reference at one config.
  scale    -- L and D sweeps for all 4 ops (supports multiple backends).
  all      -- Both modes.
"""
import argparse
import os
import tempfile
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from onnx import TensorProto, helper

import mamba_onnx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N = 16  # d_state – hardcoded in the kernels (two Vec8 chunks)

ALL_OPS = [
    ("SelectiveScan", False, False),
    ("SelectiveScanExact", False, True),
    ("SelectiveScanFused", True, False),
    ("SelectiveScanFusedExact", True, True),
]

OP_LABELS = {
    "SelectiveScan": "Linear (fastest)",
    "SelectiveScanExact": "Exact",
    "SelectiveScanFused": "Linear + Fused",
    "SelectiveScanFusedExact": "Exact + Fused",
}

SCALE_Ls = [128, 256, 512, 1024, 2048, 4096]
SCALE_Ds = [256, 512, 768, 1024, 1536, 2048]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_inputs(B, D, L, use_fused):
    """Return (torch tensors tuple, numpy feed dict)."""
    torch.manual_seed(42)
    u = torch.randn(B, D, L)
    A = -torch.rand(D, N) - 0.5
    B_in = torch.randn(B, L, N) * 0.1
    C = torch.randn(B, L, N) * 0.1
    D_in = torch.randn(D) * 0.1

    if use_fused:
        delta = torch.randn(B, D, L) * 0.5
        z = torch.randn(B, D, L)
    else:
        delta = torch.rand(B, D, L) * 0.1 + 0.01
        z = None

    feed = {
        "u": u.numpy(), "delta": delta.numpy(), "A": A.numpy(),
        "B": B_in.numpy(), "C": C.numpy(), "D": D_in.numpy(),
    }
    if z is not None:
        feed["z"] = z.numpy()

    return (u, delta, A, B_in, C, D_in, z), feed


def _create_model(tmpdir, op_name, D_sz, N_sz, use_fused):
    """Create a minimal ONNX model in *tmpdir* and return its path."""
    input_infos = [
        helper.make_tensor_value_info("u", TensorProto.FLOAT, ["batch", D_sz, "seqlen"]),
        helper.make_tensor_value_info("delta", TensorProto.FLOAT, ["batch", D_sz, "seqlen"]),
        helper.make_tensor_value_info("A", TensorProto.FLOAT, [D_sz, N_sz]),
        helper.make_tensor_value_info("B", TensorProto.FLOAT, ["batch", "seqlen", N_sz]),
        helper.make_tensor_value_info("C", TensorProto.FLOAT, ["batch", "seqlen", N_sz]),
        helper.make_tensor_value_info("D", TensorProto.FLOAT, [D_sz]),
    ]
    inputs = ["u", "delta", "A", "B", "C", "D"]

    if use_fused:
        input_infos.append(
            helper.make_tensor_value_info("z", TensorProto.FLOAT, ["batch", D_sz, "seqlen"])
        )
        inputs.append("z")

    output_infos = [
        helper.make_tensor_value_info("out", TensorProto.FLOAT, ["batch", D_sz, "seqlen"])
    ]
    node = helper.make_node(op_name, inputs=inputs, outputs=["out"], domain="mamba")
    graph = helper.make_graph([node], f"{op_name}Bench", input_infos, output_infos)
    opset_imports = [helper.make_opsetid("", 17), helper.make_opsetid("mamba", 1)]
    model = helper.make_model(graph, producer_name="mamba-bench", opset_imports=opset_imports)
    model.ir_version = 8
    path = os.path.join(tmpdir, f"{op_name.lower()}.onnx")
    onnx.save(model, path)
    return path


def _bench_onnx(op_name, feed, backend, D, use_fused, reps, tmpdir, threads=0):
    """Create session, warm up, time, return mean latency in seconds."""
    import gc
    path = _create_model(tmpdir, op_name, D, N, use_fused)
    opts = ort.SessionOptions()
    if threads > 0:
        opts.intra_op_num_threads = threads
    mamba_onnx.register_custom_ops(opts, backend=backend)
    sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])

    for _ in range(10):
        sess.run(None, feed)

    start = time.perf_counter()
    for _ in range(reps):
        sess.run(None, feed)
    elapsed = (time.perf_counter() - start) / reps
    del sess
    gc.collect()
    return elapsed


def _reference_selective_scan(u, delta, A, B, C, D, z=None, use_fused=False, use_exact=False):
    """PyTorch reference (duplicated from tests to keep benchmarks self-contained)."""
    batch, dim, seqlen = u.shape
    dstate = A.shape[1]
    h = torch.zeros(batch, dim, dstate, device=u.device, dtype=u.dtype)
    ys = []

    for l in range(seqlen):
        u_t = u[:, :, l]
        delta_t = delta[:, :, l]

        if use_fused:
            delta_t = F.softplus(delta_t)

        B_t = B[:, l, :]
        C_t = C[:, l, :]

        if use_exact:
            A_bar = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0))
        else:
            A_bar = 1.0 + delta_t.unsqueeze(-1) * A.unsqueeze(0)

        B_bar = delta_t.unsqueeze(-1) * B_t.unsqueeze(1)
        h = A_bar * h + B_bar * u_t.unsqueeze(-1)

        y_t = torch.sum(C_t.unsqueeze(1) * h, dim=-1)
        y_t = y_t + D.unsqueeze(0) * u_t

        if use_fused:
            y_t = y_t * F.silu(z[:, :, l])

        ys.append(y_t)

    return torch.stack(ys, dim=2)


def _bench_pytorch(tensors, use_fused, use_exact, reps):
    """Run PyTorch reference, return mean latency in seconds."""
    u, delta, A, B_in, C, D_in, z = tensors

    with torch.no_grad():
        for _ in range(3):
            _reference_selective_scan(u, delta, A, B_in, C, D_in, z, use_fused, use_exact)

        start = time.perf_counter()
        for _ in range(reps):
            _reference_selective_scan(u, delta, A, B_in, C, D_in, z, use_fused, use_exact)
    return (time.perf_counter() - start) / reps


# ---------------------------------------------------------------------------
# Compare mode
# ---------------------------------------------------------------------------

def _run_compare(B, D, L, reps, backends, threads=0):
    """All 4 ops x backends + PyTorch reference."""
    print(f"\n{'=' * 78}")
    print(f"Compare mode  (B={B}, D={D}, L={L}, reps={reps})")
    print(f"{'=' * 78}")

    rows = []

    for op_name, use_fused, use_exact in ALL_OPS:
        tensors, feed = _make_inputs(B, D, L, use_fused)

        pt_ms = _bench_pytorch(tensors, use_fused, use_exact, reps) * 1000

        for backend in backends:
            if backend not in mamba_onnx.available_backends():
                continue
            with tempfile.TemporaryDirectory() as tmpdir:
                onnx_ms = _bench_onnx(op_name, feed, backend, D, use_fused, reps, tmpdir, threads=threads) * 1000

            speedup = pt_ms / onnx_ms if onnx_ms > 0 else 0
            rows.append({
                "op": op_name,
                "backend": backend,
                "pytorch_ms": round(pt_ms, 3),
                "onnx_ms": round(onnx_ms, 3),
                "speedup": round(speedup, 2),
            })

    # Print table
    hdr = f"{'Op':<28} {'Backend':<8} {'PyTorch ms':>11} {'ONNX ms':>9} {'Speedup':>8}"
    print(f"\n{hdr}")
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['op']:<28} {r['backend']:<8} {r['pytorch_ms']:>11.3f} {r['onnx_ms']:>9.3f} {r['speedup']:>7.2f}x")

    # Cpp vs zig ratio (if both present)
    cpp_rows = [r for r in rows if r["backend"] == "cpp"]
    zig_rows = [r for r in rows if r["backend"] == "zig"]
    if cpp_rows and zig_rows:
        print(f"\n{'Op':<28} {'cpp/zig ratio':>14}")
        print("-" * 44)
        for cr, zr in zip(cpp_rows, zig_rows):
            ratio = cr["onnx_ms"] / zr["onnx_ms"] if zr["onnx_ms"] > 0 else 0
            print(f"{cr['op']:<28} {ratio:>14.2f}")

    return rows


# ---------------------------------------------------------------------------
# Scale mode
# ---------------------------------------------------------------------------

def _run_scale(reps, backends, threads=0):
    """L and D sweeps for all 4 ops."""
    all_rows = []

    for backend in backends:
        if backend not in mamba_onnx.available_backends():
            print(f"Backend '{backend}' not available. Skipping scale mode.")
            continue

        rows = []

        # L sweep (D=768)
        D = 768
        print(f"\n{'=' * 78}")
        print(f"Scale L  (D={D}, backend={backend}, reps={reps})")
        print(f"{'=' * 78}")
        for L in SCALE_Ls:
            for op_name, use_fused, _ in ALL_OPS:
                _, feed = _make_inputs(1, D, L, use_fused)
                with tempfile.TemporaryDirectory() as tmpdir:
                    lat = _bench_onnx(op_name, feed, backend, D, use_fused, reps, tmpdir, threads=threads) * 1000
                rows.append({"scenario": "Scaling_L", "L": L, "D": D,
                              "backend": backend, "op": op_name, "latency_ms": round(lat, 3)})
                print(f"  L={L:<5d} {op_name:<28} {lat:>8.3f} ms")

        # D sweep (L=1024)
        L = 1024
        print(f"\n{'=' * 78}")
        print(f"Scale D  (L={L}, backend={backend}, reps={reps})")
        print(f"{'=' * 78}")
        for D in SCALE_Ds:
            for op_name, use_fused, _ in ALL_OPS:
                _, feed = _make_inputs(1, D, L, use_fused)
                with tempfile.TemporaryDirectory() as tmpdir:
                    lat = _bench_onnx(op_name, feed, backend, D, use_fused, reps, tmpdir, threads=threads) * 1000
                rows.append({"scenario": "Scaling_D", "L": L, "D": D,
                              "backend": backend, "op": op_name, "latency_ms": round(lat, 3)})
                print(f"  D={D:<5d} {op_name:<28} {lat:>8.3f} ms")

        all_rows.extend(rows)

    return all_rows


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _save_csv(rows, path, mode_label):
    """Write rows (list of dicts) to CSV."""
    if not rows:
        return
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\n{mode_label} results saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Mamba custom-ops benchmark suite")
    parser.add_argument("--mode", choices=["compare", "scale", "all"], default="compare")
    parser.add_argument("--backend", choices=["cpp", "zig", "both"], default=None,
                        help="Backend (default: both for compare, zig for scale)")
    parser.add_argument("--B", type=int, default=1, help="Batch size for compare mode")
    parser.add_argument("--L", type=int, default=1024, help="Sequence length for compare mode")
    parser.add_argument("--D", type=int, default=768, help="Model dimension for compare mode")
    parser.add_argument("--reps", type=int, default=50, help="Timing iterations")
    parser.add_argument("--threads", type=int, default=0, help="Number of intra-op threads (0 = default)")
    parser.add_argument("--output", type=str, default="benchmarks/results/benchmark_data.csv",
                        help="CSV output path")
    args = parser.parse_args()

    # Pin kernel thread counts for both backends (OpenMP + Zig thread pool)
    if args.threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["MAMBA_THREADS"] = str(args.threads)

    avail = mamba_onnx.available_backends()
    print(f"Available backends: {avail}")
    if args.threads > 0:
        print(f"Kernel threads pinned to {args.threads}")

    if args.mode in ("compare", "all"):
        backends = (["cpp", "zig"] if (args.backend is None or args.backend == "both")
                    else [args.backend])
        compare_rows = _run_compare(args.B, args.D, args.L, args.reps, backends, threads=args.threads)
        if args.mode == "compare":
            _save_csv(compare_rows, args.output, "Compare")
        else:
            stem, ext = os.path.splitext(args.output)
            _save_csv(compare_rows, f"{stem}_compare{ext}", "Compare")

    if args.mode in ("scale", "all"):
        backends = (["cpp", "zig"] if (args.backend is None or args.backend == "both")
                    else [args.backend])
        scale_rows = _run_scale(args.reps, backends, threads=args.threads)
        if args.mode == "scale":
            _save_csv(scale_rows, args.output, "Scale")
        else:
            stem, ext = os.path.splitext(args.output)
            _save_csv(scale_rows, f"{stem}_scale{ext}", "Scale")


if __name__ == "__main__":
    main()
