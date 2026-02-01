"""
Profile time distribution across MambaBlock components.

Shows where time is spent in a single Mamba block so you can estimate
the impact of replacing the selective-scan step with a custom ONNX op.
"""
import argparse
import os
import tempfile
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MambaBlockProfiler:
    """Profile individual components of a MambaBlock."""

    def __init__(self, d_model=768, d_state=16, d_conv=4, expand=2):
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand
        self.dt_rank = d_model // 16  # "auto" setting

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, groups=self.d_inner,
            padding=d_conv - 1, bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.A = -torch.rand(self.d_inner, d_state) - 0.5
        self.D = torch.randn(self.d_inner)

    def profile(self, batch=1, seqlen=512, reps=50, warmup=10):
        hidden_states = torch.randn(batch, seqlen, self.d_model)

        timings = {
            "in_proj": [],
            "conv1d_silu": [],
            "x_proj": [],
            "dt_proj": [],
            "selective_scan": [],
            "z_gate": [],
            "out_proj": [],
        }

        with torch.no_grad():
            for _ in range(warmup):
                self._forward_profiled(hidden_states, timings, record=False)
            for _ in range(reps):
                self._forward_profiled(hidden_states, timings, record=True)

        results = {}
        total = 0
        for name, times in timings.items():
            avg = sum(times) / len(times) * 1000  # ms
            results[name] = avg
            total += avg
        results["total"] = total
        return results

    def _forward_profiled(self, hidden_states, timings, record=True):
        batch, seqlen, dim = hidden_states.shape

        # 1. in_proj
        t0 = time.perf_counter()
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l", l=seqlen,
        )
        t1 = time.perf_counter()
        if record:
            timings["in_proj"].append(t1 - t0)

        x, z = xz.chunk(2, dim=1)

        # 2. conv1d + SiLU
        t0 = time.perf_counter()
        x = F.silu(self.conv1d(x)[..., :seqlen])
        t1 = time.perf_counter()
        if record:
            timings["conv1d_silu"].append(t1 - t0)

        # 3. x_proj
        t0 = time.perf_counter()
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        t1 = time.perf_counter()
        if record:
            timings["x_proj"].append(t1 - t0)

        # 4. dt_proj
        t0 = time.perf_counter()
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        dt = F.softplus(dt + self.dt_proj.bias.unsqueeze(0).unsqueeze(-1))
        t1 = time.perf_counter()
        if record:
            timings["dt_proj"].append(t1 - t0)

        B = rearrange(B, "(b l) dstate -> b l dstate", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b l dstate", l=seqlen).contiguous()

        # 5. selective scan (Python reference)
        t0 = time.perf_counter()
        y = self._selective_scan_ref(x, dt, self.A, B, C, self.D)
        t1 = time.perf_counter()
        if record:
            timings["selective_scan"].append(t1 - t0)

        # 6. z-gate
        t0 = time.perf_counter()
        y = y * F.silu(z)
        t1 = time.perf_counter()
        if record:
            timings["z_gate"].append(t1 - t0)

        # 7. out_proj
        t0 = time.perf_counter()
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        t1 = time.perf_counter()
        if record:
            timings["out_proj"].append(t1 - t0)

        return out

    def _selective_scan_ref(self, u, delta, A, B, C, D):
        batch, dim, seqlen = u.shape
        dstate = A.shape[1]
        h = torch.zeros(batch, dim, dstate)
        ys = []
        for l in range(seqlen):
            u_t = u[:, :, l]
            delta_t = delta[:, :, l]
            B_t = B[:, l, :]
            C_t = C[:, l, :]
            A_bar = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0))
            B_bar = delta_t.unsqueeze(-1) * B_t.unsqueeze(1)
            h = A_bar * h + B_bar * u_t.unsqueeze(-1)
            y_t = torch.sum(C_t.unsqueeze(1) * h, dim=-1) + D.unsqueeze(0) * u_t
            ys.append(y_t)
        return torch.stack(ys, dim=2)


def _try_bench_custom_kernel(D, L, backend, reps, threads=0):
    """Optionally benchmark the ONNX custom kernel and return latency in ms, or None."""
    try:
        import onnxruntime as ort
        import numpy as np
        import onnx
        from onnx import TensorProto, helper
        import mamba_onnx
    except ImportError:
        return None

    if backend not in mamba_onnx.available_backends():
        return None

    N = 16
    with tempfile.TemporaryDirectory() as tmpdir:
        # Build a FusedExact model (closest to what the profiler's scan does)
        input_infos = [
            helper.make_tensor_value_info("u", TensorProto.FLOAT, [1, D, L]),
            helper.make_tensor_value_info("delta", TensorProto.FLOAT, [1, D, L]),
            helper.make_tensor_value_info("A", TensorProto.FLOAT, [D, N]),
            helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, L, N]),
            helper.make_tensor_value_info("C", TensorProto.FLOAT, [1, L, N]),
            helper.make_tensor_value_info("D", TensorProto.FLOAT, [D]),
            helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, D, L]),
        ]
        output_infos = [
            helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, D, L])
        ]
        node = helper.make_node(
            "SelectiveScanFusedExact",
            inputs=["u", "delta", "A", "B", "C", "D", "z"],
            outputs=["out"],
            domain="mamba",
        )
        graph = helper.make_graph([node], "ProfileBench", input_infos, output_infos)
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("mamba", 1)],
        )
        model.ir_version = 8
        path = os.path.join(tmpdir, "profile_bench.onnx")
        onnx.save(model, path)

        opts = ort.SessionOptions()
        if threads > 0:
            opts.intra_op_num_threads = threads
        mamba_onnx.register_custom_ops(opts, backend=backend)
        sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])

        np.random.seed(0)
        feed = {
            "u": np.random.randn(1, D, L).astype(np.float32),
            "delta": np.random.randn(1, D, L).astype(np.float32) * 0.5,
            "A": (-np.random.rand(D, N).astype(np.float32) - 0.5),
            "B": np.random.randn(1, L, N).astype(np.float32) * 0.1,
            "C": np.random.randn(1, L, N).astype(np.float32) * 0.1,
            "D": np.random.randn(D).astype(np.float32) * 0.1,
            "z": np.random.randn(1, D, L).astype(np.float32),
        }

        for _ in range(5):
            sess.run(None, feed)

        start = time.perf_counter()
        for _ in range(reps):
            sess.run(None, feed)
        elapsed = (time.perf_counter() - start) / reps * 1000
        return elapsed


def main():
    parser = argparse.ArgumentParser(description="Profile MambaBlock components")
    parser.add_argument("--D", type=int, default=768, help="Model dimension (d_model)")
    parser.add_argument("--L", type=int, default=512, help="Sequence length")
    parser.add_argument("--backend", default="zig", help="Backend for custom kernel comparison")
    parser.add_argument("--reps", type=int, default=50, help="Timing iterations")
    parser.add_argument("--threads", type=int, default=0, help="Number of intra-op threads (0=default)")
    args = parser.parse_args()

    # Pin kernel thread counts for both backends (OpenMP + Zig thread pool)
    if args.threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["MAMBA_THREADS"] = str(args.threads)

    d_inner = args.D * 2  # expand=2

    print("=" * 70)
    print(f"MambaBlock Component Profiling (d_model={args.D}, L={args.L})")
    print("=" * 70)

    profiler = MambaBlockProfiler(d_model=args.D, d_state=16)
    results = profiler.profile(batch=1, seqlen=args.L, reps=args.reps)

    print(f"\n{'Component':<20} {'Time (ms)':<12} {'% of Total':<12}")
    print("-" * 44)

    total = results["total"]
    for name, time_ms in results.items():
        if name != "total":
            pct = time_ms / total * 100
            print(f"{name:<20} {time_ms:<12.3f} {pct:<12.1f}%")

    print("-" * 44)
    print(f"{'TOTAL':<20} {total:<12.3f}")

    # Analysis
    print("\n" + "=" * 70)
    print("Analysis")
    print("=" * 70)

    scan_pct = results["selective_scan"] / total * 100
    gemm_time = results["in_proj"] + results["x_proj"] + results["dt_proj"] + results["out_proj"]
    gemm_pct = gemm_time / total * 100

    print(f"\nSelective scan: {scan_pct:.1f}% of total time")
    print(f"Linear layers (GEMMs): {gemm_pct:.1f}% of total time")

    # Try actual custom kernel benchmark
    kernel_ms = _try_bench_custom_kernel(d_inner, args.L, args.backend, args.reps, threads=args.threads)
    if kernel_ms is not None:
        print(f"\n" + "-" * 44)
        print(f"Measured ONNX SelectiveScanFusedExact ({args.backend}): {kernel_ms:.3f} ms")
        new_total = total - results["selective_scan"] + kernel_ms
        overall_speedup = total / new_total
        print(f"  Projected total: {new_total:.3f} ms (was {total:.3f} ms)")
        print(f"  Projected block speedup: {overall_speedup:.2f}x")
    else:
        print(f"\nCustom kernel ({args.backend}) not available; skipping speedup projection.")


if __name__ == "__main__":
    main()
