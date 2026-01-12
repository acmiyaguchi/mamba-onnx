"""
Profile time distribution across MambaBlock components.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from einops import rearrange

class MambaBlockProfiler:
    """Profile individual components of a MambaBlock."""

    def __init__(self, d_model=768, d_state=16, d_conv=4, expand=2):
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand
        self.dt_rank = d_model // 16  # "auto" setting

        # Create the layers
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, groups=self.d_inner,
            padding=d_conv - 1, bias=True
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # SSM params
        self.A = -torch.rand(self.d_inner, d_state) - 0.5
        self.D = torch.randn(self.d_inner)

    def profile(self, batch=1, seqlen=512, reps=50, warmup=10):
        """Run profiling and return time breakdown."""

        # Input
        hidden_states = torch.randn(batch, seqlen, self.d_model)

        timings = {
            'in_proj': [],
            'conv1d_silu': [],
            'x_proj': [],
            'dt_proj': [],
            'selective_scan': [],
            'z_gate': [],
            'out_proj': [],
        }

        with torch.no_grad():
            # Warmup
            for _ in range(warmup):
                self._forward_profiled(hidden_states, timings, record=False)

            # Benchmark
            for _ in range(reps):
                self._forward_profiled(hidden_states, timings, record=True)

        # Compute averages
        results = {}
        total = 0
        for name, times in timings.items():
            avg = sum(times) / len(times) * 1000  # ms
            results[name] = avg
            total += avg
        results['total'] = total

        return results

    def _forward_profiled(self, hidden_states, timings, record=True):
        batch, seqlen, dim = hidden_states.shape

        # 1. in_proj
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l", l=seqlen
        )
        t1 = time.perf_counter()
        if record: timings['in_proj'].append(t1 - t0)

        # Split
        x, z = xz.chunk(2, dim=1)

        # 2. conv1d + SiLU
        t0 = time.perf_counter()
        x = F.silu(self.conv1d(x)[..., :seqlen])
        t1 = time.perf_counter()
        if record: timings['conv1d_silu'].append(t1 - t0)

        # 3. x_proj (get dt, B, C)
        t0 = time.perf_counter()
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        t1 = time.perf_counter()
        if record: timings['x_proj'].append(t1 - t0)

        # 4. dt_proj
        t0 = time.perf_counter()
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        dt = F.softplus(dt + self.dt_proj.bias.unsqueeze(0).unsqueeze(-1))
        t1 = time.perf_counter()
        if record: timings['dt_proj'].append(t1 - t0)

        # Reshape B, C
        B = rearrange(B, "(b l) dstate -> b l dstate", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b l dstate", l=seqlen).contiguous()

        # 5. Selective scan (Python reference - this is what we replaced)
        t0 = time.perf_counter()
        y = self._selective_scan_ref(x, dt, self.A, B, C, self.D)
        t1 = time.perf_counter()
        if record: timings['selective_scan'].append(t1 - t0)

        # 6. Z-gate
        t0 = time.perf_counter()
        y = y * F.silu(z)
        t1 = time.perf_counter()
        if record: timings['z_gate'].append(t1 - t0)

        # 7. out_proj
        t0 = time.perf_counter()
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        t1 = time.perf_counter()
        if record: timings['out_proj'].append(t1 - t0)

        return out

    def _selective_scan_ref(self, u, delta, A, B, C, D):
        """Reference selective scan (Python loop)."""
        batch, dim, seqlen = u.shape
        dstate = A.shape[1]
        h = torch.zeros(batch, dim, dstate)
        ys = []

        for l in range(seqlen):
            u_t = u[:, :, l]
            delta_t = delta[:, :, l]
            B_t = B[:, l, :]
            C_t = C[:, l, :]

            # Exact discretization
            A_bar = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0))
            B_bar = delta_t.unsqueeze(-1) * B_t.unsqueeze(1)

            h = A_bar * h + B_bar * u_t.unsqueeze(-1)
            y_t = torch.sum(C_t.unsqueeze(1) * h, dim=-1) + D.unsqueeze(0) * u_t
            ys.append(y_t)

        return torch.stack(ys, dim=2)


def main():
    print("=" * 70)
    print("MambaBlock Component Profiling (d_model=768, L=512)")
    print("=" * 70)

    profiler = MambaBlockProfiler(d_model=768, d_state=16)
    results = profiler.profile(batch=1, seqlen=512, reps=50)

    print(f"\n{'Component':<20} {'Time (ms)':<12} {'% of Total':<12}")
    print("-" * 44)

    total = results['total']
    for name, time_ms in results.items():
        if name != 'total':
            pct = time_ms / total * 100
            print(f"{name:<20} {time_ms:<12.3f} {pct:<12.1f}%")

    print("-" * 44)
    print(f"{'TOTAL':<20} {total:<12.3f}")

    # Analysis
    print("\n" + "=" * 70)
    print("Analysis")
    print("=" * 70)

    scan_pct = results['selective_scan'] / total * 100
    gemm_time = results['in_proj'] + results['x_proj'] + results['dt_proj'] + results['out_proj']
    gemm_pct = gemm_time / total * 100

    print(f"\nSelective Scan: {scan_pct:.1f}% of total time")
    print(f"Linear layers (GEMMs): {gemm_pct:.1f}% of total time")

    # Calculate with our actual ONNX custom op timing
    print("\n" + "-" * 44)
    print("With our ONNX SelectiveScanFusedExact (1.75ms):")
    onnx_scan_time = 1.75  # From benchmark
    new_total = total - results['selective_scan'] + onnx_scan_time
    overall_speedup = total / new_total
    print(f"  New total: {new_total:.3f} ms (was {total:.3f} ms)")
    print(f"  Overall speedup: {overall_speedup:.2f}x")

    print("\nWith our ONNX SelectiveScan (0.50ms, fastest):")
    onnx_scan_time = 0.50
    new_total = total - results['selective_scan'] + onnx_scan_time
    overall_speedup = total / new_total
    print(f"  New total: {new_total:.3f} ms (was {total:.3f} ms)")
    print(f"  Overall speedup: {overall_speedup:.2f}x")


if __name__ == "__main__":
    main()
