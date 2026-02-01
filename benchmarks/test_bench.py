"""
pytest-benchmark tests for Mamba selective scan custom ops.

Run benchmarks:
    pytest benchmarks/ --benchmark-only
    pytest benchmarks/ --benchmark-only --benchmark-save=baseline
    pytest benchmarks/ --benchmark-only -k "not scale"
"""
import os
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import pytest
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

SCALE_Ls = [128, 256, 512, 1024, 2048, 4096]
SCALE_Ds = [256, 512, 768, 1024, 1536, 2048]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _available(backend):
    return backend in mamba_onnx.available_backends()


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


def _create_session(op_name, D, use_fused, backend, threads=0):
    """Create ONNX model + ORT session. Returns (session, tmpdir_handle).

    The tmpdir handle must be kept alive for the session's lifetime since
    the ONNX model file lives there.
    """
    tmpdir = tempfile.TemporaryDirectory()
    path = _create_model(tmpdir.name, op_name, D, N, use_fused)
    opts = ort.SessionOptions()
    if threads > 0:
        opts.intra_op_num_threads = threads
    mamba_onnx.register_custom_ops(opts, backend=backend)
    sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    return sess, tmpdir


def _reference_selective_scan(u, delta, A, B, C, D, z=None, use_fused=False, use_exact=False):
    """PyTorch reference implementation."""
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


# ---------------------------------------------------------------------------
# Compare mode: B=1, D=768, L=1024, all ops x backends
# ---------------------------------------------------------------------------

@pytest.mark.benchmark(group="compare-onnx")
@pytest.mark.parametrize("op_name,use_fused,use_exact", ALL_OPS,
                         ids=[o[0] for o in ALL_OPS])
@pytest.mark.parametrize("backend", ["cpp", "zig"])
def test_bench_onnx(benchmark, op_name, use_fused, use_exact, backend, thread_count):
    if not _available(backend):
        pytest.skip(f"{backend} not available")
    B, D, L = 1, 768, 1024
    _, feed = _make_inputs(B, D, L, use_fused)
    sess, tmpdir_handle = _create_session(op_name, D, use_fused, backend, threads=thread_count)
    benchmark.pedantic(sess.run, args=(None, feed), rounds=50, warmup_rounds=10)
    benchmark.extra_info.update(scenario="compare", op=op_name, backend=backend, B=B, D=D, L=L)


@pytest.mark.benchmark(group="compare-pytorch")
@pytest.mark.parametrize("op_name,use_fused,use_exact", ALL_OPS,
                         ids=[o[0] for o in ALL_OPS])
def test_bench_pytorch(benchmark, op_name, use_fused, use_exact):
    B, D, L = 1, 768, 1024
    tensors, _ = _make_inputs(B, D, L, use_fused)
    u, delta, A, B_in, C, D_in, z = tensors

    def run():
        with torch.no_grad():
            _reference_selective_scan(u, delta, A, B_in, C, D_in, z, use_fused, use_exact)

    benchmark.pedantic(run, rounds=50, warmup_rounds=3)
    benchmark.extra_info.update(scenario="compare", op=op_name, backend="pytorch", B=B, D=D, L=L)


# ---------------------------------------------------------------------------
# Scale L: D=768, sweep L
# ---------------------------------------------------------------------------

@pytest.mark.benchmark(group="scale-L")
@pytest.mark.parametrize("L", SCALE_Ls)
@pytest.mark.parametrize("op_name,use_fused,use_exact", ALL_OPS,
                         ids=[o[0] for o in ALL_OPS])
@pytest.mark.parametrize("backend", ["cpp", "zig"])
def test_bench_scale_L(benchmark, L, op_name, use_fused, use_exact, backend, thread_count):
    if not _available(backend):
        pytest.skip(f"{backend} not available")
    D = 768
    _, feed = _make_inputs(1, D, L, use_fused)
    sess, tmpdir_handle = _create_session(op_name, D, use_fused, backend, threads=thread_count)
    benchmark.pedantic(sess.run, args=(None, feed), rounds=50, warmup_rounds=10)
    benchmark.extra_info.update(scenario="Scaling_L", op=op_name, backend=backend, D=D, L=L)


# ---------------------------------------------------------------------------
# Scale D: L=1024, sweep D
# ---------------------------------------------------------------------------

@pytest.mark.benchmark(group="scale-D")
@pytest.mark.parametrize("D", SCALE_Ds)
@pytest.mark.parametrize("op_name,use_fused,use_exact", ALL_OPS,
                         ids=[o[0] for o in ALL_OPS])
@pytest.mark.parametrize("backend", ["cpp", "zig"])
def test_bench_scale_D(benchmark, D, op_name, use_fused, use_exact, backend, thread_count):
    if not _available(backend):
        pytest.skip(f"{backend} not available")
    L = 1024
    _, feed = _make_inputs(1, D, L, use_fused)
    sess, tmpdir_handle = _create_session(op_name, D, use_fused, backend, threads=thread_count)
    benchmark.pedantic(sess.run, args=(None, feed), rounds=50, warmup_rounds=10)
    benchmark.extra_info.update(scenario="Scaling_D", op=op_name, backend=backend, D=D, L=L)
