"""Tests for Mamba selective scan custom ops against a PyTorch reference."""

import tempfile
import os

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
import torch.nn.functional as F
from onnx import TensorProto, helper

import mamba_onnx


def reference_selective_scan(u, delta, A, B, C, D, z=None, use_fused=False, use_exact=False):
    """
    PyTorch reference implementation of selective scan.

    Args:
        u: [batch, dim, seqlen]
        delta: [batch, dim, seqlen]
        A: [dim, dstate]
        B: [batch, seqlen, dstate]
        C: [batch, seqlen, dstate]
        D: [dim]
        z: [batch, dim, seqlen] (fused variants only)
        use_fused: apply Softplus to delta and SiLU gate to output
        use_exact: use exp(delta*A) instead of linear approximation
    """
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


def _create_onnx_model(path, op_name, D_sz, N_sz, use_fused):
    """Create a minimal ONNX model for a single custom op."""
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
    graph = helper.make_graph([node], f"{op_name}Test", input_infos, output_infos)
    opset_imports = [helper.make_opsetid("", 17), helper.make_opsetid("mamba", 1)]
    model = helper.make_model(graph, producer_name="mamba-test", opset_imports=opset_imports)
    model.ir_version = 8
    onnx.save(model, path)


# N is hardcoded to 16 in the kernels (two Vec8 chunks).
N = 16

ALL_OPS = [
    ("SelectiveScan", False, False),
    ("SelectiveScanExact", False, True),
    ("SelectiveScanFused", True, False),
    ("SelectiveScanFusedExact", True, True),
]


def _make_inputs(B, D, L, use_fused, seed=42):
    """Generate random inputs for a selective scan test."""
    torch.manual_seed(seed)
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

    return u, delta, A, B_in, C, D_in, z


def _to_feed(u, delta, A, B_in, C, D_in, z):
    """Convert torch tensors to an ONNX Runtime input feed dict."""
    feed = {
        "u": u.numpy(), "delta": delta.numpy(), "A": A.numpy(),
        "B": B_in.numpy(), "C": C.numpy(), "D": D_in.numpy(),
    }
    if z is not None:
        feed["z"] = z.numpy()
    return feed


def _run_onnx(op_name, use_fused, D, L, feed, backend):
    """Build an ONNX model, load it with the given backend, and run inference."""
    opts = ort.SessionOptions()
    mamba_onnx.register_custom_ops(opts, backend=backend)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, f"{op_name.lower()}.onnx")
        _create_onnx_model(path, op_name, D, N, use_fused)
        sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    return sess.run(None, feed)[0]


def _skip_unless(backend):
    if backend not in mamba_onnx.available_backends():
        pytest.skip(f"{backend} backend not built")


# ---------------------------------------------------------------------------
# 1. Core correctness: every op × every backend vs PyTorch reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", ["cpp", "zig"])
@pytest.mark.parametrize("op_name, use_fused, use_exact", ALL_OPS)
def test_selective_scan(op_name, use_fused, use_exact, backend):
    _skip_unless(backend)
    B, D, L = 1, 64, 32

    u, delta, A, B_in, C, D_in, z = _make_inputs(B, D, L, use_fused)
    feed = _to_feed(u, delta, A, B_in, C, D_in, z)

    with torch.no_grad():
        out_ref = reference_selective_scan(
            u, delta, A, B_in, C, D_in, z, use_fused, use_exact
        ).numpy()

    out_onnx = _run_onnx(op_name, use_fused, D, L, feed, backend)
    np.testing.assert_allclose(out_onnx, out_ref, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# 2. Shape variety: batching, longer sequences, wider dimensions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", ["cpp", "zig"])
@pytest.mark.parametrize(
    "B, D, L",
    [
        (1, 64, 1),      # single timestep
        (4, 64, 32),     # batched
        (1, 256, 128),   # wider D, longer L
        (2, 768, 64),    # realistic D with batching
    ],
    ids=["L=1", "B=4", "D=256_L=128", "D=768_B=2"],
)
def test_shapes(B, D, L, backend):
    """SelectiveScan (simplest variant) across a range of shapes."""
    _skip_unless(backend)
    op_name, use_fused, use_exact = "SelectiveScan", False, False

    u, delta, A, B_in, C, D_in, z = _make_inputs(B, D, L, use_fused)
    feed = _to_feed(u, delta, A, B_in, C, D_in, z)

    with torch.no_grad():
        out_ref = reference_selective_scan(
            u, delta, A, B_in, C, D_in, z, use_fused, use_exact
        ).numpy()

    out_onnx = _run_onnx(op_name, use_fused, D, L, feed, backend)
    np.testing.assert_allclose(out_onnx, out_ref, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# 3. Numerical edge cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", ["cpp", "zig"])
@pytest.mark.parametrize(
    "regime",
    ["zero_input", "large_negative_A", "large_delta"],
    ids=["zero_input", "large_neg_A", "large_delta"],
)
def test_numerical_edge_cases(regime, backend):
    """Test tricky numerical regimes against the reference."""
    _skip_unless(backend)
    B, D, L = 1, 64, 32
    torch.manual_seed(99)

    # Start with normal inputs, then modify for the regime
    u = torch.randn(B, D, L)
    A = -torch.rand(D, N) - 0.5
    B_in = torch.randn(B, L, N) * 0.1
    C = torch.randn(B, L, N) * 0.1
    D_in = torch.randn(D) * 0.1
    delta = torch.rand(B, D, L) * 0.1 + 0.01

    if regime == "zero_input":
        u = torch.zeros_like(u)
        D_in = torch.zeros_like(D_in)
    elif regime == "large_negative_A":
        A = -torch.ones(D, N) * 10.0
    elif regime == "large_delta":
        delta = torch.ones(B, D, L) * 2.0

    feed = _to_feed(u, delta, A, B_in, C, D_in, None)

    with torch.no_grad():
        out_ref = reference_selective_scan(
            u, delta, A, B_in, C, D_in, use_exact=False
        ).numpy()

    out_onnx = _run_onnx("SelectiveScan", False, D, L, feed, backend)

    # Wider tolerance for extreme regimes — values can get large
    np.testing.assert_allclose(out_onnx, out_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("backend", ["cpp", "zig"])
def test_numerical_edge_cases_exact(backend):
    """Large negative A with exact discretization (exp path)."""
    _skip_unless(backend)
    B, D, L = 1, 64, 32
    torch.manual_seed(99)

    u = torch.randn(B, D, L)
    A = -torch.ones(D, N) * 5.0  # fast decay
    B_in = torch.randn(B, L, N) * 0.1
    C = torch.randn(B, L, N) * 0.1
    D_in = torch.randn(D) * 0.1
    delta = torch.rand(B, D, L) * 0.5 + 0.01

    feed = _to_feed(u, delta, A, B_in, C, D_in, None)

    with torch.no_grad():
        out_ref = reference_selective_scan(
            u, delta, A, B_in, C, D_in, use_exact=True
        ).numpy()

    out_onnx = _run_onnx("SelectiveScanExact", False, D, L, feed, backend)
    np.testing.assert_allclose(out_onnx, out_ref, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# 4. Cross-backend agreement: cpp and zig produce identical outputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("op_name, use_fused, use_exact", ALL_OPS)
def test_cross_backend_agreement(op_name, use_fused, use_exact):
    """Both backends should produce bitwise-close results for the same inputs."""
    backends = mamba_onnx.available_backends()
    if "cpp" not in backends or "zig" not in backends:
        pytest.skip("need both backends built")

    B, D, L = 2, 128, 64
    u, delta, A, B_in, C, D_in, z = _make_inputs(B, D, L, use_fused)
    feed = _to_feed(u, delta, A, B_in, C, D_in, z)

    out_cpp = _run_onnx(op_name, use_fused, D, L, feed, "cpp")
    out_zig = _run_onnx(op_name, use_fused, D, L, feed, "zig")

    np.testing.assert_allclose(out_cpp, out_zig, rtol=1e-4, atol=1e-4)
