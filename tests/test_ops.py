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


# Small dimensions for fast tests. The kernel hardcodes N=16 (two Vec8 chunks).
D, N, B_SZ, L = 64, 16, 1, 32


@pytest.mark.parametrize(
    "op_name, use_fused, use_exact",
    [
        ("SelectiveScan", False, False),
        ("SelectiveScanExact", False, True),
        ("SelectiveScanFused", True, False),
        ("SelectiveScanFusedExact", True, True),
    ],
)
def test_selective_scan(op_name, use_fused, use_exact):
    sess_options = ort.SessionOptions()
    mamba_onnx.register_custom_ops(sess_options)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, f"{op_name.lower()}.onnx")
        _create_onnx_model(model_path, op_name, D, N, use_fused)
        session = ort.InferenceSession(
            model_path, sess_options, providers=["CPUExecutionProvider"]
        )

    torch.manual_seed(42)

    u = torch.randn(B_SZ, D, L)
    A = -torch.rand(D, N) - 0.5
    B_in = torch.randn(B_SZ, L, N) * 0.1
    C = torch.randn(B_SZ, L, N) * 0.1
    D_in = torch.randn(D) * 0.1

    if use_fused:
        delta = torch.randn(B_SZ, D, L) * 0.5
        z = torch.randn(B_SZ, D, L)
    else:
        delta = torch.rand(B_SZ, D, L) * 0.1 + 0.01
        z = None

    with torch.no_grad():
        out_ref = reference_selective_scan(
            u, delta, A, B_in, C, D_in, z, use_fused, use_exact
        ).numpy()

    input_feed = {
        "u": u.numpy(),
        "delta": delta.numpy(),
        "A": A.numpy(),
        "B": B_in.numpy(),
        "C": C.numpy(),
        "D": D_in.numpy(),
    }
    if use_fused:
        input_feed["z"] = z.numpy()

    out_onnx = session.run(None, input_feed)[0]

    np.testing.assert_allclose(out_onnx, out_ref, rtol=1e-4, atol=1e-4)
