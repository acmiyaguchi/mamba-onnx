"""
Test suite for all 4 Mamba custom ops.

Ops:
  - SelectiveScan:          6-input, linear discretization
  - SelectiveScanExact:     6-input, exact (exp) discretization
  - SelectiveScanFused:     7-input, linear discretization + Softplus/SiLU
  - SelectiveScanFusedExact: 7-input, exact discretization + Softplus/SiLU
"""
import torch
import torch.nn.functional as F
import onnxruntime as ort
import numpy as np
import os
import sys
import onnx
from onnx import helper, TensorProto

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mamba_onnx


# =============================================================================
# PyTorch Reference Implementation
# =============================================================================

def reference_selective_scan(u, delta, A, B, C, D, z=None, use_fused=False, use_exact=False):
    """
    Reference implementation of selective scan.

    Args:
        u: [batch, dim, seqlen]
        delta: [batch, dim, seqlen] - raw if fused, post-softplus otherwise
        A: [dim, dstate]
        B: [batch, seqlen, dstate]
        C: [batch, seqlen, dstate]
        D: [dim]
        z: [batch, dim, seqlen] - only used if fused
        use_fused: if True, apply Softplus to delta and SiLU gate to output
        use_exact: if True, use exp(delta*A); otherwise use linear approximation
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


# =============================================================================
# ONNX Model Creation
# =============================================================================

def create_onnx_model(path, op_name, D_sz, N_sz, use_fused):
    """Create ONNX model for a specific op variant."""
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
    graph = helper.make_graph([node], f'{op_name}Test', input_infos, output_infos)
    opset_imports = [helper.make_opsetid("", 17), helper.make_opsetid("mamba", 1)]
    model = helper.make_model(graph, producer_name='mamba-test', opset_imports=opset_imports)
    model.ir_version = 8
    onnx.save(model, path)


# =============================================================================
# Test Functions
# =============================================================================

def test_op(op_name, use_fused, use_exact):
    """Test a specific op variant."""
    print(f"\n{'='*60}")
    print(f"Testing {op_name}")
    print(f"  Fused: {use_fused}, Exact: {use_exact}")
    print('='*60)

    sess_options = ort.SessionOptions()
    mamba_onnx.register_custom_ops(sess_options)

    D, N, B_sz, L = 768, 16, 1, 128

    model_path = f"/tmp/{op_name.lower()}_test.onnx"
    create_onnx_model(model_path, op_name, D, N, use_fused)
    session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])

    # Generate inputs
    torch.manual_seed(42)
    np.random.seed(42)

    u = torch.randn(B_sz, D, L)
    A = -torch.rand(D, N) - 0.5  # Negative for stability
    B_in = torch.randn(B_sz, L, N) * 0.1
    C = torch.randn(B_sz, L, N) * 0.1
    D_in = torch.randn(D) * 0.1

    if use_fused:
        # delta_raw can be any value (will be passed through Softplus)
        delta = torch.randn(B_sz, D, L) * 0.5
        z = torch.randn(B_sz, D, L)
    else:
        # delta should be positive (already post-Softplus)
        delta = torch.rand(B_sz, D, L) * 0.1 + 0.01
        z = None

    # Run PyTorch Reference
    with torch.no_grad():
        out_ref = reference_selective_scan(u, delta, A, B_in, C, D_in, z, use_fused, use_exact).numpy()

    # Run ONNX
    input_feed = {
        'u': u.numpy(), 'delta': delta.numpy(), 'A': A.numpy(),
        'B': B_in.numpy(), 'C': C.numpy(), 'D': D_in.numpy()
    }
    if use_fused:
        input_feed['z'] = z.numpy()

    out_onnx = session.run(None, input_feed)[0]

    # Compare
    print(f"Ref Range:  [{out_ref.min():.4f}, {out_ref.max():.4f}]")
    print(f"ONNX Range: [{out_onnx.min():.4f}, {out_onnx.max():.4f}]")

    diff = np.abs(out_onnx - out_ref)
    print(f"Max Diff: {diff.max():.6f}")
    print(f"Mean Diff: {diff.mean():.6f}")

    # Tolerance - should be tight for all variants
    try:
        np.testing.assert_allclose(out_onnx, out_ref, rtol=1e-4, atol=1e-4)
        print(f"SUCCESS: {op_name} matches reference!")
        return True
    except AssertionError as e:
        print(f"FAILURE: {op_name} mismatch!")
        print(str(e)[:500])
        return False


def main():
    print("="*60)
    print("Mamba Custom Ops Test Suite - All 4 Variants")
    print("="*60)

    # Define all 4 ops
    ops = [
        ("SelectiveScan",          False, False),  # 6-input, linear
        ("SelectiveScanExact",     False, True),   # 6-input, exact
        ("SelectiveScanFused",     True,  False),  # 7-input, linear
        ("SelectiveScanFusedExact", True,  True),  # 7-input, exact
    ]

    results = []
    for op_name, use_fused, use_exact in ops:
        passed = test_op(op_name, use_fused, use_exact)
        results.append((op_name, passed))

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    all_passed = all(r[1] for r in results)
    print("="*60)
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("="*60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
