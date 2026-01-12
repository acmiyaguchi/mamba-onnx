import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort
import numpy as np
import time
import os
import mamba_onnx
import onnx
from onnx import helper, TensorProto

# 1. Define the Python Reference (The Slow Path)
class MambaScanReference(nn.Module):
    def __init__(self, d_model=768, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
    def forward(self, u, delta, A, B, C, D):
        batch, dim, seqlen = u.shape
        dstate = self.d_state
        
        # To match the kernel exactly:
        h = torch.zeros(batch, dim, dstate, device=u.device)
        ys = []
        
        # We process time sequentially
        for l in range(seqlen):
            u_t = u[:, :, l] # (B, D)
            delta_t = delta[:, :, l] # (B, D)
            B_t = B[:, l, :] # (B, N)
            C_t = C[:, l, :] # (B, N)
            
            # A_bar = exp(delta_t * A)
            # The C++ Kernel uses 1 + delta * A approx
            # But here we use exp to be "ground truth" Mamba.
            # If inputs are small, they match.
            A_bar = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0))
            
            # B_bar = delta_t * B_t
            B_bar = delta_t.unsqueeze(-1) * B_t.unsqueeze(1) # (B, D, N)
            
            # Update h
            term2 = B_bar * u_t.unsqueeze(-1)
            h = A_bar * h + term2
            
            # Output
            C_broadcast = C_t.unsqueeze(1)
            y_t = torch.sum(C_broadcast * h, dim=-1) # (B, D)
            
            # Add D * u
            y_t = y_t + D.unsqueeze(0) * u_t
            
            ys.append(y_t)
            
        return torch.stack(ys, dim=2) # (B, D, L)

def create_fused_onnx_model(path, D_sz, N_sz):
    # Inputs
    input_infos = [
        helper.make_tensor_value_info('u', TensorProto.FLOAT, ['batch', D_sz, 'seqlen']),
        helper.make_tensor_value_info('delta', TensorProto.FLOAT, ['batch', D_sz, 'seqlen']),
        helper.make_tensor_value_info('A', TensorProto.FLOAT, [D_sz, N_sz]),
        helper.make_tensor_value_info('B', TensorProto.FLOAT, ['batch', 'seqlen', N_sz]),
        helper.make_tensor_value_info('C', TensorProto.FLOAT, ['batch', 'seqlen', N_sz]),
        helper.make_tensor_value_info('D', TensorProto.FLOAT, [D_sz]),
    ]
    
    output_infos = [
        helper.make_tensor_value_info('out', TensorProto.FLOAT, ['batch', D_sz, 'seqlen']),
    ]
    
    # Node
    node = helper.make_node(
        'SelectiveScan',
        inputs=['u', 'delta', 'A', 'B', 'C', 'D'],
        outputs=['out'],
        domain='mamba'
    )
    
    # Graph
    graph = helper.make_graph(
        [node],
        'MambaFused',
        input_infos,
        output_infos
    )
    
    # Model
    opset_imports = [
        helper.make_opsetid("", 17),
        helper.make_opsetid("mamba", 1)
    ]
    model = helper.make_model(
        graph, 
        producer_name='mamba-onnx-benchmark',
        opset_imports=opset_imports
    )
    model.ir_version = 8
    
    onnx.save(model, path)
    print(f"Manually created ONNX model at {path}")

def benchmark():
    # Parameters
    B_sz = 1
    L = 128 # Reduced to allow Baseline ONNX export (Unrolled loop explodes at 2048)
    D_sz = 768
    N = 16
    
    print(f"Benchmarking with L={L} (Small Sequence for Baseline Comparison), D={D_sz}...")
    
    # Inputs (Scaled for stability and approximation validity)
    # The kernel uses (1 + delta*A) approximation, so we need small delta*A.
    # Also for stability, (1 + delta*A) should be < 1.
    
    u = torch.randn(B_sz, D_sz, L)
    
    # delta: Small positive values
    delta = torch.rand(B_sz, D_sz, L) * 0.01
    
    # A: Negative values (e.g. -0.5 to -1.5)
    A = -torch.rand(D_sz, N) - 0.5
    
    # B, C: Standard normal
    B = torch.randn(B_sz, L, N)
    C = torch.randn(B_sz, L, N)
    D = torch.randn(D_sz)
    
    inputs = (u, delta, A, B, C, D)
    
    input_feed = {
        'u': u.numpy(),
        'delta': delta.numpy(),
        'A': A.numpy(),
        'B': B.numpy(),
        'C': C.numpy(),
        'D': D.numpy()
    }
    
    # 1. PyTorch Eager Baseline
    ref_model = MambaScanReference(D_sz, N)
    print("Running PyTorch Eager Benchmark...")
    
    # Warmup
    for _ in range(5):
        _ = ref_model(*inputs)
        
    start = time.time()
    for _ in range(20):
        ref_out = ref_model(*inputs)
    end = time.time()
    pytorch_lat = (end - start) / 20
    print(f"PyTorch Eager Latency: {pytorch_lat*1000:.2f} ms")

    # 2. Baseline ONNX (Standard Ops - Unrolled Loop)
    baseline_path = "mamba_baseline.onnx"
    print(f"\nAttempting to export Baseline (Unrolled) to {baseline_path}...")
    try:
        # Use static export to avoid dynamic shape complexity for the unrolled loop
        torch.onnx.export(
            ref_model,
            inputs,
            baseline_path,
            opset_version=17,
            input_names=['u', 'delta', 'A', 'B', 'C', 'D'],
            output_names=['out'],
            do_constant_folding=True
        )
        print("Export successful. Benchmarking Baseline ONNX...")
        
        # Load Baseline
        sess_options_base = ort.SessionOptions()
        sess_options_base.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_baseline = ort.InferenceSession(baseline_path, sess_options_base, providers=['CPUExecutionProvider'])
        
        # Warmup
        for _ in range(5):
            sess_baseline.run(None, input_feed)
            
        start = time.time()
        for _ in range(20):
            sess_baseline.run(None, input_feed)
        end = time.time()
        baseline_lat = (end - start) / 20
        print(f"Baseline ONNX Latency: {baseline_lat*1000:.2f} ms")
        print(f"Baseline vs Fused:     {baseline_lat/pytorch_lat:.2f}x (vs PyTorch)")
        
    except Exception as e:
        print(f"Skipping Baseline ONNX: Could not export/run unrolled loop. Error: {e}")
        baseline_lat = None

    # 3. ONNX Runtime Fused
    fused_path = "mamba_fused.onnx"
    create_fused_onnx_model(fused_path, D_sz, N)
    
    print("Running ONNX Runtime Fused Benchmark...")
    sess_options = ort.SessionOptions()
    mamba_onnx.register_custom_ops(sess_options)
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    sess_fused = ort.InferenceSession(fused_path, sess_options, providers=['CPUExecutionProvider'])
    
    input_feed = {
        'u': u.numpy(),
        'delta': delta.numpy(),
        'A': A.numpy(),
        'B': B.numpy(),
        'C': C.numpy(),
        'D': D.numpy()
    }
    
    # Warmup
    for _ in range(5):
        sess_fused.run(None, input_feed)
        
    start = time.time()
    for _ in range(20):
        ort_out = sess_fused.run(None, input_feed)
    end = time.time()
    fused_lat = (end - start) / 20
    print(f"Fused ONNX Latency:    {fused_lat*1000:.2f} ms")
    
    print(f"Speedup:               {pytorch_lat/fused_lat:.2f}x")
    
    # Correctness Check
    print("\nVerifying Correctness...")
    ref_np = ref_out.detach().numpy()
    out_np = ort_out[0]
    
    diff = np.abs(ref_np - out_np)
    max_diff = np.max(diff)
    print(f"Max Absolute Difference: {max_diff}")
    
    if max_diff < 1e-2: # Tolerance for approximation vs exact exp
        print("PASS: Outputs match.")
    else:
        print("FAIL: Outputs diverge significantly.")

if __name__ == "__main__":
    benchmark()