import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort
import numpy as np
import time
import os

# 1. Define the Python Reference (The Slow Path)
class MambaScanReference(nn.Module):
    def __init__(self, d_model=768, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
    def forward(self, u, delta, A, B, C, D):
        # u: (B, D, L)
        # delta: (B, D, L)
        # A: (D, N)
        # B: (B, L, N)
        # C: (B, L, N)
        # D: (D)
        
        batch, dim, seqlen = u.shape
        dstate = self.d_state
        
        # Reference Loop
        # Note: We implement the logic to match our Kernel:
        # h_t = exp(delta * A) * h_{t-1} + (delta * B) * u_t
        # y_t = C * h_t + D * u_t
        
        # Pre-compute discretized terms for efficiency in python (vectorized over batch/dim)
        # But our kernel does it step by step.
        
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
            # A: (D, N), delta_t: (B, D) -> (B, D, N)
            A_bar = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0))
            
            # B_bar = delta_t * B_t
            # B_t: (B, N), delta_t: (B, D) -> (B, D, N)
            # Broadcast B across D?
            # In our kernel, B is (B, L, N).
            # The kernel loop:
            # B_bar_0 = delta_vec * B_t_0
            # Here B_t_0 comes from B_data.
            # B_data has shape (B, L, N).
            # We treat B as shared across D.
            B_bar = delta_t.unsqueeze(-1) * B_t.unsqueeze(1) # (B, D, N)
            
            # Update h
            # h: (B, D, N)
            # u_t: (B, D) -> (B, D, 1)
            term2 = B_bar * u_t.unsqueeze(-1)
            h = A_bar * h + term2
            
            # Output
            # y_t = C_t * h_t
            # C_t: (B, N) -> (B, D, N) (broadcast)
            C_broadcast = C_t.unsqueeze(1)
            y_t = torch.sum(C_broadcast * h, dim=-1) # (B, D)
            
            # Add D * u
            y_t = y_t + D.unsqueeze(0) * u_t
            
            ys.append(y_t)
            
        return torch.stack(ys, dim=2) # (B, D, L)


# 2. Define the Custom Op Shim
class MambaScanOp(torch.autograd.Function):
    @staticmethod
    def symbolic(g, u, delta, A, B, C, D):
        return g.op("mamba::SelectiveScan", u, delta, A, B, C, D)

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D):
        # Use the reference implementation for the PyTorch forward pass
        ref = MambaScanReference(u.shape[1], A.shape[1])
        return ref(u, delta, A, B, C, D)

class MambaScanFused(nn.Module):
    def __init__(self, d_model=768, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
    def forward(self, u, delta, A, B, C, D):
        return MambaScanOp.apply(u, delta, A, B, C, D)


def benchmark():
    # Parameters
    B_sz = 1
    L = 512 # Increased L to show better speedup
    D_sz = 768
    N = 16
    
    # Inputs
    u = torch.randn(B_sz, D_sz, L)
    delta = torch.randn(B_sz, D_sz, L)
    A = torch.randn(D_sz, N)
    B = torch.randn(B_sz, L, N)
    C = torch.randn(B_sz, L, N)
    D = torch.randn(D_sz)
    
    inputs = (u, delta, A, B, C, D)
    
    # Export Reference
    ref_model = MambaScanReference(D_sz, N)
    ref_path = "mamba_reference.onnx"
    print(f"Exporting Reference to {ref_path}...")
    torch.onnx.export(
        ref_model,
        inputs,
        ref_path,
        opset_version=17,
        input_names=['u', 'delta', 'A', 'B', 'C', 'D'],
        output_names=['out']
    )
    
    # Export Fused
    fused_model = MambaScanFused(D_sz, N)
    fused_path = "mamba_fused.onnx"
    print(f"Exporting Fused to {fused_path}...")
    torch.onnx.export(
        fused_model,
        inputs,
        fused_path,
        opset_version=17,
        input_names=['u', 'delta', 'A', 'B', 'C', 'D'],
        output_names=['out']
    )
    
    # Benchmark
    print("\nBenchmarking...")
    
    # Load Library
    lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../lib/libmamba_ops.so"))
    print(f"Loading custom op library from: {lib_path}")
    
    sess_options = ort.SessionOptions()
    sess_options.register_custom_ops_library(lib_path)
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Session Fused
    sess_fused = ort.InferenceSession(fused_path, sess_options, providers=['CPUExecutionProvider'])
    
    # Session Reference (Standard ORT)
    sess_ref = ort.InferenceSession(ref_path, providers=['CPUExecutionProvider'])
    
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
        sess_ref.run(None, input_feed)
        sess_fused.run(None, input_feed)
        
    iters = 20
    
    # Measure Ref
    start = time.time()
    for _ in range(iters):
        sess_ref.run(None, input_feed)
    end = time.time()
    ref_lat = (end - start) / iters
    
    # Measure Fused
    start = time.time()
    for _ in range(iters):
        sess_fused.run(None, input_feed)
    end = time.time()
    fused_lat = (end - start) / iters
    
    print(f"\nResults (SeqLen={L}, Dim={D_sz}):")
    print(f"Reference Latency: {ref_lat*1000:.2f} ms")
    print(f"Fused Latency:     {fused_lat*1000:.2f} ms")
    print(f"Speedup:           {ref_lat/fused_lat:.2f}x")

if __name__ == "__main__":
    benchmark()
