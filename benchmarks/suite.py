import argparse
import time
import os
import torch
import onnx
import onnxruntime as ort
import numpy as np
import pandas as pd
import mamba_onnx
from mamba_onnx import register_custom_ops
import torch.nn as nn
from memory_profiler import memory_usage

# --- 1. Model Definitions (Reuse from benchmark_scan.py) ---
class MambaScanReference(nn.Module):
    def __init__(self, d_model=768, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
    def forward(self, u, delta, A, B, C, D):
        batch, dim, seqlen = u.shape
        dstate = self.d_state
        h = torch.zeros(batch, dim, dstate, device=u.device)
        ys = []
        for l in range(seqlen):
            u_t = u[:, :, l]
            delta_t = delta[:, :, l]
            B_t = B[:, l, :]
            C_t = C[:, l, :]
            A_bar = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0))
            B_bar = delta_t.unsqueeze(-1) * B_t.unsqueeze(1)
            term2 = B_bar * u_t.unsqueeze(-1)
            h = A_bar * h + term2
            C_broadcast = C_t.unsqueeze(1)
            y_t = torch.sum(C_broadcast * h, dim=-1)
            y_t = y_t + D.unsqueeze(0) * u_t
            ys.append(y_t)
        return torch.stack(ys, dim=2)

def create_fused_onnx_model(path, D_sz, N_sz):
    from onnx import helper, TensorProto
    input_infos = [
        helper.make_tensor_value_info('u', TensorProto.FLOAT, ['batch', D_sz, 'seqlen']),
        helper.make_tensor_value_info('delta', TensorProto.FLOAT, ['batch', D_sz, 'seqlen']),
        helper.make_tensor_value_info('A', TensorProto.FLOAT, [D_sz, N_sz]),
        helper.make_tensor_value_info('B', TensorProto.FLOAT, ['batch', 'seqlen', N_sz]),
        helper.make_tensor_value_info('C', TensorProto.FLOAT, ['batch', 'seqlen', N_sz]),
        helper.make_tensor_value_info('D', TensorProto.FLOAT, [D_sz]),
    ]
    output_infos = [helper.make_tensor_value_info('out', TensorProto.FLOAT, ['batch', D_sz, 'seqlen'])]
    node = helper.make_node('SelectiveScan', inputs=['u', 'delta', 'A', 'B', 'C', 'D'], outputs=['out'], domain='mamba')
    graph = helper.make_graph([node], 'MambaFused', input_infos, output_infos)
    opset_imports = [helper.make_opsetid("", 17), helper.make_opsetid("mamba", 1)]
    model = helper.make_model(graph, producer_name='mamba-onnx-benchmark', opset_imports=opset_imports)
    model.ir_version = 8
    onnx.save(model, path)

# --- 2. Benchmarking Engine ---

class BenchmarkRunner:
    def __init__(self, use_cuda=False):
        self.providers = ['CUDAExecutionProvider'] if use_cuda else ['CPUExecutionProvider']
        os.makedirs("benchmarks/results", exist_ok=True)
        os.makedirs("benchmarks/models", exist_ok=True)

    def _generate_inputs(self, B_sz, L, D_sz, N=16):
        u = torch.randn(B_sz, D_sz, L)
        delta = torch.rand(B_sz, D_sz, L) * 0.01
        A = -torch.rand(D_sz, N) - 0.5
        B = torch.randn(B_sz, L, N)
        C = torch.randn(B_sz, L, N)
        D = torch.randn(D_sz)
        
        feed = {
            'u': u.numpy(), 'delta': delta.numpy(), 'A': A.numpy(),
            'B': B.numpy(), 'C': C.numpy(), 'D': D.numpy()
        }
        return (u, delta, A, B, C, D), feed

    def run_pytorch(self, inputs, reps=20):
        # inputs[0] is (u, delta, A, B, C, D) tuple
        model = MambaScanReference(inputs[0][0].shape[1], 16)
        # Warmup
        for _ in range(3):
            _ = model(*inputs[0])
        
        start = time.time()
        for _ in range(reps):
            _ = model(*inputs[0])
        end = time.time()
        return (end - start) / reps

    def run_onnx_baseline(self, inputs, reps=20):
        # Only works for small L due to unrolling!
        L = inputs[0][0].shape[2]
        if L > 1024:
            print(f"Skipping Vanilla ONNX for L={L} (Graph export too slow)")
            return None
        
        path = "benchmarks/models/baseline.onnx"
        model = MambaScanReference(inputs[0][0].shape[1], 16)
        try:
            torch.onnx.export(
                model, inputs[0], path, opset_version=17,
                input_names=['u', 'delta', 'A', 'B', 'C', 'D'], output_names=['out']
            )
            sess = ort.InferenceSession(path, providers=self.providers)
            
            for _ in range(3): sess.run(None, inputs[1])
            start = time.time()
            for _ in range(reps): sess.run(None, inputs[1])
            end = time.time()
            return (end - start) / reps
        except Exception as e:
            print(f"Baseline Export/Run Failed: {e}")
            return None

    def run_onnx_fused(self, inputs, reps=20):
        path = "benchmarks/models/fused.onnx"
        D, L = inputs[0][0].shape[1], inputs[0][0].shape[2]
        create_fused_onnx_model(path, D, 16)
        
        sess_opt = ort.SessionOptions()
        register_custom_ops(sess_opt)
        sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(path, sess_opt, providers=self.providers)
        
        for _ in range(3): sess.run(None, inputs[1])
        start = time.time()
        for _ in range(reps): sess.run(None, inputs[1])
        end = time.time()
        return (end - start) / reps

    def run_suite(self):
        results = []
        
        # Scenario 1: Scaling L (D=768)
        print("Benchmarking Scaling Sequence Length (L)...")
        D = 768
        for L in [128, 512, 1024, 2048, 4096]:
            t_data, feed = self._generate_inputs(1, L, D)
            
            lat_pt = self.run_pytorch([t_data, feed])
            lat_bl = self.run_onnx_baseline([t_data, feed])
            lat_fused = self.run_onnx_fused([t_data, feed])
            
            print(f"L={L}: PyTorch={lat_pt*1000:.2f}ms, Fused={lat_fused*1000:.2f}ms, Vanilla={lat_bl*1000 if lat_bl else 0:.2f}ms")
            
            results.append({
                'Scenario': 'Scaling_L', 'L': L, 'D': D,
                'PyTorch_Latency': lat_pt, 
                'Baseline_Latency': lat_bl,
                'Fused_Latency': lat_fused,
                'Throughput_Fused': (L * D) / lat_fused if lat_fused else 0
            })
            
        # Scenario 2: Scaling D (L=1024)
        print("\nBenchmarking Scaling Dimension (D)...")
        L = 1024
        for D in [256, 512, 768, 1024, 1536, 2048]:
            t_data, feed = self._generate_inputs(1, L, D)
            
            lat_pt = self.run_pytorch([t_data, feed])
            lat_fused = self.run_onnx_fused([t_data, feed])
            
            print(f"D={D}: PyTorch={lat_pt*1000:.2f}ms, Fused={lat_fused*1000:.2f}ms")
            
            results.append({
                'Scenario': 'Scaling_D', 'L': L, 'D': D,
                'PyTorch_Latency': lat_pt,
                'Baseline_Latency': None, # Too slow to export every time
                'Fused_Latency': lat_fused,
                'Throughput_Fused': (L * D) / lat_fused if lat_fused else 0
            })

        df = pd.DataFrame(results)
        df.to_csv("benchmarks/results/benchmark_data.csv", index=False)
        print("\nResults saved to benchmarks/results/benchmark_data.csv")

if __name__ == "__main__":
    runner = BenchmarkRunner()
    runner.run_suite()
