import torch
from transformers import MambaConfig, MambaModel
import onnxruntime as ort
import time
import numpy as np
import os

def export_and_benchmark():
    # Configuration to match the goal: d_model=768, d_state=16
    # We use 1 layer to isolate the performance of the block(s) without excessive depth.
    config = MambaConfig(
        d_model=768,
        d_state=16,
        num_hidden_layers=1,
        use_cache=False # We are benchmarking the prompt processing (parallel scan), not generation (step-by-step)
    )
    
    model = MambaModel(config)
    model.eval()
    
    # Input: (Batch, SeqLen)
    # We pass inputs_embeds to avoid the embedding layer lookup in the benchmark if desired, 
    # but passing input_ids is standard. Let's use input_ids for simplicity.
    batch = 1
    seqlen = 64
    input_ids = torch.randint(0, 1000, (batch, seqlen))
    
    # Export to ONNX
    onnx_path = "mamba_baseline.onnx"
    print("Exporting model to ONNX...")
    
    # We need to trace it.
    # transformers Mamba might have dynamic control flow or complex logic.
    # We'll try a direct export.
    try:
        torch.onnx.export(
            model, 
            input_ids, 
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input_ids'],
            output_names=['last_hidden_state']
        )
        print(f"Model exported to {onnx_path}")
    except Exception as e:
        print(f"Export failed: {e}")
        return

    # Benchmark with ONNX Runtime
    print("Benchmarking with ONNX Runtime (CPU)...")
    # Silence warnings
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3
    
    session = ort.InferenceSession(onnx_path, sess_options, providers=['CPUExecutionProvider'])
    
    # Input data for benchmark
    input_feed = {'input_ids': input_ids.numpy()}
    
    # Warmup
    for _ in range(5):
        session.run(None, input_feed)
        
    # Measure
    iters = 20
    start = time.time()
    for _ in range(iters):
        session.run(None, input_feed)
    end = time.time()
    
    avg_latency = (end - start) / iters
    latency_per_token = avg_latency / seqlen
    
    print(f"Average Latency: {avg_latency:.4f}s")
    print(f"Latency per token: {latency_per_token * 1000:.4f}ms")
    
    # Graph info (basic)
    model_onnx = onnx.load(onnx_path)
    print(f"Number of nodes in the graph: {len(model_onnx.graph.node)}")
    
import onnx

if __name__ == "__main__":
    export_and_benchmark()
