import onnxruntime as ort
import numpy as np
import os
import sys

def test_custom_op():
    print("Testing Custom Op...")
    
    # Path to library
    lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../lib/libmamba_ops.so"))
    if not os.path.exists(lib_path):
        print(f"Library not found at {lib_path}")
        return

    print(f"Loading library: {lib_path}")
    sess_options = ort.SessionOptions()
    try:
        sess_options.register_custom_ops_library(lib_path)
    except Exception as e:
        print(f"Failed to register library: {e}")
        return

    # Create a simple graph using the custom op?
    # Alternatively, load the already exported 'mamba_fused.onnx' if it exists.
    model_path = "mamba_fused.onnx"
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found. Run benchmark first or create a dummy model.")
        return

    print(f"Loading model: {model_path}")
    try:
        session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"Failed to create session: {e}")
        return

    # Inputs (Small)
    B = 1
    L = 128
    D = 768
    N = 16
    
    u = np.random.randn(B, D, L).astype(np.float32)
    delta = np.random.randn(B, D, L).astype(np.float32)
    A = np.random.randn(D, N).astype(np.float32)
    B_in = np.random.randn(B, L, N).astype(np.float32)
    C = np.random.randn(B, L, N).astype(np.float32)
    D_in = np.random.randn(D).astype(np.float32)

    input_feed = {
        'u': u,
        'delta': delta,
        'A': A,
        'B': B_in,
        'C': C,
        'D': D_in
    }

    print("Running Inference...")
    try:
        out = session.run(None, input_feed)
        print("Inference Successful!")
        print("Output shape:", out[0].shape)
    except Exception as e:
        print(f"Inference Failed: {e}")

if __name__ == "__main__":
    test_custom_op()
