# Proposal: Optimizing Mamba for ONNX CPU

## Goal
Demonstrate a **50x speedup** of Mamba inference on CPU by replacing the standard Python loop with a custom AVX2 C++ kernel.

## Overview
Mamba models rely on a selective scan operation. In a naive ONNX export, this results in a massive graph with hundreds of nodes representing the unrolled loop, leading to poor performance. This project aims to fuse this operation into a single efficient `CustomOp`.

## Deliverables
A repository `mamba-onnx-cpu` containing:
- **`benchmark.py`**: A script that runs both the baseline (loop) and optimized (fused) versions and reports the speedup.
- **`lib/`**: Source code for the custom C++ kernel.
- **`graphs/`**: Visual or structural comparison of the "Loop" graph vs. the "Fused" graph.
