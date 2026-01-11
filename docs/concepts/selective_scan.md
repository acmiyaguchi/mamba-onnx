# Selective Scan Logic

## Overview
The Selective Scan is the core operation of Mamba (State Space Models). It is a recurrent operation that updates a hidden state $h_t$ based on the current input $x_t$ and the previous state $h_{t-1}$.

## Mathematical Formulation

The continuous-time SSM is defined as:
$$ h'(t) = A h(t) + B x(t) $$
$$ y(t) = C h(t) + D x(t) $$

In Mamba, these parameters are discretized using the "zero-order hold" (ZOH) method, which depends on a step size $\Delta$ (delta).

### Discretization
Given input-dependent step size $\Delta_t$:
$$ \bar{A}_t = \exp(\Delta_t A) $$
$$ \bar{B}_t = (\Delta_t A)^{-1} (\exp(\Delta_t A) - I) \cdot \Delta_t B \approx \Delta_t B $$

### Recurrence (The Scan)
The discrete recurrence is:
$$ h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t $$
$$ y_t = C_t h_t + D x_t $$

## Reference Implementation
Based on `vendor/mamba/mamba_ssm/ops/selective_scan_interface.py`, the python reference loop is:

```python
# Pre-compute discretized terms
deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)

# Recurrence Loop
x = 0  # Initial state
ys = []
for i in range(L):
    # State update: h_t = A_bar * h_{t-1} + B_bar * u_t
    x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
    
    # Output projection: y_t = C * h_t
    y = torch.einsum('bdn,dn->bd', x, C) 
    ys.append(y)
```

## Optimization Goal
The native Python loop (or even an unrolled ONNX graph) performs $L$ separate kernel launches (where $L$ is sequence length, often 2048+). 
Our goal is to fuse this entire loop into a single C++ kernel using AVX2 instructions (`_mm256_fmadd_ps`) to perform the `x = a*x + b` update efficiently in registers.
