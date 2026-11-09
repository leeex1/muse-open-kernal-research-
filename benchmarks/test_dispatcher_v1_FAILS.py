#!/usr/bin/env python3
"""Integration test: BitLinear patched with DLL dispatcher vs original, parity + speed."""

import sys, time, torch

sys.path.insert(0, r"C:\02_QUILLAN\03 - Training & Model\scripts")
from quillan_v5_4_oni import BitLinear, _weight_quant, _quantize_act
import torch.nn.functional as F
import quillan_cuda_dispatcher as D

torch.manual_seed(0)
layer = BitLinear(1024, 2048)
layer.eval()

# Reference (original STE path, CPU)
x = torch.randn(1, 1024)
with torch.no_grad():
    w = _weight_quant(layer.weight)
    x_q = _quantize_act(x)
    ref = F.linear(x_q, w, layer.bias)

# Patched (DLL dispatch). Needs CUDA tensors.
ok = D.install(layer)
if not ok:
    print("SKIP: DLL unavailable")
    sys.exit(0)

x_g = x.cuda()
with torch.no_grad():
    out = layer(x_g).cpu()

err = (ref - out).abs().max().item() / ref.abs().max().item()
print(
    f"Parity: rel_err={err:.4f}  (ref range [{ref.min():.2f},{ref.max():.2f}] out [{out.min():.2f},{out.max():.2f}])"
)
print("PARITY", "PASS" if err < 0.05 else "FAIL")

# Speed: decode GEMV path (M=1)
for _ in range(5):
    _ = layer(x_g)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(100):
    _ = layer(x_g)
torch.cuda.synchronize()
gpu_ms = (time.perf_counter() - t0) / 100 * 1000

# CPU reference timing
t0 = time.perf_counter()
for _ in range(100):
    with torch.no_grad():
        w2 = _weight_quant(layer.weight)
        F.linear(_quantize_act(x), w2, layer.bias)
cpu_ms = (time.perf_counter() - t0) / 100 * 1000
print(
    f"Decode M=1: GPU {gpu_ms:.3f} ms vs CPU {cpu_ms:.3f} ms = {cpu_ms / gpu_ms:.1f}x speedup"
)
D.uninstall()
print("INTEGRATION TEST DONE")
