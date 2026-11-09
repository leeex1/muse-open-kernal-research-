#!/usr/bin/env python3
"""Integration test: BitLinear patched with DLL dispatcher vs original, parity + speed.
Model runs on CPU (sm_61 PyTorch unsupported); dispatcher ships int8 to GPU internally.
"""

import sys, time, torch

sys.path.insert(0, r"C:\02_QUILLAN\03 - Training & Model\scripts")
from quillan_v5_4_oni import BitLinear, _weight_quant, _quantize_act
import torch.nn.functional as F
import quillan_cuda_dispatcher as D

torch.manual_seed(0)
layer = BitLinear(1024, 2048)
layer.eval()

x = torch.randn(1, 1024)
with torch.no_grad():
    w = _weight_quant(layer.weight)
    ref = F.linear(_quantize_act(x), w, layer.bias)

ok = D.install(layer)
if not ok:
    print("SKIP: DLL unavailable")
    sys.exit(0)

with torch.no_grad():
    out = layer(x)

err = (ref - out).abs().max().item() / ref.abs().max().item()
print(
    f"Parity: rel_err={err:.4f}  (ref [{ref.min():.2f},{ref.max():.2f}] out [{out.min():.2f},{out.max():.2f}])"
)
print("PARITY", "PASS" if err < 0.05 else "FAIL")

# Speed: decode M=1 — patched (GPU GEMV) vs CPU reference
with torch.no_grad():
    for _ in range(5):
        _ = layer(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        _ = layer(x)
    torch.cuda.synchronize()
    gpu_ms = (time.perf_counter() - t0) / 50 * 1000

    t0 = time.perf_counter()
    for _ in range(50):
        w2 = _weight_quant(layer.weight)
        F.linear(_quantize_act(x), w2, layer.bias)
    cpu_ms = (time.perf_counter() - t0) / 50 * 1000

print(
    f"Decode M=1: patched(GPU) {gpu_ms:.3f} ms vs CPU {cpu_ms:.3f} ms = {cpu_ms / gpu_ms:.1f}x"
)

# Prefill M=128 — GEMM v2 path
x128 = torch.randn(128, 1024)
with torch.no_grad():
    ref128 = F.linear(_quantize_act(x128), w, layer.bias)
    out128 = layer(x128)
err128 = (ref128 - out128).abs().max().item() / ref128.abs().max().item()
for _ in range(3):
    _ = layer(x128)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(20):
    _ = layer(x128)
torch.cuda.synchronize()
g128 = (time.perf_counter() - t0) / 20 * 1000
t0 = time.perf_counter()
for _ in range(20):
    F.linear(_quantize_act(x128), w, layer.bias)
c128 = (time.perf_counter() - t0) / 20 * 1000
print(
    f"Prefill M=128: parity {err128:.4f} | patched {g128:.3f} ms vs CPU {c128:.3f} ms = {c128 / g128:.1f}x"
)
D.uninstall()
print("INTEGRATION TEST DONE")
