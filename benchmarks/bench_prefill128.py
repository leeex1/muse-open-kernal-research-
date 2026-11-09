#!/usr/bin/env python3
"""Prefill T=128: CPU-only vs DLL-dispatched (GEMM v2 path active)."""

import sys, time, torch

sys.path.insert(0, r"C:\02_QUILLAN\03 - Training & Model\scripts")
from quillan_v5_4_oni import QuillanRoninOni, QuillanOniConfig
import quillan_cuda_dispatcher as D

torch.manual_seed(42)
cfg = QuillanOniConfig()
model = QuillanRoninOni(cfg).eval()
ids = torch.randint(0, cfg.vocab_size, (1, 128))


@torch.no_grad()
def bench(reps=3):
    for _ in range(1):
        _ = model(ids, use_cache=True, compute_aux=False)
    t0 = time.perf_counter()
    for _ in range(reps):
        _ = model(ids, use_cache=True, compute_aux=False)
    return (time.perf_counter() - t0) / reps * 1000


cpu_ms = bench()
print(f"CPU-only (patched):  prefill T=128 = {cpu_ms:.1f} ms")

D.install(model)
gpu_ms = bench()
print(f"DLL-dispatched:      prefill T=128 = {gpu_ms:.1f} ms")


# parity
@torch.no_grad()
def logits(m):
    out = m(ids, use_cache=True, compute_aux=False)
    return (out.logits if hasattr(out, "logits") else out[0]).cpu()


D.uninstall()
l_cpu = logits(model)
D.install(model)
l_gpu = logits(model)
err = (l_cpu - l_gpu).abs().max().item() / l_cpu.abs().max().item()
print(f"Parity: {err:.4f}")
print(f"PREFILL SPEEDUP: {cpu_ms / gpu_ms:.2f}x")
print("DONE")
