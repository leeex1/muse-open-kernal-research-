#!/usr/bin/env python3
"""Full-model inference with DLL dispatcher vs patched-only (CPU)."""

import sys, time, torch

sys.path.insert(0, r"C:\02_QUILLAN\03 - Training & Model\scripts")
from quillan_v5_4_oni import QuillanRoninOni, QuillanOniConfig
import quillan_cuda_dispatcher as D

torch.manual_seed(42)
cfg = QuillanOniConfig()
model = QuillanRoninOni(cfg)
model.eval()
print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

D.install(model)

ids = torch.randint(0, cfg.vocab_size, (1, 32))


@torch.no_grad()
def bench_prefill(reps=3):
    for _ in range(1):
        _ = model(ids, use_cache=True, compute_aux=False)
    t0 = time.perf_counter()
    for _ in range(reps):
        _ = model(ids, use_cache=True, compute_aux=False)
    return (time.perf_counter() - t0) / reps * 1000


@torch.no_grad()
def bench_decode(steps=8):
    out = model(ids, use_cache=True, compute_aux=False)
    kv = out.past_key_values if hasattr(out, "past_key_values") else None
    nxt = torch.randint(0, cfg.vocab_size, (1, 1))
    t0 = time.perf_counter()
    for s in range(steps):
        out = model(nxt, past_key_values=kv, use_cache=True, compute_aux=False)
        kv = out.past_key_values if hasattr(out, "past_key_values") else kv
    return (time.perf_counter() - t0) / steps * 1000


p = bench_prefill()
d = bench_decode()
print(
    f"FULL MODEL (CPU+DLL): prefill {p:.1f} ms | decode {d:.1f} ms/tok | {1000 / d:.2f} tok/s"
)
print(
    f"Compare: CPU-only patched was prefill 860.2 ms | decode 684.6 ms/tok | 1.46 tok/s"
)


# parity spot check: argmax token match
@torch.no_grad()
def top_token():
    out = model(ids, use_cache=True, compute_aux=False)
    logits = out.logits if hasattr(out, "logits") else out[0]
    return int(logits[0, -1].argmax())


print(f"Top token: {top_token()}")
D.uninstall()
print("FULL BENCH DONE")
