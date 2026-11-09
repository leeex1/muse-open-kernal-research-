#!/usr/bin/env python3
"""Baseline inference benchmark for Quillan ONI on GTX 1050 Ti.
Measures: forward (prefill) + decode loop latency, with compute_aux on/off comparison.
Run BEFORE and AFTER upgrades to show speedup.
"""

import sys, time, torch

sys.path.insert(0, r"C:\02_QUILLAN\03 - Training & Model\scripts")
from quillan_v5_4_oni import QuillanRoninOni, QuillanOniConfig

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device} {torch.cuda.get_device_name(0) if device == 'cuda' else ''}")

cfg = QuillanOniConfig()
model = QuillanRoninOni(cfg).to(device)
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"Params: {n_params / 1e6:.1f}M")


@torch.no_grad()
def bench_forward(T=64, reps=5):
    ids = torch.randint(0, cfg.vocab_size, (1, T), device=device)
    # warmup
    for _ in range(2):
        _ = model(ids, use_cache=True)
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.perf_counter()
    for _ in range(reps):
        _ = model(ids, use_cache=True)
    torch.cuda.synchronize() if device == "cuda" else None
    return (time.perf_counter() - t0) / reps * 1000


@torch.no_grad()
def bench_decode(steps=16, reps=3):
    ids = torch.randint(0, cfg.vocab_size, (1, 32), device=device)
    # warmup prefill
    out = model(ids, use_cache=True)
    kv = out.past_key_values if hasattr(out, "past_key_values") else None
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.perf_counter()
    for r in range(reps):
        next_id = torch.randint(0, cfg.vocab_size, (1, 1), device=device)
        # reset KV each rep for fairness
        out = model(ids, use_cache=True)
        kv = out.past_key_values if hasattr(out, "past_key_values") else None
        for s in range(steps):
            out = model(next_id, past_key_values=kv, use_cache=True)
            kv = out.past_key_values if hasattr(out, "past_key_values") else kv
    torch.cuda.synchronize() if device == "cuda" else None
    return (time.perf_counter() - t0) / (reps * steps) * 1000


prefill_ms = bench_forward()
decode_ms = bench_decode()
print(
    f"RESULT prefill_T64: {prefill_ms:.2f} ms  | decode_tok: {decode_ms:.2f} ms  | tok/s: {1000 / decode_ms:.1f}"
)
print("BASELINE DONE")
