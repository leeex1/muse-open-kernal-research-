#!/usr/bin/env python3
"""Interleaved A/B of compute_aux flag on the SAME (patched) model file.
This isolates the only Python change that matters in forward()."""

import sys, time, torch

sys.path.insert(0, r"C:\02_QUILLAN\03 - Training & Model\scripts")
from quillan_v5_4_oni import QuillanRoninOni, QuillanOniConfig

torch.manual_seed(42)
cfg = QuillanOniConfig()
model = QuillanRoninOni(cfg).eval()
ids = torch.randint(0, cfg.vocab_size, (1, 32))


@torch.no_grad()
def bench_prefill(compute_aux, reps=2):
    for _ in range(1):
        _ = model(ids, use_cache=True, compute_aux=compute_aux)
    t0 = time.perf_counter()
    for _ in range(reps):
        _ = model(ids, use_cache=True, compute_aux=compute_aux)
    return (time.perf_counter() - t0) / reps * 1000


@torch.no_grad()
def bench_decode(compute_aux, steps=6):
    out = model(ids, use_cache=True, compute_aux=compute_aux)
    kv = out.past_key_values if hasattr(out, "past_key_values") else None
    nxt = torch.randint(0, cfg.vocab_size, (1, 1))
    t0 = time.perf_counter()
    for s in range(steps):
        out = model(nxt, past_key_values=kv, use_cache=True, compute_aux=compute_aux)
        kv = out.past_key_values if hasattr(out, "past_key_values") else kv
    return (time.perf_counter() - t0) / steps * 1000


on, off = [], []
for r in range(3):
    on.append(bench_prefill(True))
    off.append(bench_prefill(False))
    print(f"prefill r{r}: aux=ON {on[-1]:.1f}  aux=OFF {off[-1]:.1f}", flush=True)
op, fp = sum(on) / 3, sum(off) / 3
print(f"AVG prefill: aux=ON {op:.1f} | aux=OFF {fp:.1f} | {op / fp:.2f}x")

don, doff = [], []
for r in range(3):
    don.append(bench_decode(True))
    doff.append(bench_decode(False))
    print(f"decode r{r}: aux=ON {don[-1]:.1f}  aux=OFF {doff[-1]:.1f}", flush=True)
dp, dfp = sum(don) / 3, sum(doff) / 3
print(f"AVG decode: aux=ON {dp:.1f} | aux=OFF {dfp:.1f} | {dp / dfp:.2f}x")

# Parity: top token with aux on vs off
with torch.no_grad():

    def top(ca):
        out = model(ids, use_cache=True, compute_aux=ca)
        logits = out.logits if hasattr(out, "logits") else out[0]
        return int(logits[0, -1].argmax())

    print(f"Top token aux=ON {top(True)}  aux=OFF {top(False)}")
print("DONE")
