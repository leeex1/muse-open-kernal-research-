#!/usr/bin/env python3
"""Old (unpatched) model bench only — for A/B vs patched numbers."""

import sys, time, torch, importlib.util


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


sys.path.insert(0, r"C:\02_QUILLAN\03 - Training & Model\scripts")
torch.manual_seed(42)
mod_old = load_mod(r"C:\Users\Admin\oni_old.py", "oni_old")
cfg = mod_old.QuillanOniConfig()
model = mod_old.QuillanRoninOni(cfg)
model.eval()

ids = torch.randint(0, cfg.vocab_size, (1, 32))


@torch.no_grad()
def bench_prefill():
    for _ in range(1):
        _ = model(ids, use_cache=True)
    t0 = time.perf_counter()
    for _ in range(3):
        _ = model(ids, use_cache=True)
    return (time.perf_counter() - t0) / 3 * 1000


@torch.no_grad()
def bench_decode(steps=8):
    out = model(ids, use_cache=True)
    kv = out.past_key_values if hasattr(out, "past_key_values") else None
    nxt = torch.randint(0, cfg.vocab_size, (1, 1))
    t0 = time.perf_counter()
    for s in range(steps):
        out = model(nxt, past_key_values=kv, use_cache=True)
        kv = out.past_key_values if hasattr(out, "past_key_values") else kv
    return (time.perf_counter() - t0) / steps * 1000


p = bench_prefill()
d = bench_decode()
print(f"OLD: prefill {p:.1f} ms | decode {d:.1f} ms/tok | {1000 / d:.2f} tok/s")
print(f"NEW: prefill 860.2 ms | decode 684.6 ms/tok | 1.46 tok/s")
print(f"SPEEDUP: prefill {p / 860.2:.2f}x | decode {d / 684.6:.2f}x")
print("DONE")
