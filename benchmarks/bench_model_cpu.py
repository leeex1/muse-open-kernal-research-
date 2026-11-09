#!/usr/bin/env python3
"""Model-level benchmark (CPU) — measures Python quick wins.
compares patched vs unpatched via git-stash-free A/B: loads backup file as module."""

import sys, time, torch, importlib.util


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


SCRIPTS = r"C:\02_QUILLAN\03 - Training & Model\scripts"
sys.path.insert(0, SCRIPTS)

torch.manual_seed(42)
cfg_path = None
mod_new = load_mod(SCRIPTS + r"\quillan_v5_4_oni.py", "oni_new")
cfg = mod_new.QuillanOniConfig()
model = mod_new.QuillanRoninOni(cfg)
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"Params: {n_params / 1e6:.1f}M  device: cpu")

T, reps = 32, 3
ids = torch.randint(0, cfg.vocab_size, (1, T))


@torch.no_grad()
def bench_prefill(model, ids, reps):
    for _ in range(1):
        _ = model(ids, use_cache=True)
    t0 = time.perf_counter()
    for _ in range(reps):
        _ = model(ids, use_cache=True)
    return (time.perf_counter() - t0) / reps * 1000


@torch.no_grad()
def bench_decode(model, reps_prefill=1, steps=8):
    out = model(ids, use_cache=True)
    kv = out.past_key_values if hasattr(out, "past_key_values") else None
    nxt = torch.randint(0, cfg.vocab_size, (1, 1))
    t0 = time.perf_counter()
    for s in range(steps):
        out = model(nxt, past_key_values=kv, use_cache=True)
        kv = out.past_key_values if hasattr(out, "past_key_values") else kv
    return (time.perf_counter() - t0) / steps * 1000


p = bench_prefill(model, ids, reps)
d = bench_decode(model)
print(
    f"RESULT prefill_T32: {p:.1f} ms | decode_tok: {d:.1f} ms | tok/s: {1000 / d:.2f}"
)
print("BENCH DONE")
