#!/usr/bin/env python3
"""Controlled interleaved A/B: backup vs current file, same process, alternating."""

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
mod_new = load_mod(
    r"C:\02_QUILLAN\03 - Training & Model\scripts\quillan_v5_4_oni.py", "oni_new"
)
cfg = mod_old.QuillanOniConfig()
model_old = mod_old.QuillanRoninOni(cfg).eval()
model_new = mod_new.QuillanRoninOni(cfg).eval()
model_new.load_state_dict(model_old.state_dict())

ids = torch.randint(0, cfg.vocab_size, (1, 32))


@torch.no_grad()
def bench_prefill(model, reps=2):
    for _ in range(1):
        _ = model(ids, use_cache=True)
    t0 = time.perf_counter()
    for _ in range(reps):
        _ = model(ids, use_cache=True)
    return (time.perf_counter() - t0) / reps * 1000


# Interleaved: alternate old/new across 3 rounds, average
old_p, new_p = [], []
for r in range(3):
    old_p.append(bench_prefill(model_old))
    new_p.append(bench_prefill(model_new))
    print(f"round {r}: old={old_p[-1]:.1f} new={new_p[-1]:.1f}", flush=True)

op = sum(old_p) / len(old_p)
np_ = sum(new_p) / len(new_p)
print(f"AVG prefill: OLD {op:.1f} ms | NEW {np_:.1f} ms | {op / np_:.2f}x")


# Parity spot-check (same weights, same input)
@torch.no_grad()
def top(model):
    out = model(ids, use_cache=True)
    logits = out.logits if hasattr(out, "logits") else out[0]
    return int(logits[0, -1].argmax())


print(
    f"Top token old={top(model_old)} new={top(model_new)} (should match for swarm-only diff)"
)
print("DONE")
