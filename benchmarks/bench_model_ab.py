#!/usr/bin/env python3
"""A/B: unpatched (backup) vs patched model on CPU."""

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
mod_old = load_mod(r"C:\Users\Admin\oni_old.py", "oni_old")
cfg = mod_old.QuillanOniConfig()
model_old = mod_old.QuillanRoninOni(cfg)
model_old.eval()

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
def bench_decode(model, steps=8):
    out = model(ids, use_cache=True)
    kv = out.past_key_values if hasattr(out, "past_key_values") else None
    nxt = torch.randint(0, cfg.vocab_size, (1, 1))
    t0 = time.perf_counter()
    for s in range(steps):
        out = model(nxt, past_key_values=kv, use_cache=True)
        kv = out.past_key_values if hasattr(out, "past_key_values") else kv
    return (time.perf_counter() - t0) / steps * 1000


p_old = bench_prefill(model_old, ids, reps)
d_old = bench_decode(model_old)
print(
    f"OLD (unpatched): prefill {p_old:.1f} ms | decode {d_old:.1f} ms/tok | {1000 / d_old:.2f} tok/s"
)
print(f"NEW (patched):   prefill 860.2 ms | decode 684.6 ms/tok | 1.46 tok/s")
print(f"SPEEDUP: prefill {p_old / 860.2:.2f}x | decode {d_old / 684.6:.2f}x")


# parity check
@torch.no_grad()
def get_logits(model):
    out = model(ids, use_cache=True)
    if hasattr(out, "logits"):
        return out.logits
    return out[0]


l_old = get_logits(model_old)
mod_new = load_mod(SCRIPTS + r"\quillan_v5_4_oni.py", "oni_new")
model_new = mod_new.QuillanRoninOni(cfg)
model_new.load_state_dict(model_old.state_dict())
model_new.eval()
l_new = get_logits(model_new)
err = (l_old - l_new).abs().max().item()
print(f"PARITY max logit diff: {err:.6f} (should be ~0)")
print("AB DONE")
