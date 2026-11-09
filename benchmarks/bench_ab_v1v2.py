#!/usr/bin/env python3
"""A/B benchmark: v1 GEMM vs v2 GEMM (vectorized + double-buffer + fp32)."""

import sys, time, torch

sys.path.insert(
    0, r"C:\02_QUILLAN\09 - Projects\projects\Chip design\quillan_sm61_kernels"
)
import quillan_sm61_qgemm as Q1
import quillan_sm61_qgemm_v2 as Q2

torch.manual_seed(1)


def prep(M, K, N):
    x = (torch.randn(M, K) * 2).float()
    w = (torch.randn(N, K) * 2).float()
    xs = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-6) / 127.0
    ws = w.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-6) / 127.0
    xi = torch.clamp((x / xs).round(), -128, 127).to(torch.int8)
    wi = torch.clamp((w / ws).round(), -128, 127).to(torch.int8)
    return xi.cuda(), wi.cuda(), xs.cuda(), ws.cuda(), x, w


def bench(fn, args, reps=50):
    for _ in range(5):
        _ = fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        _ = fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000


print(
    f"{'M':>5} {'K':>5} {'N':>5} {'v1_ms':>9} {'v2_ms':>9} {'speedup':>8} {'parity':>7}"
)
for M, K, N in [
    (32, 1024, 1024),
    (128, 1024, 1024),
    (128, 4096, 4096),
    (512, 1024, 4096),
    (512, 4096, 4096),
    (1024, 4096, 4096),
]:
    xi, wi, xs, ws, xf, wf = prep(M, K, N)
    y1 = Q1.qgemm_forward(xi, wi, xs, ws).cpu().float()
    y2 = Q2.qgemm_forward_v2(xi, wi, xs, ws).cpu()
    err = (y1 - y2).abs().max().item() / y1.abs().max().item()
    t1 = bench(Q1.qgemm_forward, (xi, wi, xs, ws))
    t2 = bench(Q2.qgemm_forward_v2, (xi, wi, xs, ws))
    print(
        f"{M:>5} {K:>5} {N:>5} {t1:>9.3f} {t2:>9.3f} {t1 / t2:>7.2f}x {err:>7.4f}",
        flush=True,
    )
print("AB BENCH DONE")
