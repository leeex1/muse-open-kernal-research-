#!/usr/bin/env python3
"""A/B/C: v2 GEMM vs split-K GEMM vs CPU at skinny M (1..64)."""

import sys, time, torch

sys.path.insert(
    0, r"C:\02_QUILLAN\09 - Projects\projects\Chip design\quillan_sm61_kernels"
)
import quillan_sm61_qgemm_v2 as Q2
import quillan_sm61_gemm_sk as QK

torch.manual_seed(1)


def prep(M, K, N):
    x = (torch.randn(M, K) * 2).float()
    w = (torch.randn(N, K) * 2).float()
    xs = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-6) / 127.0
    ws = w.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-6) / 127.0
    xi = torch.clamp((x / xs).round(), -128, 127).to(torch.int8)
    wi = torch.clamp((w / ws).round(), -128, 127).to(torch.int8)
    return xi.cuda(), wi.cuda(), xs.cuda(), ws.cuda(), x, w


def bench(fn, args, reps=30):
    for _ in range(5):
        _ = fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        _ = fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000


print(
    f"{'M':>5} {'K':>5} {'N':>5} {'v2_ms':>8} {'sk_ms':>8} {'cpu_ms':>8} {'par':>7} {'sk/v2':>6}"
)
for M, K, N in [
    (1, 1024, 1024),
    (1, 4096, 1024),
    (8, 1024, 1024),
    (8, 4096, 4096),
    (32, 1024, 1024),
    (32, 4096, 4096),
    (64, 4096, 4096),
]:
    xi, wi, xs, ws, xf, wf = prep(M, K, N)
    y2 = Q2.qgemm_forward_v2(xi, wi, xs, ws).cpu()
    yk = QK.qgemm_forward_sk(xi, wi, xs, ws, nsplits=5).cpu()
    err = (y2 - yk).abs().max().item() / y2.abs().max().item()
    t2 = bench(Q2.qgemm_forward_v2, (xi, wi, xs, ws))
    tk_ = bench(QK.qgemm_forward_sk, (xi, wi, xs, ws))
    t0 = time.perf_counter()
    for _ in range(30):
        _ = torch.matmul(xf, wf.t())
    cpu = (time.perf_counter() - t0) / 30 * 1000
    print(
        f"{M:>5} {K:>5} {N:>5} {t2:>8.3f} {tk_:>8.3f} {cpu:>8.3f} {err:>7.4f} {t2 / tk_:>5.2f}x",
        flush=True,
    )
print("SK BENCH DONE")
