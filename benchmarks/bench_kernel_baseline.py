#!/usr/bin/env python3
"""Kernel-level benchmark: current sm_61 GEMM vs upgrade targets.
Pure CUDA kernel timing (no PyTorch CUDA ops - direct DLL calls).
"""

import sys, time, torch

sys.path.insert(
    0, r"C:\02_QUILLAN\09 - Projects\projects\Chip design\quillan_sm61_kernels"
)
import quillan_sm61_qgemm as Q

torch.manual_seed(1)


def bench_kernel(M, K, N, reps=50):
    """INT8 GEMM: Y = X[M,K] @ W[K,N]^T, W pre-transposed to [N,K] as kernel expects."""
    x = (torch.randn(M, K) * 2).float()
    w = (torch.randn(N, K) * 2).float()
    xs = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-6) / 127.0
    ws = w.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-6) / 127.0
    xi = torch.clamp((x / xs).round(), -128, 127).to(torch.int8)
    wi = torch.clamp((w / ws).round(), -128, 127).to(torch.int8)
    xg, wg, xsg, wsg = xi.cuda(), wi.cuda(), xs.cuda(), ws.cuda()
    torch.cuda.synchronize()
    # warmup
    for _ in range(5):
        _ = Q.qgemm_forward(xg, wg, xsg, wsg)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        _ = Q.qgemm_forward(xg, wg, xsg, wsg)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / reps * 1000
    gflops = 2 * M * K * N / (ms * 1e6)  # device ms -> TFLOPS (int8)
    # CPU fp32 reference
    t0 = time.perf_counter()
    for _ in range(reps):
        _ = torch.matmul(x, w.t())
    cpu_ms = (time.perf_counter() - t0) / reps * 1000
    return ms, cpu_ms, gflops


print(
    f"{'M':>5} {'K':>5} {'N':>5} {'GPU_ms':>9} {'CPU_ms':>9} {'TOPS':>7} {'winner':>6}"
)
for M, K, N in [
    (1, 1024, 1024),
    (1, 4096, 1024),
    (1, 8192, 1024),
    (8, 1024, 1024),
    (8, 4096, 4096),
    (32, 1024, 1024),
    (32, 4096, 4096),
    (128, 1024, 1024),
    (128, 4096, 4096),
    (512, 1024, 4096),
]:
    g, c, t = bench_kernel(M, K, N)
    w = "GPU" if g < c else "CPU"
    print(
        f"{M:>5} {K:>5} {N:>5} {g:>9.3f} {c:>9.3f} {t:>7.2f} {w:>6}  ({g / c:.2f}x cpu)",
        flush=True,
    )
print("KERNEL BENCH DONE")
