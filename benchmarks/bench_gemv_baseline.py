#!/usr/bin/env python3
"""Benchmark the BitNet GEMV kernel (quillan_cuda_sm61.dll) vs CPU and vs GEMM at M=1."""

import sys, time, torch, ctypes
from pathlib import Path

DLL = Path(r"C:\02_QUILLAN\kernels\quillan_cuda_sm61.dll")
print(f"DLL exists: {DLL.exists()}")
dll = ctypes.CDLL(str(DLL))
print(f"SM support: {dll.query_sm61_support()}")

# Set argtypes for GEMV launcher
dll.launch_bitnet_gemv_dp4a.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_float,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
]
dll.launch_bitnet_gemv_dp4a.restype = None

torch.manual_seed(1)


def bench_gemv(M, K, reps=100):
    """BitNet GEMV: W[M,K] ternary int8 @ x[K] int8 -> Y[M] fp32."""
    W = torch.randint(-1, 2, (M, K), dtype=torch.int8)
    x = torch.randint(-128, 127, (K,), dtype=torch.int8)
    Wp = W.view(torch.int32).contiguous()
    Xp = x.view(torch.int32).contiguous()
    Y = torch.empty((M,), dtype=torch.float32)
    rs = torch.ones((M,), dtype=torch.float32)
    Wg, Xg, Yg, Rg = Wp.cuda(), Xp.cuda(), Y.cuda(), rs.cuda()
    torch.cuda.synchronize()
    for _ in range(10):
        dll.launch_bitnet_gemv_dp4a(
            Wg.data_ptr(),
            Xg.data_ptr(),
            Yg.data_ptr(),
            Rg.data_ptr(),
            1.0,
            M,
            K // 4,
            0,
        )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        dll.launch_bitnet_gemv_dp4a(
            Wg.data_ptr(),
            Xg.data_ptr(),
            Yg.data_ptr(),
            Rg.data_ptr(),
            1.0,
            M,
            K // 4,
            0,
        )
    torch.cuda.synchronize()
    gemv_ms = (time.perf_counter() - t0) / reps * 1000
    # CPU fp32 reference
    Wf, xf = W.float(), x.float()
    t0 = time.perf_counter()
    for _ in range(reps):
        _ = torch.matmul(Wf, xf)
    cpu_ms = (time.perf_counter() - t0) / reps * 1000
    return gemv_ms, cpu_ms


print(f"{'M':>5} {'K':>5} {'GEMV_ms':>9} {'CPU_ms':>9} {'winner':>6}")
for M, K in [
    (1024, 8192),
    (2048, 8192),
    (4096, 8192),
    (50257, 1024),
    (1024, 4096),
    (2048, 4096),
    (8192, 1024),
    (8192, 2048),
]:
    try:
        g, c = bench_gemv(M, K)
        w = "GPU" if g < c else "CPU"
        print(
            f"{M:>5} {K:>5} {g:>9.3f} {c:>9.3f} {w:>6}  ({g / c:.2f}x cpu)", flush=True
        )
    except Exception as e:
        print(f"{M:>5} {K:>5} ERROR: {e}", flush=True)
print("GEMV BENCH DONE")
