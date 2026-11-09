#!/usr/bin/env python3
"""A/B benchmark: v1 GEMV vs v2 GEMV (split-K + smem X staging)."""

import sys, time, torch, ctypes
from pathlib import Path


def load_dll(name):
    dll = ctypes.CDLL(str(Path(r"C:\02_QUILLAN\kernels") / name))
    return dll


v1 = load_dll("quillan_cuda_sm61.dll")
v2 = load_dll("quillan_cuda_sm61_v2.dll")
print(f"SM: v1={v1.query_sm61_support()} v2={v2.query_sm61_support_v2()}")

for dll, fn in [(v1, "launch_bitnet_gemv_dp4a"), (v2, "launch_bitnet_gemv_v2")]:
    f = getattr(dll, fn)
    f.argtypes = [ctypes.c_void_p] * 4 + [
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    f.restype = None

torch.manual_seed(1)


def bench(fn, Wg, Xg, Yg, Rg, M, Kd4, reps=100):
    for _ in range(10):
        fn(Wg.data_ptr(), Xg.data_ptr(), Yg.data_ptr(), Rg.data_ptr(), 1.0, M, Kd4, 0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(Wg.data_ptr(), Xg.data_ptr(), Yg.data_ptr(), Rg.data_ptr(), 1.0, M, Kd4, 0)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000


print(f"{'M':>6} {'K':>5} {'v1_ms':>9} {'v2_ms':>9} {'speedup':>8} {'parity':>7}")
for M, K in [
    (1024, 8192),
    (2048, 8192),
    (4096, 8192),
    (50257, 1024),
    (1024, 4096),
    (8192, 1024),
]:
    Kd4 = K // 4
    W = torch.randint(-1, 2, (M, K), dtype=torch.int8)
    x = torch.randint(-128, 127, (K,), dtype=torch.int8)
    Wp = W.view(torch.int32).contiguous()
    Xp = x.view(torch.int32).contiguous()
    Y1 = torch.empty((M,), dtype=torch.float32)
    Y2 = torch.empty((M,), dtype=torch.float32)
    rs = torch.ones((M,), dtype=torch.float32)
    Wg, Xg, Rg = Wp.cuda(), Xp.cuda(), rs.cuda()
    Y1g, Y2g = Y1.cuda(), Y2.cuda()
    v1.launch_bitnet_gemv_dp4a(
        Wg.data_ptr(), Xg.data_ptr(), Y1g.data_ptr(), Rg.data_ptr(), 1.0, M, Kd4, 0
    )
    v2.launch_bitnet_gemv_v2(
        Wg.data_ptr(), Xg.data_ptr(), Y2g.data_ptr(), Rg.data_ptr(), 1.0, M, Kd4, 0
    )
    torch.cuda.synchronize()
    err = (Y1g.cpu() - Y2g.cpu()).abs().max().item()
    t1 = bench(v1.launch_bitnet_gemv_dp4a, Wg, Xg, Y1g, Rg, M, Kd4)
    t2 = bench(v2.launch_bitnet_gemv_v2, Wg, Xg, Y2g, Rg, M, Kd4)
    print(
        f"{M:>6} {K:>5} {t1:>9.3f} {t2:>9.3f} {t1 / t2:>7.2f}x {err:>7.1f}", flush=True
    )
print("GEMV AB DONE")
