# Getting Started — Build & Run (GTX 1050 / sm_61)

## Prerequisites
- Windows + NVIDIA driver ≥ 580, GTX 1050 (or any sm_61 card)
- Python 3.11–3.14 + PyTorch 2.x (tensor plumbing; CUDA kernels not required!)
- CUDA 12.x toolkit (only for `nvrtc64_*.dll` — no nvcc needed for GEMM)

## Step 1: Build the upgraded GEMM (v2)

```powershell
python kernels/ptx_build_v2.py
# writes kernels/sm61_qgemm_v2_sm61.ptx
```

Uses NVRTC to compile `sm61_qgemm_v2.cu` → PTX. No nvcc, no MSVC dance. If NVRTC path fails, tell it where CUDA lives:
`$env:QUILLAN_CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"`

## Step 2: Build the GEMV DLL (optional — prebuilt `quillan_cuda_sm61.dll` ships in Quillan `kernels/`)

```powershell
# Needs nvcc 12.9 + VS2022 vcvars (see troubleshooting #2)
cmd /c build_v2_dll.bat   # produces quillan_cuda_sm61_v2.dll from quillan_cuda_sm61_v2.cu
```

## Step 3: Benchmark

```powershell
# GEMM A/B (v1 vs v2 — expect ~4-5x):
python benchmarks/bench_ab_v1v2.py

# GEMV vs CPU (expect 15-20x):
python benchmarks/bench_gemv_baseline.py

# Kernel sweep (GPU vs CPU across shapes):
python benchmarks/bench_kernel_baseline.py
```

All benchmarks print parity checks — if `parity > 0.01`, something's wrong; stop and investigate.

## Step 4: Use in your own model

```python
import sys
sys.path.insert(0, "kernels")
from quillan_sm61_qgemm_v2 import qgemm_forward_v2
import torch

# INT8 GEMM: Y[M,N] = X_int8[M,K] @ W_int8[N,K]^T, per-row/col scales
x_q  = quantize_int8(x)          # [M,K] on CUDA (bytes only, no PyTorch kernels needed)
w_q  = quantize_int8(W.t())      # [N,K]
y    = qgemm_forward_v2(x_q.cuda(), w_q.cuda(), x_s.cuda(), w_s.cuda())  # fp32
```

For BitNet-style ternary linears, see `kernels/quillan_cuda_dispatcher.py` (`D.install(model)` monkey-patches every `BitLinear`; decode falls back to CPU by design, prefill uses GEMM v2).

## Step 5: Verify the sm_61 environment

```python
import torch
print(torch.cuda.get_device_name(0))   # NVIDIA GeForce GTX 1050
x = torch.empty(1, device="cuda"); print("memcpy path OK")  # H2D/D2H works
# NOTE: torch compute ops (x.abs(), x@w) will CRASH on sm_61 with PyTorch >= 2.x.
# That's expected — the custom kernels are the GPU path.
```

## Repo map

- `kernels/` — production kernels (v2 GEMM, v1 GEMV) + bindings + dispatcher
- `benchmarks/` — every script behind the reported numbers
- `results/RESULTS.md` — the full measured tables (copy-pasted from runs)
- `results/UPGRADE_SUMMARY.md` — architecture notes + rejected experiments
- `guides/` — Pascal optimization guide, OKC learnings, troubleshooting
