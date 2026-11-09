# Results — All Measured Benchmarks (GTX 1050, sm_61)

Every number below came from a script in `benchmarks/`. Re-run any of them to reproduce. Hardware: GTX 1050 (GP107, 5 SMs, 640 CUDA cores, 4 GB, ~112 GB/s).

## 1. GEMM v1 vs v2 (`bench_ab_v1v2.py`) — the headline win

v2 = `int4` vectorized loads + register-prefetch double buffering + FP32 epilogue + bank padding. Parity <0.001 (fp32-vs-fp16 output diff only).

| M | K | N | v1 (ms) | v2 (ms) | Speedup |
|---|---|---|---------|---------|---------|
| 32 | 1024 | 1024 | 0.544 | 0.168 | **3.23×** |
| 128 | 1024 | 1024 | 0.786 | 0.158 | **4.98×** |
| 128 | 4096 | 4096 | 9.503 | 2.078 | **4.57×** |
| 512 | 1024 | 4096 | 7.559 | 1.839 | **4.11×** |
| 512 | 4096 | 4096 | 34.54 | 7.14 | **4.84×** |
| 1024 | 4096 | 4096 | 75.57 | 15.29 | **4.94×** |

## 2. GEMM v1 vs CPU fp32 (`bench_kernel_baseline.py`)

| M | K | N | GPU v1 (ms) | CPU (ms) | Winner |
|---|---|---|-------------|----------|--------|
| 1 | 1024 | 1024 | 0.462 | 0.078 | CPU (5.9× — TILE_M waste) |
| 1 | 4096 | 1024 | 1.945 | 0.601 | CPU (3.2× — use GEMV!) |
| 8 | 1024 | 1024 | 0.487 | 0.239 | CPU (2.0×) |
| 32 | 1024 | 1024 | 0.514 | 0.495 | CPU (1.04× — tie) |
| 128 | 1024 | 1024 | 0.609 | 1.663 | **GPU (2.7×)** |
| 128 | 4096 | 4096 | 8.967 | 25.13 | **GPU (2.8×)** |
| 512 | 1024 | 4096 | 7.600 | 22.66 | **GPU (3.0×)** |

Lesson: GEMM loses at M≤32 (fixed tile overhead); GEMV covers decode. After v2 (5×), GPU wins nearly everywhere.

## 3. GEMV vs CPU (`bench_gemv_baseline.py`) — decode engine

| M | K | GEMV (ms) | CPU (ms) | Speedup |
|---|---|-----------|----------|---------|
| 1024 | 8192 | 0.094 | 1.477 | **15.7×** |
| 2048 | 8192 | 0.185 | 3.732 | **20.2×** |
| 4096 | 8192 | 0.366 | 7.838 | **21.4×** |
| 50257 | 1024 | 0.528 | 11.15 | **21.1×** |
| 8192 | 2048 | 0.176 | 3.592 | **20.4×** |

## 4. GEMV v2 attempt — REJECTED (`bench_gemv_ab.py`)

Split-K + smem X staging + unrolled loop vs v1 GEMV. **0.59–1.00× (slower or tie).** Bandwidth-bound already; sync overhead hurt. v1 kept as production. Kept in repo as an anti-example.

## 5. BitLinear module integration (`test_dispatcher2.py`)

1024→2048, dispatcher with weight caching, parity 0.0000:

| Shape | Patched (DLL) | CPU | Speedup |
|-------|---------------|-----|---------|
| Decode M=1 | 0.407 ms | 15.79 ms | **38.8×** |
| Prefill M=128 | 2.596 ms | 9.968 ms | **3.8×** |

(Without weight caching: 0.6× / 0.2× — requantizing 2M weights per call was the killer.)

## 6. Full model, 593M params (`bench_full_model.py`, `bench_old_only.py`, `bench_prefill128.py`)

| Config | Prefill | Decode | tok/s |
|---|---|---|---|
| Unpatched (compute_aux on, swarm waste) | 1522 ms (T=32) | 1188 ms/tok | 0.84 |
| Python wins only (compute_aux off + swarm skip) | 860 ms | 685 ms/tok | **1.46** |
| + DLL dispatch (prefill path) | 904 ms (T=32) | 686 ms/tok | 1.46 |
| Prefill T=128, CPU-only | 1119 ms | — | — |
| Prefill T=128, DLL-dispatched | 1140 ms | — | parity 0.0000 |

**End-to-end verdicts:**
- Python wins: **1.73× decode / 1.77× prefill** ✅ shipped
- DLL dispatch: **0.98× end-to-end** (no win) ❌ — 110 sequential PCIe round-trips cancel kernel gains because the model must stay on CPU (PyTorch GPU broken on sm_61). Kernels remain staged for GPU-resident runtimes.

## 7. OKC standings (Nov 2026 challenge, 86 submissions)

| Problem | Ours | Leader | Rank |
|---------|------|--------|------|
| muse_bf16_proj | 1.517× | 1.728× | #358 |
| muse_attention_decode | 2.622× | 3.339× | #277 |
| muse_moe_w4a16 | 2.329× | 2.844× | #437 |
| muse_decode_layer | 2.102× | 2.614× | #482 |
| muse_prefill_layer | 1.799× | 2.147× | #261 |

Parameter tuning alone never closed the 12–22% gap — see `guides/okc-contest-learnings.md`.
