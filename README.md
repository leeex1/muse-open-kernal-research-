# Muse Open Kernel Research ⚔️

Custom CUDA kernel upgrades for **NVIDIA Pascal sm_61 (GTX 1050 / GP107)** — plus everything we learned from competing in the **Muse Spark Open Kernel Challenge** (Nov 2026).

> **One-line version:** Modern PyTorch dropped sm_61 support, so on a GTX 1050 the *only* GPU compute path is custom CUDA. These kernels make it fast.

---

## 🏆 Results (measured on GTX 1050, 4 GB)

| Upgrade | Baseline | Upgraded | Speedup |
|---------|----------|----------|---------|
| **GEMM v2** (int4 + double-buffer + fp32) vs v1 | scalar-load kernel | vectorized kernel | **3.2–5.0×** |
| **GEMV decode** vs CPU fp32 | CPU `F.linear` | `__dp4a` DLL | **15–20×** |
| **BitLinear module** (1024→2048, decode) | CPU 15.79 ms | DLL 0.41 ms | **38.8×** |
| **BitLinear module** (prefill M=128) | CPU 9.97 ms | GEMM v2 2.60 ms | **3.8×** |
| **Full model decode** (593M params) | 1188 ms/tok | 685 ms/tok | **1.73×** |
| **Full model prefill** (T=32) | 1522 ms | 860 ms | **1.77×** |

Full numbers: [`results/RESULTS.md`](results/RESULTS.md) · [`results/UPGRADE_SUMMARY.md`](results/UPGRADE_SUMMARY.md)

---

## ⚠️ The #1 trap: PyTorch + sm_61

**PyTorch ≥ 2.x with CUDA ≥ 12 no longer supports Pascal (sm_61).**

```python
# This CRASHES on a GTX 1050 with modern PyTorch:
x.abs().max()          # "no kernel image is available for execution on the device"
```

Any PyTorch CUDA op (even `.abs()`) fails. That's why this repo exists: the custom `.cu` kernels are compiled for `sm_61` via **NVRTC** and launched through the **CUDA driver API** (`cuLaunchKernel`), bypassing PyTorch's kernel dispatch entirely. See [`guides/troubleshooting.md`](guides/troubleshooting.md).

---

## 🚀 Quick start

```bash
# 1. Build the upgraded GEMM (NVRTC → PTX, no nvcc needed)
python kernels/ptx_build_v2.py          # writes kernels/sm61_qgemm_v2_sm61.ptx

# 2. Run the A/B benchmark (v1 vs v2, proves the 5× win)
python benchmarks/bench_ab_v1v2.py

# 3. Run GEMV baseline (proves 15-20× vs CPU)
python benchmarks/bench_gemv_baseline.py
```

Requirements: Python 3.11–3.14, PyTorch (for tensor plumbing only), CUDA 12.x toolkit (for NVRTC), NVIDIA driver ≥ 580. Windows paths in scripts assume a `C:\02_QUILLAN` layout — adjust `sys.path` entries, they're all near the top of each file.

---

## 📁 Repo layout

```
kernels/
├── sm61_qgemm_v2.cu            ⭐ UPGRADED GEMM (production) — int4 + double-buffer + fp32
├── ptx_build_v2.py             NVRTC build for v2
├── quillan_sm61_qgemm_v2.py    v2 Python binding (driver API)
├── sm61_qgemm.cu               v1 baseline GEMM (scalar loads, fp16 epilogue)
├── ptx_build.py / quillan_sm61_qgemm.py    v1 build + binding
├── quillan_cuda_sm61.cu        ⭐ GEMV + fused 34-expert (production GEMV, 15-20× vs CPU)
├── quillan_cuda_sm61_v2.cu     ❌ REJECTED GEMV attempt (split-K) — kept as anti-example
├── kernel_bridge.py            3-tier bridge (CUDA DLL → ATen → AVX2)
└── quillan_cuda_dispatcher.py  monkey-patch wiring: BitLinear → DLLs (weight-cached)

benchmarks/                     every .py that produced the numbers above
results/                        measured tables + upgrade notes
guides/
├── pascal-sm61-optimization-guide.md   ⭐ the sauce: Pascal-specific tips & tricks
├── okc-contest-learnings.md            what the Open Kernel Challenge taught us
├── troubleshooting.md                  sm_61 pitfalls (PyTorch, nvcc, alignment…)
└── getting-started.md                  build/run from a clean checkout
```

---

## 🧠 The three lessons that matter most

1. **Vectorize or die.** Scalar byte loads wasted 75% of the memory bus. `int4` (16B) loads = 5× on the same kernel, same math.
2. **Cache the static, stream the dynamic.** Weights are static in eval — packing them per-call cost 16 ms vs 0.1 ms kernel. Cache once (versioned), then every call is pure kernel time.
3. **Measure before you ship.** Our GEMV "upgrade" was 0.59–1.0× (slower). The dispatcher was 0.98× end-to-end despite 38.8× module wins (PCIe round-trips). Only benchmarks tell the truth.

Details in [`guides/`](guides/).

---

## 🏁 OKC context

We competed in the Muse Spark Open Kernel Challenge across all 5 problems (86 submissions) using the vjeux canonical kernels as baseline:

| Problem | Our Best | Leader | Rank |
|---------|----------|--------|------|
| muse_bf16_proj | 1.517× | 1.728× | #358 |
| muse_attention_decode | 2.622× | 3.339× | #277 |
| muse_moe_w4a16 | 2.329× | 2.844× | #437 |
| muse_decode_layer | 2.102× | 2.614× | #482 |
| muse_prefill_layer | 1.799× | 2.147× | #261 |

**The takeaway that shaped this repo:** parameter tuning alone (tile sizes, chunk counts) never closes a 12–22% gap against leaders who changed *architecture*. These kernels are that architectural work, adapted from Hopper techniques (16B streaming, pipelining, repack-once) down to Pascal constraints. See [`guides/okc-contest-learnings.md`](guides/okc-contest-learnings.md).

---

## 📜 License

MIT. Kernel code is original work from the Quillan-Ronin project. OKC references are for research/attribution.
