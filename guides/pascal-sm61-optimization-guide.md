# Pascal (sm_61) Optimization Guide — Tips, Tricks & Sauce

Hard-won lessons from optimizing INT8 GEMM/GEMV on the GTX 1050 (GP107). Everything below is **measured**, not theory. Hardware: 5 SMs, 640 CUDA cores, ~112 GB/s DRAM, 1 MB L2, no tensor cores, no `cp.async`, no TMA, no wgmma.

---

## 1. 🚫 Never compute in FP16 on Pascal

**FP16 math on sm_61 runs at ~1/64 the FP32 rate** (consumer Pascal has no FP16 FMA units; it's emulated). Our v1 GEMM wrote FP16 outputs — wasteful.

```cuda
// ❌ WRONG on Pascal: slow + overflow risk
Y[row*N+col] = __float2half(acc * xs * ws);

// ✅ RIGHT: compute FP32, store FP32 (convert to FP16 later on CPU if needed)
Y[row*N+col] = (float)acc * xs * ws;
```

Rule of thumb: **store** whatever you want, **compute always in FP32** (or INT32 via `__dp4a`).

## 2. 🔢 `__dp4a` is your tensor core

Pascal has no tensor cores (they start at Volta sm_70). The fastest integer primitive is `__dp4a(int32 a, int32 b, int32 c)` — a 4-wide INT8 dot product + accumulate in **one instruction**. It's sm_61+ only (sm_60 doesn't have it).

```cuda
#include <sm_61_intrinsics.h>   // declares __dp4a
acc = __dp4a(a4[bm], b4[bn], acc);  // 4 int8 MACs, single cycle
```

Peak estimate: 640 cores × ~1.5 GHz × 4 MACs ≈ 3.8 TOPS INT8. cuBLAS INT8 on Pascal *also* lowers to dp4a — there's no tensor-core moat, so a well-tuned dp4a kernel matches or beats it on LLM shapes (M=1..128).

## 3. 📐 Vectorize your loads (`int4` = 16 int8s)

The single biggest win in our GEMM: scalar byte loads → `int4` loads. **3.2–5.0×** faster, identical math.

```cuda
// ❌ Scalar: 16 loads for 16 int8s, 75% bus waste
sX[r][kk] = X[g_row * K + g_k];

// ✅ Vectorized: 1 LDG.128 for 16 int8s
int4 v = *reinterpret_cast<const int4*>(X + g_row * K + g_k);
*reinterpret_cast<int4*>(&sX[stage][row][c4]) = v;
```

**Gotcha:** shared-memory `int4` stores must be 16-byte aligned. If you pad the row stride for bank conflicts, pad to a multiple of 16 (e.g. 48, not 36) or you get `misaligned address` crashes. We hit exactly this with stride 36.

## 4. 🏦 Bank conflicts: pad by 4

Shared memory has 32 banks × 4B. Unpadded power-of-2 strides → 8-way conflicts on column reads. Pad the K dimension: `BK + 4` (or XOR-swizzle). Combined with rule 3, use `+16` so int4 stays aligned.

## 5. 🔁 Double-buffering without cp.async

Pascal has no `cp.async` (Ampere+). Software-pipeline through registers instead:

```cuda
load_tile(0, 0);           // prologue
__syncthreads();
for (t = 0; t < kTiles; ++t) {
    if (t + 1 < kTiles) load_tile(stage ^ 1, ...);  // prefetch NEXT tile
    compute(stage);                                 // compute CURRENT tile
    __syncthreads();
}
```

Two stages max (register pressure doesn't allow 3–4 like Hopper). The compiler handles the rest if you structure `load → for { compute; load_next; }`.

## 6. ⚖️ Register blocking: 4×4 to 4×8 per thread

Each thread should own a small 2D tile of outputs so one loaded value feeds many `__dp4a`s:

| Thread tile | Acc regs | Verdict |
|---|---|---|
| 8×4 (32) | 32 | ✅ our v1/v2 choice |
| 4×4 (16) | 16 | ✅ safer occupancy, slightly more reloads |
| 8×8 (64) | 64 | ❌ spills on 5-SM parts |

Block: 128–256 threads, `__launch_bounds__(128|256)`. Target ≤60 regs/thread. **25–50% occupancy is fine** for GEMM — ILP (blocking + prefetch) matters more than warps/SM. But never below 2 warps/SM (can't hide smem latency).

## 7. 🎯 Persistent grids for a 5-SM GPU

GP107 has **5 SMs**. A grid of 6 tiles = 2 waves with wave 2 using 1/5 of the GPU → 60% idle. Options:

- **Data-parallel persistent:** launch exactly 5 CTAs, each loops `for (tile = blockIdx.x; tile < nTiles; tile += gridDim.x)`. Fixes most quantization when tiles >> SMs.
- **Split-K when parallelism-starved:** GEMV (M=1) or attention decode with tiny batch. Sweep splits 4–64 (phase-2 reduction grows linearly; max isn't best).
- **Rasterize for L2:** order tiles along the shorter of M/N so one operand stays in the tiny 1 MB L2.

## 8. 🧊 Ternary weights: 2 bits each, decode in registers

`{-1, 0, +1}` needs only `log2(3) ≈ 1.585` bits. BitNet's W2A8 layout packs **16 weights per uint32**, interleaved so one word supplies 4 byte-lanes at once:

```cuda
// One 32-bit load → 4 dp4a (16 weights)
uint32_t lane = (word >> 2*g) & 0x03030303;
lane = lane - 2;                       // {1,2,3} -> {-1,0,1}
acc = __dp4a(lane, x_word[g], acc);
```

Bandwidth: 4× vs INT8, 16× vs FP32. Requires `N%16==0 && K%32==0` — gate on it, fall back to INT8 for tails. (We didn't ship this — it needs a repacker — but it's the next win for decode.)

## 9. 🗜️ Cache static weights, stream dynamic activations

Weights don't change during eval. **Pack + quantize + transfer ONCE, then serve from cache.** Our dispatcher measured 16 ms/call when requantizing per call vs 0.4 ms cached — same kernel, 40× difference. Cache keyed on `weight._version`:

```python
if hit is not None and hit[version] == module.weight._version:
    return hit  # pre-packed int8 + scales, already on GPU
```

## 10. 🔌 Transfers can kill you (the PCIe tax)

Kernel at 0.1 ms + 2× PCIe round-trip + sync ≈ 1.5 ms/module. Across 110 sequential modules that's 164 ms/token — **slower than CPU** despite 38.8× module-level wins. Rules:

- Batch weight transfers (cache them on GPU — rule 9).
- Only dispatch when the GEMM is big enough that kernel time >> transfer overhead (we use M≥64 AND out×in ≥ 1M).
- Tiny GEMVs in a long sequential chain stay on CPU.

## 11. 📉 GEMV is bandwidth-bound — bytes are the win

For decode, you stream the whole weight matrix per token. Compute is irrelevant; **traffic is everything**:
- FP32 weights: 4 B/param
- INT8: 1 B/param (4× win)
- Ternary 2-bit: 0.25 B/param (16× win)

Our GEMV kernel already hits 15–20× vs CPU fp32. The remaining headroom is 2-bit packing (rule 8).

## 12. 🧰 Build without nvcc? Use NVRTC

If nvcc chokes on your MSVC headers (common: VS14.44 STL demands CUDA ≥12.4):
- Compile device code with **NVRTC** (`ptx_build.py` pattern): parse only CUDA headers, emit `compute_61` PTX, driver JITs to SASS at load.
- Launch via **CUDA driver API** (`cuModuleLoadDataEx` + `cuLaunchKernel`), not `torch.utils.cpp_extension`.
- Strip everything after `// Host-side launcher.` before NVRTC (it can't parse MSVC headers or `<<<>>>`).

## 13. 📊 Benchmark methodology (trust only this)

- Median of interleaved trials at sub-ms latencies (single-shot timings lie).
- Compare against the *actual* bottleneck (CPU fp32, v1 kernel, bandwidth roofline) — not "looks fast".
- **Rejected ideas stay in the repo as anti-examples** (`quillan_cuda_sm61_v2.cu`): split-K GEMV measured 0.59–1.0×. Negative results save the next person a week.

---

## Quick-reference tile configs

| Kernel | Block tile | Thread tile | BK | Threads | Notes |
|---|---|---|---|---|---|
| GEMM (prefill) | 128×64 | 8×4 | 32 | 256 | int4 loads, double-buffer, fp32 out |
| GEMV (decode) | 1 warp/row | 32-wide K | full K | 128 | stream W, warp-shuffle reduce |
| GEMM small-M | 64×64 | 4×4 | 32 | 128 | better tail behavior on 5 SMs |
