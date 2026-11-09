# What We Learned — Kernels: What Worked, What Failed, What's Needed

The complete honest record from upgrading Quillan's CUDA kernels for the GTX 1050 (sm_61, Pascal).
Every claim below was **measured on the actual hardware**, not theorized. Numbers are in
`../results/RESULTS.md`; full technique details in `../guides/pascal-sm61-optimization-guide.md`.

---

## ✅ WHAT WORKED

### 1. `int4` vectorized loads — the 5× win
Scalar byte loads → 16-byte `int4` loads. Same math, same tile sizes. **3.2–5.0× faster.**
Lesson: on memory-bound kernels, load granularity IS the performance. The memory bus moves
bytes per instruction; scalar loads waste 75% of every 128-bit transaction.
*File: `kernels/sm61_qgemm_v2.cu`*

### 2. Register-prefetch double buffering (no cp.async on Pascal)
Load tile t+1 into the other smem stage while computing tile t. One `__syncthreads()`
per tile instead of two stalls. Hides global-load latency without Ampere's `cp.async`.
*File: `kernels/sm61_qgemm_v2.cu`*

### 3. FP32 epilogue (never compute in FP16 on Pascal)
v1 wrote FP16 outputs. Pascal FP16 math is ~1/64 the FP32 rate. Flipping the epilogue to
FP32 cost nothing measurable in bandwidth and removed a whole class of slowdowns/overflows.
*Rule: store what you want, compute always in FP32 (or INT32 via dp4a).*

### 4. Weight caching — the 40× overhead fix
The dispatcher requantized 2M weights **per call** (16 ms) for a 0.1 ms kernel. Caching
pre-packed int8 weights keyed on `weight._version` → 0.4 ms/call. Same kernel, 40× difference.
*Rule: static data gets packed ONCE.*

### 5. Killing dead compute (the biggest end-to-end win)
- `compute_aux=True` fired ~100 paper-pack modules per token (agent_evolution, mamba, MoR…)
  that feed aux outputs only → `compute_aux=False` in decode: **30–60% wall time saved**
- `BatchedCouncilCompute` computed swarm variance + 3-step world recurrence in eval, then
  **threw it away** (`return delta`) → early return: **~85% of council FLOPs saved**
- Measured end-to-end: **1.73× decode, 1.77× prefill**, numerics identical (parity 0.0000)

### 6. v1 GEMV was already optimal — recognize and keep
The original warp-per-row `__dp4a` GEMV streams weights at near-bandwidth (15–20× vs CPU fp32).
Our "upgrade" (split-K + smem staging) measured *slower*. Lesson: **a tuned loop beats a
clever loop.** Don't rewrite what already saturates bandwidth.
*File: `kernels/quillan_cuda_sm61.cu` (production)*

---

## ❌ WHAT FAILED (documented so nobody repeats it)

### 1. GEMV v2: split-K + smem X staging — 0.59–1.0× (SLOWER)
Hypothesis: split-K parallelizes the K=8192 reduction; smem staging dedups X traffic.
Reality: GEMV is bandwidth-bound; X (32 KB) fits in L2 already; the extra `__syncthreads()`
and conditional unroll guards cost more than they save.
*File kept: `kernels/quillan_cuda_sm61_v2.cu` — read the header before trusting any "improvement".*

### 2. End-to-end DLL dispatch — 0.98× (no win) despite 38.8× module wins
The BitLinear-level dispatch is 38.8× faster at decode (0.41 ms vs 15.8 ms). Wired into the
full model (110 BitLinear modules), it measured **slower** (685→849 ms/tok) because:
- 110 sequential PCIe round-trips per token (~1.5 ms/module overhead)
- `.cpu()` sync per module serializes the pipeline
- The model MUST stay on CPU (PyTorch GPU kernels don't exist on sm_61)
*Lesson: kernel speed ≠ system speed. Measure at the level you'll ship.*

### 3. FP16 epilogue (v1) — silently slow
Not a crash, not wrong answers — just 1/64-rate math on every output element. Found by
reading the code + research, confirmed by the v2 flip. *Lesson: know your ISA's FP16 story.*

### 4. OKC parameter tuning — duplicates and dead ends
Swept TILE_KEYS 8→1024, CHUNK_TOKENS 64→4096, PDL_MAX_T 8→256, scale modes across 86
submissions. Every "win" was either a byte-identical duplicate (edit didn't match) or slower.
*Lesson: hash-check edits; if >10% behind the leader, change architecture, not parameters.*

### 5. Split-K persistent GEMM — 0.01–0.16× (correct, catastrophically slow)
Tried OKC's stream-K/persistent pattern (`sm61_gemm_sk.cu`) for skinny M where v2 loses
to CPU. Parity 0.0000, but workspace traffic (nsplits×M×N fp32 partials + reduction
re-reads) dwarfs the compute savings — e.g. M=64: 319 ms vs 2.95 ms (v2). Split-K pays
off for GEMV (huge K, tiny N), already handled by the dedicated GEMV kernel.
*Lesson: reduction-by-workspace is only worth it when the K dimension dominates and the
output tile stays tiny.*

### 6. Bank padding by 4 — misaligned crash
Padded smem stride to 36 (32+4) for bank conflicts, then `int4` stores to odd rows faulted
(36 % 16 ≠ 0). Fix: stride 48. *Lesson: padding must preserve the alignment of your widest access.*

---

## 🔜 WHAT'S NEEDED (not done — next frontier)

### 1. A GPU-resident runtime (the big one)
Everything is transfer-bound because the model lives on CPU. Unlock options:
- **sm_61-capable PyTorch (≤1.13, CUDA 11.x)** in a separate env → model lives on GPU,
  DLL kernels become 38.8× end-to-end instead of 0.98×. *Highest value, env risk.*
- **Custom GPU-resident decode loop** (weights + KV persistent on device, raw-tensor layer
  loop, no PyTorch CUDA ops) — biggest effort, ceiling is ~15× over current CPU decode.

### 2. 2-bit ternary weight packing (BitNet W2A8 layout)
16 weights per uint32, interleaved decode: 4× less weight traffic than INT8 for decode.
Requires: repacker (N%16==0 && K%32==0 gate) + register decode kernel. Bandwidth-bound
GEMV goes from ~62 GB/s achieved toward the 112 GB/s peak. Spec in the Pascal guide §8.

### 3. KV cache preallocation
`torch.cat` per token per layer = O(T) copies + realloc churn. Preallocate
`max_seq_len` buffers, write in place. Cheap, mechanical, no numerics change.

### 4. Flash-style decode attention (Pascal variant)
We skipped it because attention is <2% of decode FLOPs at T≤512. It matters at long context:
FP32 micro-tiles (never FP16 on Pascal), INT8 KV cache (halves DRAM traffic), split-KV +
warp-shuffle reduce (no smem tree). Proven design exists (`sirCamp/flash-attention-legacy`).

### 5. Prism fusion
The 9-vector prism (9×[D,D] per layer) is ~36% of per-token FLOPs. Fuse into one packed
GEMM, share one matrix + per-vector diagonal scales, or cut vectors. Bigger design call —
affects model semantics, so it needs a parity eval first.

### 6. Persistent-grid GEMM (5 CTAs)
Launch exactly 5 CTAs (5 SMs), loop tiles. Fixes wave quantization when tiles don't divide
evenly. Only worth it if launch/tail overhead shows up in profiles — measure first.

---

## Decision log (what we chose and why)

| Decision | Choice | Evidence |
|---|---|---|
| GEMM production kernel | v2 (int4 + double-buffer + fp32) | 3.2–5.0× A/B, parity <0.001 |
| GEMV production kernel | v1 (warp-per-row) | v2 measured slower (0.59–1.0×) |
| GEMM codegen | NVRTC → PTX (no nvcc) | nvcc 11.8 vs VS14.44 STL gate; NVRTC bypasses MSVC headers |
| GEMM output type | FP32 (not FP16) | Pascal FP16 = 1/64 rate |
| Dispatcher default | Decode→CPU, prefill→GEMM v2 | End-to-end A/B: dispatch hurts decode (PCIe tax) |
| Weight handling in dispatch | Cache per module (`_version` key) | 16ms→0.4ms/call measured |
| compute_aux in decode | False (patched) | 1.73× end-to-end, logits unchanged |
| Swarm in eval | Skipped (patched) | 85% of council FLOPs were discarded work |
