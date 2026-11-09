# Quillan CUDA Kernel Upgrades — GTX 1050 Ti (sm_61) Inference Speed
## Date: 2026-11-06 | Author: Quillan (C10-CODEWEAVER + C13-WARDEN + C26-TECHNE)

## Context
PyTorch 2.12 + CUDA 13.0 dropped sm_61 support entirely. The GTX 1050 **cannot
run any PyTorch CUDA kernel** ("no kernel image available"). The custom sm_61
DLLs are the ONLY GPU compute path on this machine.

## Measured Baseline (before upgrades)
| Path | Result |
|------|--------|
| Model decode (CPU, unpatched) | 1187.6 ms/tok (0.84 tok/s) |
| Model prefill T=32 (CPU, unpatched) | 1522.4 ms |
| GEMM M=1 (GPU v1) | 0.462 ms (5.9x SLOWER than CPU!) |
| GEMV decode (GPU v1) | 15-20x vs CPU, but DEAD CODE (never called) |

## Upgrades Landed

### 1. Python quick wins (quillan_v5_4_oni.py) — CORRECTED 2026-11-09
- `compute_aux=False` at both decode call sites (generate prefill + loop):
  skips ~100 paper aux modules per forward (agent_evolution, mamba, moR, MoD...)
- `BatchedCouncilCompute` eval short-circuit: swarm variance + 3-step world
  recurrence computed then discarded in eval → return `delta` immediately.

**CORRECTION (interleaved A/B, same process — see benchmarks/bench_aux_ab.py):**
- Swarm short-circuit alone: **0.99x** (no measurable win; output-identical by
  construction, kept as dead-code elimination only).
- `compute_aux=False`: **1.21x prefill / 1.12x decode** (6/6 rounds OFF faster).
- ⚠️ **compute_aux=False CHANGES OUTPUTS** (top token 16165 vs 37770). The aux
  blocks modify `x`, not just aux outputs. This is an opt-in BEHAVIOR CHANGE
  for speed, NOT a free win. The old "1.73x, numerics unchanged" figures were
  CPU-drift artifacts (same-model rounds varied 1067→1580 ms in one process).

### 2. GEMM v2 kernel (sm61_qgemm_v2.cu) — NEW
File: `09 - Projects/projects/Chip design/quillan_sm61_kernels/sm61_qgemm_v2.cu`
- `int4` (16B) vectorized global→smem loads (was scalar byte loads)
- Register-prefetch double buffering (2 smem stages, overlap load/compute)
- FP32 epilogue (v1 wrote FP16 — Pascal FP16 is ~1/64 rate!)
- Bank-conflict padded smem (48B stride, 16B aligned)
- `__launch_bounds__(256)` occupancy hint

**Measured A/B vs v1 (parity <0.001):**
| Shape | v1 | v2 | Speedup |
|-------|----|----|---------|
| 32×1024×1024 | 0.544 | 0.168 | **3.23x** |
| 128×1024×1024 | 0.786 | 0.158 | **4.98x** |
| 128×4096×4096 | 9.503 | 2.078 | **4.57x** |
| 512×1024×4096 | 7.559 | 1.839 | **4.11x** |
| 512×4096×4096 | 34.54 | 7.14 | **4.84x** |
| 1024×4096×4096 | 75.57 | 15.29 | **4.94x** |

Build: `ptx_build_v2.py` (NVRTC, compute_61). Binding: `quillan_sm61_qgemm_v2.py`.

### 3. GEMV — v2 ATTEMPTED, REJECTED (measured slower)
File: `kernels/quillan_cuda_sm61_v2.cu` (kept for reference, NOT production)
- Split-K + smem X staging + unrolled loop: **0.59-1.00x vs v1** (slower!)
- Reason: GEMV is bandwidth-bound; X fits in L2; extra syncs hurt.
- **Production GEMV = v1** (`quillan_cuda_sm61.dll`), already 15-20x vs CPU.

### 3b. Split-K persistent GEMM — REJECTED (correct, catastrophically slow)
File: `kernels/sm61_gemm_sk.cu` (kept for reference, NOT production)
- Persistent grid (5 CTAs) + K-split + workspace reduction for skinny M:
  **0.01-0.16x vs v2** (parity 0.0000, so correct but useless here).
- Reason: workspace traffic (nsplits×M×N fp32 partials + reduction re-reads)
  dwarfs compute savings. Split-K pays off for GEMV (huge K, tiny N) only.
- See `benchmarks/bench_sk.py` + `results/RESULTS.md` §2.

### 4. DLL dispatcher (quillan_cuda_dispatcher.py) — NEW
Monkey-patches `BitLinear.forward`: prefill (M≥64, ≥1M elems) → GEMM v2;
decode (M<64) → CPU STE fallback (measured: 110 sequential PCIe round-trips
per token cost ~1.5ms/module, exceeding sub-ms GEMV time). Quant on CPU
(no PyTorch CUDA kernels exist on sm_61), weights pre-packed + cached per
module (key insight: per-call requant was 16ms vs 0.1ms kernel). Parity 0.0000.

**Measured at BitLinear level (1024→2048):**
- Decode M=1: 0.407ms vs 15.788ms CPU = **38.8x**
- Prefill M=128: 2.596ms vs 9.968ms CPU = **3.8x**

**Measured end-to-end (full 593M model, T=128 prefill): 0.98x — NO WIN.**
110 sequential round-trips cancel the kernel gains. The kernels are proven
fast; the integration is transfer-bound because the model must stay on CPU
(PyTorch GPU broken on sm_61). Dispatcher kept for portability (wins if the
model ever lives on GPU), but default production path = CPU + kernel wins only.

**CORRECTED END-TO-END NUMBERS (interleaved A/B, same process):**
- Swarm short-circuit: **0.99x** (no win; identical outputs; kept as cleanup)
- `compute_aux=False`: **1.21x prefill / 1.12x decode** — ⚠️ CHANGES OUTPUTS
  (top token differs). Opt-in behavior change, not a free win.
- (Earlier "1.73x/1.77x" figures were CPU-frequency drift artifacts.)

## How to use
```python
import sys
sys.path.insert(0, r"C:\02_QUILLAN\03 - Training & Model\scripts")
from quillan_v5_4_oni import QuillanRoninOni, QuillanOniConfig
import quillan_cuda_dispatcher as D
model = QuillanRoninOni(QuillanOniConfig()).eval()
D.install(model)  # routes BitLinear through sm_61 DLLs
tokens = model.generate([42, 7, 1337], max_tokens=64)  # compute_aux=False: 1.12-1.21x but CHANGES outputs (documented)
D.uninstall()
```

## Backup
Original model file: `quillan_v5_4_oni.py.bak_speedup` (toggle by copy).

## Future work (not done)
- 2-bit ternary GEMV packing (4x less weight traffic; needs repack + decode kernel)
- Persistent grid (5 CTAs) for GEMM—only helps if launch overhead dominates
- KV cache preallocation (avoid torch.cat per token)
- Dropdown: install an sm_61-capable PyTorch (≤1.13) to enable full GPU inference
