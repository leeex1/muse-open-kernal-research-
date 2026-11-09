# Troubleshooting — sm_61 / Pascal Pitfalls

## 1. "CUDA error: no kernel image is available for execution on the device"

**Cause:** PyTorch (≥2.x, CUDA ≥12) dropped sm_61 support. The GTX 1050 cannot run *any* PyTorch CUDA kernel — even `x.abs().max()` crashes.

**Fixes (pick one):**
- ✅ Use the custom sm_61 kernels in this repo (NVRTC + driver API, bypasses PyTorch dispatch).
- Install PyTorch ≤1.13 (CUDA 11.x) which still ships sm_61 kernels. Beware: may break other projects sharing the env.
- Keep the model on CPU and only shift int8 bytes to GPU for the DLL kernels (what `quillan_cuda_dispatcher.py` does).

## 2. nvcc: "unsupported Microsoft Visual Studio version" (C1189)

**Cause:** nvcc 11.8 vs VS2022 14.44 STL headers (which demand CUDA ≥12.4).

**Fix:** add `-allow-unsupported-compiler`, or better — **skip nvcc entirely**:
- Compile device code with **NVRTC** (`ptx_build.py`): no MSVC headers involved, emits `compute_61` PTX.
- Link host code with `cl.exe` (handles VS2022 fine).
- If you must use nvcc, prefer CUDA 12.9's nvcc + `-allow-unsupported-compiler`.

## 3. nvcc: "STL1002: Unexpected compiler version"

Same root cause as #2 (VS2022 STL gates on CUDA 12.4+). Use NVRTC or CUDA 12.9 nvcc.

## 4. "misaligned address" crash in vectorized kernel

**Cause:** `int4` (16B) shared-memory stores to rows whose stride isn't a multiple of 16.

```cuda
// stride 36 → row 1 starts at byte 36 (4 mod 16) → int4 store misaligned → crash
#define SMEM_K_PAD (TILE_K + 4)   // ❌

// stride 48 → all rows 16B aligned ✅
#define SMEM_K_PAD 48
```

Rule: any `int4` smem access needs row stride ≡ 0 (mod 16).

## 5. Kernel runs but output is garbage (fp16 on Pascal)

**Cause:** converting to/from FP16 on sm_61 without care, or writing the epilogue in half precision. FP16 math is ~1/64 rate AND overflows where FP32 wouldn't.

**Fix:** compute and store FP32. If you need FP16 downstream, convert once at the boundary (CPU-side is free).

## 6. "Duplicate" submissions on OKC burning your day

**Cause:** parameter rename didn't match (e.g. editing `TILE_KEYS` when the file uses `TILE_K`), so the file was byte-identical.

**Fix:** hash before submit:
```bash
md5sum submission.py  # compare to baseline; identical = don't submit
```

## 7. PowerShell quoting eats your loop/ssh commands

`$variables` and quotes get mangled through PowerShell → WSL chains. **Fix:** write `.sh` / `.py` files and execute the file, never inline complex one-liners.

## 8. Dispatcher slower than CPU?

**Cause (measured):** per-call requantization of weights (16 ms) + 110 sequential PCIe round-trips per token.

**Fix checklist:**
- Cache pre-packed int8 weights per module, keyed on `weight._version` ✅ (done — 40× overhead cut)
- Only dispatch M≥64 + out×in≥1M (transfer amortizes) ✅
- Decode stays on CPU ✅
- If still slow: profile — usually it's still the `.cpu()` sync per module; batch or drop dispatch.

## 9. Which kernel for which shape?

| Shape | Kernel | Why |
|---|---|---|
| M=1, decode | GEMV (`quillan_cuda_sm61`) | 15–20× vs CPU; GEMM wastes 99.2% of threads |
| M=8..32 | GEMV or CPU | borderline — benchmark both |
| M≥64, prefill | GEMM v2 (`sm61_qgemm_v2`) | 3.2–5× vs v1, 2–3× vs CPU |
| M≥64, small N | GEMM v2 | still wins; verify N%64 padding |

## 10. Occupancy questions

- GP107: 5 SMs, 2048 threads/SM max, 64K regs/SM, 48KB smem/block (default).
- GEMM at 256 threads + 6KB smem → ~8 blocks/SM by threads, smem fine. Accumulators (32/thread) eat the reg budget — that's by design; ILP > occupancy.
- If a variant drops below 2 warps/SM resident, smem latency can't hide → redesign, don't tune.

## 11. NVRTC specifics

- Include path: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\include`
- DLL: `...\bin\nvrtc64_120_0.dll` (version-suffix may differ; find via `dir`).
- Strip host code: split source at `// Host-side launcher.` — NVRTC must never see `<<<>>>` or MSVC headers.
- Replace `#include <cstdint>` with manual typedefs (NVRTC chokes on MSVC's `<cstdint>`).
- Flags: `--gpu-architecture=compute_61 --std=c++17`.
