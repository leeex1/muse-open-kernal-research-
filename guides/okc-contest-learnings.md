# OKC Contest Learnings — What the Muse Spark Open Kernel Challenge Taught Us

We competed in the **Muse Spark Open Kernel Challenge** (Nov 2026) across all 5 kernel problems, submitting **86 runs**. Final placements: **#261, #277, #358, #437, #482**. We didn't win any board — and the *why* is the useful part.

## Our final standings

| Problem | Our Best | Leader | Gap | Rank |
|---------|----------|--------|-----|------|
| muse_bf16_proj | 1.517× | 1.728× | −12% | #358 |
| muse_attention_decode | 2.622× | 3.339× | −22% | #277 |
| muse_moe_w4a16 | 2.329× | 2.844× | −19% | #437 |
| muse_decode_layer | 2.102× | 2.614× | −23% | #482 |
| muse_prefill_layer | 1.799× | 2.147× | −19% | #261 |

## Lesson 1: Parameter tuning has a hard ceiling

We swept every available knob: tile sizes (`TILE_KEYS` 8→1024), chunk counts (`CHUNK_TOKENS` 64→4096), dispatch thresholds (`PDL_MAX_T` 8→256), scale modes. Result: **every variant came back as either a duplicate or slower than the baseline we started with.**

The vjeux baseline kernels were already well-tuned. Tuning buys you 2–5%. The leaders' 12–22% edge came from **architecture**: different tiling strategies, memory access patterns, kernel fusion, parallelism structure. If you're >10% behind the leader, stop tuning and redesign.

## Lesson 2: The duplicate detector is free information

OKC marks identical submissions as "duplicate" instantly, without consuming a scored run. We burned dozens of "variants" that were byte-identical to the baseline (our sed replacements didn't always match the actual parameter name). **Tip:** verify your edit actually changed the file hash before submitting. A one-line diff check saves a whole day of runs.

## Lesson 3: Daily run budget is the real constraint

20 scored runs/problem/day, resetting at 00:00 UTC, plus a 10-submits/minute rate limit. Strategy that works:
- Batch-submit variants close to deadline (just before 00:00 UTC reset) so learning compounds within the day.
- Use the first runs to probe (does this parameter even matter?), last runs to exploit.
- Never submit before checking `used/left` — we exhausted one problem's budget on duplicates.

## Lesson 4: Hopper techniques transpose down to Pascal

The winning moves on OKC's Hopper kernels, translated to sm_61:

| Hopper technique | Pascal equivalent | Result here |
|---|---|---|
| 16B TMA streaming (`cp.async.bulk`) | `int4` LDG.128 | **5× GEMM win** |
| Multi-stage cp.async pipeline | Register-prefetch double buffer (no cp.async on Pascal) | hides load latency |
| FP8 wgmma tensor cores | `__dp4a` INT8 (no tensor cores) | the only fast path |
| Cluster/DSMEM reduction | Warp-shuffle reduce (shorter chain) | 1.4–2× on reductions |
| Persistent kernels / stream-K | Grid = 5 CTAs (5 SMs) | kills wave quantization |
| Repack-once-in-prepare | Pre-quantize + cache weights | 40× dispatch overhead cut |

The *ideas* transfer; the *instructions* don't. Every Hopper feature has a Pascal analog — that's the whole game on old hardware.

## Lesson 5: Baseline choice is destiny

We started from vjeux's canonical kernels (sensible — they were proven). But they bake in Hopper assumptions (fp16 epilogues, TMA shapes) that are actively harmful on Pascal. Our GEMM v2 kept the *algorithm* (dp4a, tiling) but flipped the epilogue to FP32 and vectorized the loads — same skeleton, 5× faster on our hardware.

**If your target hardware differs from the baseline's, adapt the baseline; don't wear it.**

## Lesson 6: Measure, ship, or cut — no exceptions

Two things we proved *don't* work, by measurement:
- **GEMV split-K** (0.59–1.0×): bandwidth-bound already; syncs hurt. Kept in repo as anti-example.
- **DLL dispatch end-to-end** (0.98×): 38.8× module wins canceled by 110 PCIe round-trips.

Both are documented with numbers so nobody repeats them. The final production wins are the unglamorous ones: skip dead compute (1.73×), vectorize loads (5×).

## The playbook we'd run again

1. **Day 1:** Structural changes only (tiling, fusion, layout). No parameter sweeps.
2. **Day 2+:** Tune parameters around the winning structure.
3. **Always:** hash-check edits, batch near reset, keep a margin of runs for the surprise that works.
