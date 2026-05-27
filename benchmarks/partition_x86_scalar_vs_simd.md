# `np.partition` on x86: scalar vs Intel SIMD

Benchmark data gathered while reviewing the `np.partition` VQSelect work in
numpy/numpy#31506. The PR adds a size gate (`HWY_QSELECT_MIN_N = 16384`) to the
ARM/PPC Highway path to avoid small-array regressions; the x86 path has **no
such gate**. This document measures whether x86 needs one.

## TL;DR

On x86 (AVX-512), Intel's `x86-simd-sort` `qselect` beats scalar introselect at
**every** size and dtype tested — including the smallest L1-resident size
(1024 elements). There is no crossover, so the x86 path has no regression to
gate against. The ARM regression is specific to Highway VQSelect's dispatch
overhead and an ARM-only threshold is justified on its own merits; the x86
"no gate" precedent is not evidence against gating ARM.

## Method

- **Machine:** Intel Xeon, AVX-512 (skx + `AVX512_ICL` + `AVX512_SPR`), x86_64.
- **NumPy:** built `2.6.0.dev0` from the current main base. PR #31506 only touches
  the ARM/PPC Highway path, so the x86 `qselect` path is unchanged and
  representative.
- **x86 dispatch:** `np.partition` with a single `kth` routes to
  `x86simdsortStatic::qselect` via `numpy/_core/src/npysort/selection.cpp`
  (`quickselect_dispatch`, called for `nkth == 1`) — with no minimum-size check.
- **Scalar vs SIMD toggle:** same binary, switched with
  `NPY_DISABLE_CPU_FEATURES="X86_V3 X86_V4 AVX512_ICL AVX512_SPR"` (disables all
  dispatchable qselect targets, forcing scalar introselect).
- **Sanity check on the toggle:** `np.sort` of 1M float32 goes 3.9 ms (SIMD) →
  91 ms (scalar) when disabled, confirming the switch engages/bypasses SIMD.

## 1. `bench_function_base.Partition.time_partition` (ARRAY_SIZE = 100,000)

The literal asv benchmark requested in the PR, run twice
(`asv run --python=same`), all 126 parameter combinations. Speedup is
SCALAR ÷ SIMD; values > 1 mean Intel SIMD is faster.

**Intel SIMD is faster than scalar in all 126 cases — zero regressions.**
Smallest margin: float32 `ordered` at 1.05x.

Representative summary (speedup, SCALAR ÷ SIMD):

| dtype   | random | reversed | sorted_block/10 | ordered | uniform |
|---------|--------|----------|-----------------|---------|---------|
| int16   | 28.0x  | 13.5x    | 13.3x           | 2.8x    | 6.6x    |
| int32   | 12.3x  | 13.6x    | 6.2x            | 2.4x    | 6.3x    |
| int64   | 7.2x   | 9.0x     | 4.4x            | 1.45x   | 3.2x    |
| float16 | 11.1x  | 15.3x    | 3.7x            | 3.4x    | 2.5x    |
| float32 | 6.0x   | 8.4x     | 3.5x            | 1.07x   | 3.0x    |
| float64 | 5.7x   | 9.8x     | 3.6x            | 1.18x   | 3.2x    |

Full per-case table: see `partition_x86_results_raw.txt` in this directory
(SIMD µs, SCALAR µs, speedup for every dtype × array_type × k).

## 2. Size sweep: 1,024 → 10,000,000 (the small-array regime the ARM gate targets)

The asv benchmark only tests 100k elements (well outside L1), so it cannot
expose a small-array regression. This sweep drops down to 1024 elements
(`random`, k=100). Speedup = SCALAR ÷ SIMD; > 1 means SIMD wins.

| size       | int16  | int32 | int64 | float16 | float32 | float64 |
|------------|--------|-------|-------|---------|---------|---------|
| 1,024      | 1.76x  | 1.48x | 1.27x | 1.88x   | 1.48x   | 1.50x   |
| 4,096      | 2.12x  | 2.04x | 1.88x | 2.04x   | 1.60x   | 1.77x   |
| 8,192      | 7.16x  | 6.78x | 6.66x | 3.63x   | 7.68x   | 4.00x   |
| 16,384     | 11.97x | 9.74x | 7.46x | 8.39x   | 6.87x   | 5.03x   |
| 65,536     | 19.97x | 22.0x | 8.84x | 11.25x  | 10.02x  | 7.30x   |
| 100,000    | 28.43x | 12.3x | 7.20x | 11.93x  | 6.16x   | 5.70x   |
| 1,000,000  | 7.40x  | 3.35x | 1.72x | 8.16x   | 2.11x   | 1.50x   |
| 10,000,000 | 16.30x | 3.40x | 2.39x | 9.83x   | 3.07x   | 2.28x   |

**No crossover anywhere** — Intel `qselect` wins from 1024 elements up, across
all six dtypes. The 16,384 column (the value chosen for the ARM gate) shows x86
SIMD already 5x–12x ahead.

## Conclusion

x86 has no gate because it does not need one: Intel's `x86-simd-sort` qselect
has low enough fixed/dispatch cost that it wins even on tiny, L1-resident
arrays. The ARM `HWY_QSELECT_MIN_N` threshold should be sized from the actual
Highway-vs-scalar crossover measured on the target ARM hardware (same method:
toggle the Highway dispatch and sweep sizes). It is independent of x86 and will
not affect the x86 path.
