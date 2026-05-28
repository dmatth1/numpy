# Can the evolve-sort primitives improve numpy (and pandas)?

> Feasibility investigation, 2026-05-28. Grounds the `evolve` repo's
> `sort/NOVELTY_BRAINSTORM.md` thesis against the *actual* numpy source on this
> branch (cut from `main` at the 2.6.0-dev BEG commit). All evolve performance
> numbers below were measured **in the evolve repo** (M1/x86, via ctypes vs
> numpy / std::sort / VQSort) — **not yet inside numpy**. Treat them as an upper
> bound on the plausible win, pending a real benchmark in numpy's harness.
> Status: **investigation, no code committed yet.**

## TL;DR

| Opportunity | Where | Novelty | Confidence | Risk | Verdict |
|---|---|---|---|---|---|
| **Herf float radix sort/argsort** | `npysort` float path | low (Herf 2001) | **high** | low–med | **Do this first** |
| **Pack-index radix argsort for 32/64-bit ints** | `aradixsort` / dispatch | low (ips2ra) | high | med | Strong second |
| **Single-pass lexsort via composite order-preserving key** | `item_selection.c` lexsort | **med–high (publishable)** | med | med–high | The novel bet |
| **`np.unique` / sort-based grouping** | `_arraysetops_impl.py` | none | high | low | Free rider on the above |

The cleanest high-value win is **radix on floats** (numpy never radix-sorts
floats today). The **novel + publishable** bet is **single-pass multi-key
lexsort** via order-preserving composite encoding — numpy currently does N
comparison-sort passes, and "frontier order-preserving multi-key sort on ARM"
is the defensible, unstaked result the brainstorm is chasing.

---

## What numpy actually does today (verified, file:line)

### `np.lexsort` — N comparison passes, one per key

`item_selection.c:1809–2030`. The core loop (≈1978–2000) walks keys
least-significant-last and, for each key, runs the dtype's **stable** argsort
(or falls back to `npy_atimsort`) over a permuted buffer:

```c
for (j = 0; j < n; j++) {              // one pass PER KEY
    argsort = ...->argsort[NPY_STABLESORT];
    if (argsort == NULL) argsort = npy_atimsort;
    rcode = argsort(valbuffer, indbuffer, N, mps[j]);   // comparison sort
    // reorder indices by this key, feed into next pass
}
```

So lexsort = **k stable comparison sorts**, not a radix. Brainstorm claim
**confirmed**. Encoding all key columns into a single order-preserving
composite key and radix-sorting **once** is a structural win here.

### Radix sort exists — but **only for ≤16-bit integers**

`npysort_methods.cpp:58–86` gates radix to `bool, ubyte, byte, ushort, short`
only:

```cpp
constexpr bool use_radixsort =
    std::is_same_v<Tag, npy::bool_tag>  || std::is_same_v<Tag, npy::ubyte_tag> ||
    std::is_same_v<Tag, npy::byte_tag>  || std::is_same_v<Tag, npy::ushort_tag> ||
    std::is_same_v<Tag, npy::short_tag>;
```

`int32/int64/uint32/uint64` and **all** floats fall through to timsort
(stable) or quicksort/SIMD (default). The LSD radix itself
(`radixsort.hpp:49–92`) handles descending via a `reverse` template param. So
numpy's radix is real but deliberately narrow — **32/64-bit integer radix is
not present**, which is exactly where the evolve adaptive radix lives.

### Floats are never radix-sorted

Confirmed by the gate above. Floats route to timsort (stable) / quicksort
(default) / SIMD qsort. There is no order-preserving float encoding anywhere
in `npysort`. Brainstorm claim **confirmed**.

### SIMD: ARM **is** covered (a correction to the brainstorm)

`quicksort.hpp:28–68` dispatches 32/64-bit element sorts to:
- **x86**: `x86-simd-sort` (`x86_simd_qsort.dispatch.cpp`) — **ascending only**
  for the SIMD path.
- **ARM/other**: Highway (`highway_qsort.dispatch.cpp:29–34`) — `QSort<float>`
  and `QSort<double>`, **both directions**.

So the brainstorm's "floats fall back to comparison on ARM" is **wrong**: float
sort on ARM is Highway VQSort, not scalar comparison. The evolve `f64_radix`
win (3.3–3.7× vs `np.sort` on M1, `NUMPY_BENCH.md`) was measured against this
Highway path — so the opportunity is real, but the competitor is VQSort, and
the win **must be re-measured inside numpy's current dispatch**, not assumed.

### `np.unique` rides on argsort

`_arraysetops_impl.py:350–414`: when indices are requested it calls
`argsort(kind='mergesort')` (timsort) then scans for transitions. Any faster
stable argsort flows straight through — **no API change, low risk**.

---

## Where the evolve primitives map in

### 1. Herf float radix sort + argsort — **recommended first**

**Target:** the float branch of `npysort_methods.cpp` dispatch + a new
`aradixsort`/`radixsort` float specialization in `radixsort.hpp`.

**Primitive:** evolve `scalar/candidates/f64_radix.cpp` — the order-preserving
IEEE→u64 bit flip (positive: flip sign bit; negative: flip all bits), then the
adaptive LSD radix. ~40 lines, no SIMD, decode is the inverse flip.

**Why it fits:** numpy has *no* float radix; the encoding is self-contained and
order-preserving, so it composes with the existing LSD `reverse` param for
descending. Flows into `np.unique`, `np.sort`, `np.argsort`, and pandas
(`sort_values`, `groupby`, `factorize` all bottom out in numpy sorts).

**Expected win (measured elsewhere; re-measure):** 3.3–3.7× over `np.sort` on
random f64 (M1). **But low-cardinality floats lose** (`NUMPY_BENCH.md`: 0.47×) —
the ordered-int range is too wide for counting-collapse — so a cardinality/range
gate is mandatory, and the comparison/Highway path stays as fallback for
small `n`, near-sorted, top-k (`np.partition`), and low-cardinality.

**Risk:** low–med. Additive, gated, fallback preserved; NaN ordering and
`-0.0/+0.0` must match numpy's existing semantics (the encode/decode must round-trip
NaN to where numpy puts it). Bench harness: `benchmarks/` + the sort asv suite.

### 2. Pack-index radix argsort for 32/64-bit integers

**Target:** the `aradixsort` dispatch (`npysort_methods.cpp:115–117`) and the
32/64-bit integer argsort path (today quicksort/SIMD).

**Primitive:** evolve `unified/argsort.hpp` + `sort_integers.hpp` — radix
co-sort of `(value, index)` with the index *packed into the radix word*,
adaptive 11-bit digit, counting-collapse (`range ≤ 4n`), constant-digit skip,
range-normalized pass count.

**Why it fits:** numpy's stock radix stops at 16-bit; 32/64-bit integer argsort
is comparison/SIMD. evolve measured 8.9× (u32 1M), 4.2–8.2× (u64), 20×
(low-cardinality) vs `np.argsort`. **Note:** the fork's existing
`neon-argsort-radix` branch already routes large disordered *integer* argsort
to a radix on ARM and reports 5.5–7.4× — but it uses numpy's *stock* LSD, is
**integer-only**, and **single-key only**. The evolve frontier engine (adaptive
digit width + counting-collapse + constant-digit-skip + pack-index) is the
upgrade, and floats (item 1) + multi-key (item 3) are the un-taken extensions.

**Risk:** med — stability (numpy argsort is stable for `mergesort`; the radix
must be stable), scratch memory (2–3N), and the near-sorted blind spot (keep
the profitability gate the fork branch already uses).

### 3. Single-pass `lexsort` via order-preserving composite key — the novel bet

**Target:** `item_selection.c` lexsort (replace the k-pass comparison loop with
a single composite radix when all key dtypes carry an order-preserving encoding).

**Idea:** encode each key column into order-preserving bytes (ints: bias/sign-flip;
floats: Herf; fixed strings: big-endian; descending: invert), concatenate into
one composite key per row, radix-sort once. The dtype heterogeneity that the
numpy agent flagged as the blocker is **exactly what order-preserving encoding
dissolves** — every key becomes comparable bytes.

**Why it's the publishable one:** this is Arrow's row-format trick, which numpy
**does not have**; combined with the evolve *deferred-materialization* angle
(encode a fixed-width prefix + index, materialize deeper key bytes only for
prefix-collision runs) it becomes "frontier order-preserving multi-key sort,
**on ARM**" — the defensible, unstaked result the brainstorm targets. k-pass
timsort → one adaptive radix pass is a structural, not constant-factor, change.

**Risk:** med–high. Object/structured/datetime dtypes and custom comparators
need a comparison fallback; variable-length strings need the deferred path or
length handling. Scope to the common case first (numeric + fixed-width keys),
fall back otherwise.

### 4. `np.unique` / sort-based grouping — free rider

No new code beyond items 1–2: a faster stable argsort makes
`unique(return_index=True)` and pandas sort-based `groupby`/`factorize` faster
transparently (`_arraysetops_impl.py:381`).

---

## Verification of brainstorm claims

| Brainstorm claim | Verdict | Evidence |
|---|---|---|
| lexsort does a comparison sort per key | **TRUE** | `item_selection.c:1978–2000` |
| numpy radix is integer-only | **TRUE, and narrower** — ≤16-bit ints only | `npysort_methods.cpp:58` |
| floats fall back to comparison on ARM | **FALSE** — Highway VQSort covers f32/f64 on ARM | `quicksort.hpp:28`, `highway_qsort.dispatch.cpp:29` |
| shipped `neon-argsort-radix` PR is int-only / stock-radix / single-key | **TRUE** = the opportunity (floats, frontier engine, multi-key all open) | fork branch `neon-argsort-radix` |
| multi-key lexsort via composite encoding is the highest-reach lever | **plausible**; blocker (dtype heterogeneity) is what encoding solves | this doc, item 3 |

## Suggested next step

Ship **item 1 (Herf float radix sort+argsort, gated, with `np.unique` riding
along)** as the first low-risk PR with an asv sort benchmark, then **item 3
(single-pass lexsort)** as the higher-effort, novel follow-up benchmarked on
ARM vs the current k-pass path. I can start either on request.

---
*Code citations are against this branch's numpy checkout. evolve numbers are
from the evolve repo's `sort/{NUMPY_BENCH,UNIVERSAL_SORT_LOG,KEYREADER_LOG}.md`
(cross-architecture, not numpy-internal benchmarks).*
