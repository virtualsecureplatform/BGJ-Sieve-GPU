# Indexed (compressed) buckets — design & staging

Goal: replace materialized full-vector bucket copies with compact **postings**
(pool index + metadata), gathering vectors from the RAM-resident pool at
reduce time. Target: the ~1.5–2 TB/run bucket-spill traffic and the 32 GiB
BWC pinned arena — the last measured bottleneck at n≥140 (the 1.4–1.7× tax on
spill dims). Reference implementation: **BGJ-Sieve-AMX** `bucket_epi8_t`, which
already stores `uint32` pool indices instead of copies.

## Current data flow (measured, not assumed)

Per bgj sieve (one CSD), bucketer and reducer run **concurrently**
(`bgj_hd_device.cu:38-39`, both threads joined at dim end). The bucketer's
`_batch` (`:1615-1821`):

1. Pops a batch of pool chunks as `dst` → `working_chunk[task_chunks]`
   (64 chunks/batch = ~524K vecs).
2. **Inserts** reducer solutions into those same chunks first: score-threshold
   replacement overwrites low-score slots in place — `dst->u/vec/norm/score[pos]`
   (`:1665-1689`). So a chunk is *mutated before it is bucketed, in the same batch*.
3. GPU-buckets the post-insertion chunks (`_buc_buf->run`, `:1731`).
4. Scatters entries into bwc bucket chunks by **copying** `working_chunk[..]->vec`
   (`:1745-1774`) — the ~150 B/entry copy that becomes the spill stream.
5. Releases the chunks back to the pool (`release_sync`, `:1707/1780`).

Reducer `_ld_sbuc` (`:4126`) reads materialized bucket chunks via
`bwc->fetch_for_read`. The copy exists so the reducer sees a **stable snapshot**
decoupled from ongoing pool mutation.

## The hazard for indexed buckets

A posting `(chunk_id, pos)` can go stale if slot `pos` in `chunk_id` is
overwritten by an insertion between when the entry is bucketed and when the
reducer gathers it. Overwrites happen only to slots with `score < replace_th`,
and only when the chunk is popped by a later batch.

## Staleness rate — the load-bearing empirical question

A first (WRONG) estimate guessed a chunk is re-popped only every ~450 batches
(mistaking the per-batch bucket-*center* count, 64, for a chunk count). The pop
logic refutes this: `buc_iterator_t::reset()` (`:1181`, called every batch)
rescans from `last_insert_chunk_id`, and `pop()` (`:1207-1226`) walks
`curr_full_id` **sequentially through all chunks** (`(id+1) % (first_empty+1)`).
So **each outer batch is a full-pool sweep** — every chunk is popped,
insert-mutated, and bucketed *once per batch* (~6.6 s in exp19). "batch (64)" =
64 bucket centers, not 64 chunks.

⇒ Re-pop interval ≈ **1 batch**, comparable to the coupled reducer's consumption
lag (buc elapsed ≈ red elapsed). A posting `(C,pos)` from batch N is therefore at
genuine risk: batch N+1 re-pops C and may overwrite `pos` while the reducer is
still draining batch N's buckets. **Staleness is NOT obviously low.**

One mitigating factor decides viability: insertions overwrite only
`score < replace_th` slots (the *longest* vectors), whereas a bucketed vector
passed a center's α-filter (it's relatively *short*). If "bucketed" and
"overwritten-next-batch" are weakly correlated, effective staleness can still be
small. This correlation is exactly what Step 1 must measure — the cheap Option B
lives or dies on it, and the honest prior is now "unknown," not "near-zero."

## MEASUREMENT RESULT (2026-07-31, SVP-130, `HD_MEASURE_STALE=1`) — Option B DEAD

Staleness upper bound = **55–72% every deep dim** (steady ~60% at CSD 100–112;
e.g. CSD 112: 131.3M overwrites, 94.8M bucketed-last-batch = 72.2%). ~60% of
slot overwrites hit a slot with an in-flight posting. The "bucketed-short vs
overwritten-long are weakly correlated" hope is FALSE — the full-pool sweep
buckets most of the pool into some center each batch, so most overwritten slots
also carry a live posting. Correctness confirmed unaffected (solved 2812.38 =
true min). ⇒ **Option B (index + drop-stale) is not viable**: it would silently
drop ~60% of every bucket's pairs → the reducer searches <half the intended
pairs → convergence craters. This also explains *why* AMX uses epoch-deferred
insertion — it is REQUIRED for indexed buckets, not incidental. Only Option A
remains.

## Two designs

- **Option B (lightweight, index + staleness check).** Posting =
  `(chunk_id:24, pos:13, sign:1, norm:16)` = 8 B (vs ~150 B). At gather, skip if
  `pool[chunk_id].norm[pos] != posting.norm` (stale → drop; safe because every
  surviving pair is exactly re-verified downstream anyway). No pipeline refactor.
  Viable **iff** measured staleness is low (est. <<1%). ~19× smaller buckets;
  can drop the whole bwc arena for bgj2 (whole-dim postings ~6 GB fit RAM).
- **Option A (heavy, epoch-deferred insertion, AMX-style).** Buffer all
  solutions; apply only at epoch boundary so the pool is read-only during
  bucket/reduce → postings never stale. Correct unconditionally, but changes
  convergence dynamics (pool doesn't improve mid-epoch — AMX accepts this; the
  GPU authors chose live insertion) and needs a solution-buffer. Fallback if B's
  staleness proves too high.

## Staging (each step independently validated; env-gated; opt-in)

1. **Measure staleness (no format change).** [IMPLEMENTED — `HD_MEASURE_STALE=1`,
   binary `app/hd_sieve_140M`.] Per-slot `uint8` array = batch # a slot was last
   bucketed in (mod 256); at each insertion overwrite, count whether the slot was
   bucketed in the immediately preceding batch (a posting still in flight). Emits
   `[STALE] CSD n: batches B, overwrites O, stale-next-batch H (P%)` per dim.
   P = **upper bound** on Option-B posting staleness (upper because it ignores
   that the reducer often consumes a batch's postings before the next batch's
   insertion phase reaches that slot). Decision rule: P << 1% → Option B is safe
   (drop stale pairs, exact-verify downstream). P large → need Option A. Zero cost
   when off (all touches guarded by `_measure_stale`).
2. If staleness low → **Option B**: add posting format + gather path behind
   `HD_INDEXED_BUCKETS=1`; keep the copy path as default. Validate SVP-130
   (correctness + true-min) then n=140 (perf + disk delta).
3. If staleness high → **Option A**: epoch-deferred insertion refactor first,
   then indexed buckets on top.

## Projected payoff (n=140)

Bucket bytes ~19× smaller → spill stream ~2 TB → ~100 GB; removes the 1.4–1.7×
spill tax on dims ≥118; frees the 32 GiB BWC arena (→ larger PWC or page cache).
Does **not** move the RAM ceiling (that's pool residency / item 3).

## Validation invariants (any option)

- SVP-130 seed 52 must still find true min 2812.38 (0.9752·GH), verified.
- n=140 seed 5 must still solve ≤1.05·GH.
- Compare per-dim times + `disk_written_gb` vs exp19 baseline (9541s / 1824 GB).
- Watch reducer `u`/`r` counts: a drop in unique inserts = silently lost pairs
  (the Option-B failure signature) → abort.
