# Two-node pump-and-jump BKZ

The MPI backend extends the existing distributed sieve to `--task pump` and
`--task bkz`. It requires two distinct nodes, one rank and four visible GPUs per
node. The full basis is replicated; the vector pool and UID entries are sharded
by normalized UID. Distributed dual-hash insertion is intentionally unsupported
(`--BDH 0` is required). The first production target is CSD144, not a claim of
support for arbitrary larger dimensions.

## Control and correctness

Each insertion searches both pool shards. The ranks exchange fixed-size lift
records and agree on the weighted insertion objective, primitive coefficients,
insertion/removal positions and sign. Ties prefer the lower insertion position,
then rank. Both ranks apply the same basis change and transform their local
vectors. UID coefficients are synchronized and vectors redistributed before the
next sieve. Basis fingerprints must agree after transitions; an exact integer
lattice-equivalence check runs at each completed pump before output publication.

`tools/bkz_scheduler.py --backend mpi` reuses the existing schedule JSON and
launches two-rank Slurm steps. Give `--binary` the executable
`reproduce/svp167-search/mpi_rank.sh` wrapper and export `MPI_BINARY`, `MPI_BASE`,
`MPI_SIF`, and `MPI_LIBDIR` as shown in the pilot script. MPI run identities are
separate from serial identities. A scheduled final sieve is not supported by
this backend yet; use the existing MPI final-sieve entrypoint separately.

## Recovery and I/O policy

`LATEST-BKZ` points to an immutable version2 generation containing both rank
snapshots. Snapshots contain binary quad-float bases, pool shards, UID
coefficients, the pump cursor and explicit next operation, configuration/input
fingerprints and file checksums. Binary bases avoid rounding projected lattices
on recovery. Both ranks finish and fsync their snapshots before rank0 publishes
the manifest. The previous committed generation remains available; incomplete
generations are not resumed or overwritten. Version1 final-sieve checkpoints
remain separate.

Checkpoint boundaries are completed pump-up sieves and completed pump-down
insertion/pruning/sieve transactions. Restarting repeats only the uncommitted
phase. Sampling/center randomness is reseeded per phase; recovery is logically
consistent but does not promise bit-identical GPU execution. Configuration or
input mismatches, missing shards and checksum failures are fatal.

`HD_FAIL_ON_SPILL=1` forbids unforced pool, bucket, solution and UID spill writes.
Use `HD_LAZY_SYNC=1`, `HD_BWC_LAZY_SYNC=1`, `HD_HBM_PWC_WRITEBACK=1`, and
`HD_PWC_NO_GROW=1` with this policy. Deliberate checkpoint/context-transition
flushes still write to disk. This is not an entirely disk-free mode.

Bucket windows are bounded from the full cross-rank bucket-size upper bound and
host-cache budget, including in-flight reduction buckets. Transport receives
are bounded, bucket records are streamed by chunk, candidates are routed in
frame-sized windows, and the pending remote candidate queue has a hard1GiB
limit. Capacity violations abort rather than truncate vectors or silently spill.
Every60seconds a local heartbeat reports phase, memory and communication/queue
counters without issuing MPI collectives. An unchanged heartbeat is evidence to
investigate, not an automatic deadlock verdict.

## Validation and bounded pilot

Submit `reproduce/svp167-search/mpi_bkz_validate.sbatch` with an explicit
`SOURCE_COMMIT`. It builds archived sources on a compute node, generates and
preprocesses a small fixture, then tests insertion agreement, phase recovery,
multiple pumps, interrupted publication, corrupt/missing snapshots and final
sieve compatibility. `BUILD_ONLY=1` permits a one-GPU compilation-only allocation.

After validation passes, use `mpi_bkz144_pilot.sbatch` with a fresh `PILOT_ROOT`.
It continues seed38's saved BKZ143 result, through CSD144 and the first completed
pump-down transaction, then exits75 internally (the batch script records this
expected pause as success). Submit the same commit and root with
`PILOT_RESUME=1` to execute one additional phase and pause again. No full two-tour
chain is automatically launched. Every job is limited to two nodes; builds and
GPU tests must never run on the login node.

Initial per-node budgets are300GiB PWC,32GiB BWC and16GiB SWC, with16 reducer
workers. HBM PWC is disabled. The pilot keeps BGJ2 through CSD144 instead of
crossing into disk-oriented BGJ3L; vector-count and saturation targets remain
unchanged. These are candidate settings, not established high-dimension results.
Phase snapshots can be large and expensive: check available filesystem space
for live scratch plus both retained generations before starting a large pilot.
