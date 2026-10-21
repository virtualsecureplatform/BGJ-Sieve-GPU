# Two-node pump-and-jump BKZ

Experimental: passing a build does not establish distributed correctness or
CSD144 feasibility. Gate real work on the integration tests and bounded pilot.

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
frame-sized windows, and the pending remote candidate queue has a hard 1 GiB
limit. Capacity violations abort rather than truncate vectors or silently spill.
Every 60 seconds a local heartbeat reports phase, memory and communication/queue
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

Initial per-node budgets are 260 GiB PWC, 128 GiB BWC and 16 GiB SWC, with 16 reducer
workers. HBM PWC is disabled. The pilot keeps BGJ2 through CSD144 instead of
crossing into disk-oriented BGJ3L; vector-count and saturation targets remain
unchanged. These are candidate settings, not established high-dimension results.
The originally planned 300/32 GiB split fails the conservative CSD144 bucket-window
bound: two 16-center waves require about 109 GiB of BWC. Lossless chunk compaction
after redistribution and pruning keeps each half-pool near its 234 GiB packed size.
Phase snapshots can be large and expensive: check available filesystem space
for live scratch plus both retained generations before starting a large pilot.

Remote candidate producers now wait at the 1 GiB queue limit instead of
aborting. The bucketer coordinator drains candidates between bucket fragments,
while waiting for bucket capacity, and before reducer shutdown. Workers never
issue MPI calls, and no candidate is dropped or spilled. An additional outgoing
buffer of at most 1 GiB can coexist with the producer queue during exchange.
`HD_MPI_CANDIDATE_QUEUE_KIB` (1 through 1048576) lowers the queue limit for
stress tests; validation forces 1 KiB to exercise producer waits and draining.

BKZ v2 snapshot recovery uses a lossless record loader, not the legacy GPU
pool-repair loader. It verifies filenames, chunk lengths and headers, dense live
slots, UID uniqueness/ownership and the recorded total, then rebuilds UID tables
and score statistics without re-rounding coordinates or filtering records. The
snapshot inventory and recovered-pool check remain mandatory.

## Transport and performance validation

Wire protocol v2 packs bucket chunks in columnar norm/vector blocks, batching
them into at most 64 MiB (or one chunk when the configured frame is smaller).
It uses hardware-accelerated interleaved CRC32C lanes for transport corruption
checks; checkpoint inventories retain their existing FNV hashes and format.
Both ranks must run the same binary. Candidate queues remain bounded at 1 GiB
with producer backpressure and no dropping/spilling. Bucket waits aggregate
candidate traffic before routing it, including under the tiny-queue test.

Checkpoint copying, checksumming and fsync use bounded parallel workers,
default four per rank (`HD_MPI_CHECKPOINT_THREADS=1..8`). Publication still
waits for all workers, and errors abort instead of publishing a partial image.
All snapshot files and directories remain explicitly synced.

`submit_phase_bench.sh` queues separate jobs using `mpi_phase_bench.sbatch`:
one real serial sieve phase (one node/four GPUs), followed by one two-rank
phase (two nodes/eight GPUs). MPI uses an `afterany` dependency so a serial
failure does not suppress the independent measurement, and the pair never
reserves more than two nodes simultaneously. Export `SOURCE_COMMIT` and
`VALIDATED_BUILD` before running the helper with `bash`. Each job uses
the same input basis, global pool-size target and algorithm settings, reports
sampling separately, and excludes checkpoint/insertion I/O from the sieve
timer. Randomized pools differ between partitionings; this is a throughput
comparison, not a bit-identical trajectory. Default CSD is 128. No production
pilot should be restarted on an assumption of speedup: inspect the measured
phase times and transfer counts first. Capacity and solution quality are not
established by this benchmark.

Each backend has a three-hour test timeout (`BENCH_TIMEOUT_SECONDS`, default
10800), including initial sampling, inside a four-hour Slurm allocation.
`BENCH_VALIDATION_JOB` optionally gates both jobs on successful validation;
MPI still runs after a serial timeout/failure once that gate has passed.
Validation executes the actual serial and MPI frontends at CSD60, checking
normal completion, no-spill cleanup and rejection of previously claimed scratch.
`PHASE_BENCH_ONLY=1` runs just these frontend tests after the build. Each rank
atomically claims its scratch directory before MPI runtime initialization
creates rank-local logs. Only the disposable benchmark pool is discarded after
timing, preventing dirty-cache destructor waits when disk writes are forbidden;
production BKZ checkpoint behavior is unchanged.

With `HD_FAIL_ON_SPILL=1`, SWC can borrow the phase's unused PWC chunk
reservation. The phase pool bound matches the bucket iterator, including the
local MPI shard and an oversized existing pool. Borrowing is limited by the
original combined PWC+SWC budget and by the reducer's solution bound plus
worker/prefetch headroom; it does not allocate an additional PWC-sized arena,
shrink the vector target, or enable disk spill. Explicit `HD_SWC_GROW_GB` still
requests extra memory separately under its existing physical-memory guard.

This fixes the cold CSD128 benchmark's logical/cache capacity mismatch:
the reducer could retain about 39,700 solution chunks, while the fixed 16 GiB
SWC held only 13,273. It now borrows about 32 GiB of otherwise unused pool
reservation. At larger dimensions unused PWC space may be insufficient to
cover the entire solution bound; this does not prove CSD144 feasibility.
Timeout diagnostics identify the manager kind and chunk totals. Sampling start
and completion are logged separately because a cold pool is not comparable
to the much shorter warm CSD128 phase in a progressive BKZ pump.

The phase benchmark defaults to serial with a one-node allocation. Direct MPI
submission requires `--rsc=g=8` and `BENCH_BACKEND=mpi`; the script rejects a
backend/allocation mismatch. Both backends explicitly request four GPUs per
participating node, and step creation fails after 60 seconds rather than
waiting for the full sieve timeout on an impossible resource request.

Diagnostic launches additionally run `phase_bench_rank.sh` on each allocated
node. It records GPU utilization, process/thread CPU and wait states, RSS and
I/O counters every minute, plus bounded GDB thread backtraces every 20 minutes
when available. The launcher is archived from `BENCH_LAUNCH_COMMIT`, separately
from the already-validated binary source. Attaching GDB slightly perturbs the
timing; treat these reruns as diagnostic, not clean performance measurements.

## MPI interconnect diagnostics and host-runtime launch

`mpi_transport_probe.sbatch` compares UCX device discovery and bounded 1 GiB
bidirectional transfers on two allocated nodes. Old container runs silently
used `tcp/ib0` and `tcp/ib1`: the image lacked Mellanox/netlink dependencies,
and even after supplying them its unprivileged user namespace could not open
the host RDMA devices. Seeing `/dev/infiniband` is not sufficient to establish
that verbs actually works. The probe retains device-access checks and UCX
endpoint logs instead of assuming that an IB-named network interface is RDMA.

An optional `MPI_HOST_RUNTIME` directory selects a host-namespace launcher in
`mpi_rank.sh`. `tools/bundle_mpi_runtime.py`, run **inside the existing trusted
image on a compute node**, copies that executable's image userspace dependencies
and loader into a fresh directory and records SHA256/source paths in a manifest.
The original binary is executed by that loader as the ordinary cluster UID;
host MPI, UCX, GPU drivers and RDMA providers remain host-provided. No system
libraries, permissions, container image or checkpoint files are modified.
Algorithm `APPTAINERENV_HD_*`, `UCX_*` and `OMP_*` settings are translated for the
host launch. The existing container launcher remains the default.

The probe rejects a host-runtime test that still selects TCP, and runs an actual
small GPU/MPI phase before the launcher is accepted. Full integration validation
must pass with the selected runtime before a new high-dimensional run. Pin
`UCX_TLS=rc,sm,self` for a host-runtime production test to fail if RDMA is absent
instead of silently falling back. This mode does not enable GPU-direct MPI;
the existing bounded pinned-host wire protocol is unchanged.

MPI phase benchmarks now emit one-minute local heartbeats with bucket/ownership
exchange counters, pending candidate bytes, transferred bytes and processed pool
chunks. The heartbeat thread never calls MPI. Counter movement distinguishes
slow transport from a true progress deadlock; GPU utilization alone cannot.
