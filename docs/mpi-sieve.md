# Two-node MPI sieve

## Experimental asynchronous pipeline

UID-checker capacity changes are permitted only with producers stopped and
queues/workers drained. Resizing preserves the sentinel layout, atomically
replaces descriptor storage, frees only owned UID buffers, and rejects busy or
invalid requests without losing queued work. Allocation failure preserves the
previous capacity. The one-GPU validation build runs executable ASan/UBSan
resize tests before releasing any dependent two-node tests; targeted async
cache/UID pressure runs before the full integration suites.

`HD_MPI_PIPELINE=async` enables the opt-in two-node pipeline; the default is
`legacy`. Both ranks must agree on the mode and `--mpi-frame-mib`. The host
runtime/RDMA launcher is unchanged. Do not promote this mode to production
until the replay and cold CSD128 validation gates below have passed.

The Bucketer coordinator owns all phase MPI calls with
`MPI_THREAD_SERIALIZED`. Two CPU workers pack/check/apply bucket frames while
a separate worker applies remote candidates. Each completed owner bucket
wakes GPU reducers immediately. Bucket and candidate channels each have two
send and two receive slots; receiver acknowledgements are returned only after
application, preventing unlimited MPI buffering behind a slow cache.

At the default 256 MiB frame setting, bucket slots are 64 MiB and candidate
slots are 16 MiB, for 320 MiB reusable pinned storage plus two 1 MiB buffers
for synchronous traffic outside waves. Smaller frame settings scale the slots
down. Buckets may span frames without rounding coordinates or changing norms.
The existing `HD_MPI_CANDIDATE_QUEUE_KIB` cap includes queued and in-flight
remote records until peer application. Partial candidate batches are sent.

While copied buckets are being reduced, bounded insertion passes can relieve
SWC pressure without new GPU pool readers or the final-cleanup discard pass.
The current replacement threshold remains fixed for that wave. No new lossy
filter, pool-size reduction, or dimension-limit change is enabled. Final
candidate routing and acknowledgements complete before wave termination.
Fatal protocol/cache errors abort both ranks; previously completed checkpoints
remain the recovery point. Checkpoint source-identity checks are not relaxed.

`[MPI-PIPELINE]` reports per-wave elapsed time, packing/bucket-application/
candidate-application worker times, frame counts and candidate high-water
bytes. Worker times overlap and must not be added to infer elapsed time.
`[MPI-PRESSURE]` separates checksum work, SWC acquisition waits, UID commit
waits and ready-to-dequeue latency; benchmark heartbeats expose pressure
counters even before a wave completes.

Validation uses `VALIDATE_PIPELINES="legacy async"` with the existing
`mpi_bkz_validate.sbatch` allocation and the validated `MPI_HOST_RUNTIME`.
It runs serial/MPI CSD60 frontend tests and the full MPI/BKZ suite for each
mode; async additionally tests delayed application and corrupted frames.
An additional async test limits the real SWC to 256 chunks and the real UID
checker to eight held chunks, with two concurrent producers and a delayed
independent consumer. It must reach cache pressure and deliver every record
without spilling. Async SWC prefetch reserves live-chunk permits, including
UID-held and pending-prefetch chunks, and waits instead of growing past its
cache capacity.
After validation, `mpi_pipeline_replay.sbatch` uses that exact build for one
warmup pair and three alternating measured pairs per dimension. Its
`summary.json` records the medians and fails the job if either dimension
misses the performance threshold. The replay verifies every coordinate,
norm, candidate score and UID, plus missing/duplicate records, without disk
spill; actual UID/SWC behavior remains covered by the GPU integration suite.
Larger runs must remain disabled until frozen-data CSD128/CSD144 replay shows
at least 20% lower median elapsed time over three alternating pairs, and the
same cold seed-38 CSD128 phase completes within three hours without spill or
integrity errors. A completed benchmark alone does not establish a speedup
over the single-node implementation.

`submit_mpi_pipeline_validation.sh` submits the complete sequential chain
with `SOURCE_COMMIT` and `MPI_HOST_RUNTIME` supplied: one-GPU compile, two-node
async/legacy integration, two-node replay, then cold CSD128 only if replay
passes. Dependencies prevent overlapping node allocations; failed gates
cancel later jobs. Timing runs disable debugger attachment. No BKZ144 job is
submitted by this helper.

## Existing topology and checkpoint behavior

`app/hd_sieve_mpi` distributes only the final GPU sieve. Preprocessing remains
the existing PotLLL/BKZ job. The supported production topology is two Slurm
nodes, one MPI rank and four visible A100 GPUs per node.

Vectors and UID-table entries are owned by `normalize(uid) % 2`. Both ranks
scan their local pool shard against the same centers. A logical bucket is owned
by its global sequence number modulo two; its peer fragment is transferred as
exact int8 coordinates plus int32 norms before the bucket becomes visible to
the existing reducer. Reducer output is routed again by UID ownership.

Inter-node payloads use bounded pinned-host MPI frames. The transport choice is
intentional: the cluster probe measured about 40 GB/s aggregate for pinned host
buffers and about 36 GB/s for direct CUDA buffers. Local four-GPU processing is
unchanged and NCCL is not required.

Build and launch through `run_mpi_a100x8.sbatch`. It uses the CUDA/NTL
Apptainer image together with the cluster's Open MPI headers and libraries, and
launches through `srun --mpi=pmix_v2`. Large pool/checkpoint files are kept
under `/LARGE0`; rank-local working directories prevent filename collisions.

The MPI-only options are:

- `--checkpoint-root PATH` (required)
- `--resume auto|none` (default `auto`)
- `--mpi-frame-mib N` (default 256, range 1 through 1024)

At every completed CSD, each rank force-flushes its local pool and atomically
writes a rank manifest. Rank zero publishes `LATEST` only after both ranks have
completed the checkpoint. `--resume auto` validates the two-rank generation,
rebuilds ephemeral UID state from the stored vectors, and continues at the next
CSD rather than repeating the completed one. Only the newest complete
generation is retained.

For an initial run, `--resume none` creates `RUN_ROOT/run-$SLURM_JOB_ID`, so a
later submission should set `RUN_ROOT` to that directory and use
`--resume auto`.
