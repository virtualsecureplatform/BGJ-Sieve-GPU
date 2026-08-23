# Two-node MPI sieve

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
