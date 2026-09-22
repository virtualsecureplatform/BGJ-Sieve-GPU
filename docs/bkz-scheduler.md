# Progressive BKZ scheduler

`tools/bkz_scheduler.py` runs a checkpointed sequence of BGJ-backed pnj-BKZ
tours. It is an orchestration tool: invoke it only in a Slurm GPU allocation,
not on a login node. Each stage calls the existing `hd_sieve --task bkz` path.

The scheduler uses paper-style block size `beta`. BGJ's command line instead
splits this into `BSD` (the maximum sieve dimension) and `D4F`, so
`beta = BSD + D4F`. If `d4f` is omitted, the default from Algorithm 9 of Wang,
Wang, and Wang (ASIA CCS 2023) is used. Explicit `sieve_dimension` may be used
instead of `block_size`.

Inspect a schedule safely on a login node:

```bash
python3 tools/bkz_scheduler.py \
  --input /path/to/basis \
  --schedule example/bkz-schedule.json \
  --work-dir /path/to/job-work \
  --dry-run
```

Run the same command without `--dry-run` inside a GPU job. The work directory
must contain the hardware-specific `.pool`, `.bucket`, `.sol`, and `.uid`
directories required by `hd_sieve`. Completed tours are retained under
`WORK_DIR/bkz-checkpoints`; rerunning the command resumes at the first missing
checkpoint. Use `--output /path/to/reduced.basis` to copy out the final basis.
Existing output files are never overwritten.

[`example/bkz-scheduler.sbatch`](../example/bkz-scheduler.sbatch) is a generic
Slurm wrapper. For example, after adapting its `#SBATCH` resource lines:

```bash
BGJ_INPUT=/path/to/input.basis \
BGJ_OUTPUT=/path/to/reduced.basis \
sbatch example/bkz-scheduler.sbatch
```

The block-size schedules from Table 3 of the paper are built in for dimensions
130, 140, 150, and 154:

```bash
python3 tools/bkz_scheduler.py \
  --input /path/to/svp-150.basis \
  --paper-preset 150 \
  --binary /path/to/hd_sieve \
  --work-dir "$SLURM_TMPDIR/bgj-bkz" \
  --output /path/to/svp-150.progressive.basis
```

The preset is preprocessing only. Select the final pump/sieve dimension from
the available GPU memory and then run `hd_sieve --task sieve` on the reduced
basis. A custom JSON plan can run this step directly by adding, for example,
`"final_sieve": {"target_sieving_dimension": 120,
"min_lifting_dimension": 120}`. Do not assume that a paper schedule transfers unchanged to another BGJ
implementation or hardware profile; compare GSO slope and peak memory at each
checkpoint first.

Schedule fields are:

- `block_size` or `sieve_dimension` (exactly one is required per stage)
- `d4f`, `jump`, `tours`, `start_index`, and `dual_hash_ratio`
- top-level `defaults`, overridden by a stage
- optional top-level `final_sieve`, with `target_sieving_dimension`,
  `current_sieving_dimension`, and/or `min_lifting_dimension`

The scheduler deliberately provides no automatic cleanup. This preserves both
BGJ crash-recovery data and the user's checkpoints after preemption. A run
identity derived from the input contents and normalized schedule prevents a
changed input or plan from being mistaken for a resumable run.
