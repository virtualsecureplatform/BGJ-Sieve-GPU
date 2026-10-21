#!/usr/bin/env python3
"""Smoke-test the actual serial/MPI benchmark frontends in a two-node job."""
import argparse
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wrapper', type=Path, required=True)
    parser.add_argument('--basis', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('requires a two-node Slurm allocation')
    binary_dir = Path(os.environ['MPI_BINARY']).parent
    args.root.mkdir(parents=True, exist_ok=True)
    for backend, nodes, binary in [('serial', 1, 'hd_phase_bench'),
                                   ('mpi', 2, 'hd_phase_bench_mpi')]:
        scratch = args.root / backend
        env = dict(os.environ, MPI_BINARY=str(binary_dir / binary),
                   APPTAINERENV_HD_HBM_PWC_WRITEBACK='0',
                   APPTAINERENV_HD_HBM_PWC_GB='0',
                   APPTAINERENV_HD_HBM_BWC_GB='12')
        launch = ['srun', '--mpi=pmix_v2', '--immediate=60',
                  '--nodes=%d' % nodes, '--ntasks=%d' % nodes,
                  '--ntasks-per-node=1', '--cpus-per-task=128',
                  '--gpus=%d' % (nodes*4), '--gres=gpu:4',
                  '--cpu-bind=none', '--kill-on-bad-exit=1', str(args.wrapper)]
        command = launch + [str(args.basis), '60', str(scratch)]
        result = subprocess.run(command, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=600)
        (args.root / (backend + '.log')).write_bytes(result.stdout)
        text = result.stdout.decode(errors='replace')
        assert result.returncode == 0, (result.returncode, text[-5000:])
        assert '[PHASE-BENCH] ranks=%d csd=60 ' % nodes in text, text[-5000:]
        # Duplicate-run rejection must occur before MPI runtime creates any
        # files, and must never delete a prior job's scratch directory.
        marker = scratch / 'rank0' / 'keep-marker'
        marker.write_text('preserve existing benchmark scratch')
        repeated = subprocess.run(command, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, timeout=120)
        (args.root / (backend + '-duplicate.log')).write_bytes(repeated.stdout)
        assert repeated.returncode != 0
        assert b'benchmark scratch already exists' in repeated.stdout, repeated.stdout[-5000:]
        assert marker.read_text() == 'preserve existing benchmark scratch'
        for rank in range(nodes):
            for kind in ('.pool', '.bucket', '.sol', '.uid'):
                assert not any(path.is_file() for path in
                               (scratch / ('rank%d' % rank) / kind).rglob('*'))
        print('PASS: %s phase benchmark startup, completion, cleanup and stale-scratch rejection' % backend,
              flush=True)


if __name__ == '__main__':
    main()
