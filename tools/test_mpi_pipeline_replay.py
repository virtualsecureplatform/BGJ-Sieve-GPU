#!/usr/bin/env python3
"""Frozen two-node transport/CPU replay; requires a Slurm GPU allocation."""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess

RESULT = re.compile(r'\[MPI-REPLAY\] csd=(\d+) mode=(legacy|async) '
                    r'elapsed_seconds=([0-9.]+) bucket_records=(\d+) candidate_records=(\d+)')


def summarize(results):
    summary = {}
    for dimension in (128, 144):
        times = {mode: [r['seconds'] for r in results if r['csd'] == dimension and
                        r['mode'] == mode and not r['warmup']] for mode in ('legacy', 'async')}
        if any(len(values) != 3 for values in times.values()):
            raise ValueError('three measured pairs per dimension are required')
        legacy, asynchronous = (statistics.median(times[mode]) for mode in ('legacy', 'async'))
        summary[str(dimension)] = {'legacy_median_seconds': legacy,
                                   'async_median_seconds': asynchronous,
                                   'improvement': 1-asynchronous/legacy,
                                   'passed': asynchronous <= .8*legacy}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wrapper', type=Path, required=True)
    parser.add_argument('--basis', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('run only inside a two-node Slurm allocation')
    args.root.mkdir(parents=True, exist_ok=False)
    results = []
    launch = ['srun', '--mpi=pmix_v2', '--immediate=60', '--nodes=2', '--ntasks=2',
              '--ntasks-per-node=1', '--cpus-per-task=128', '--gpus=8', '--gres=gpu:4',
              '--cpu-bind=none', '--kill-on-bad-exit=1', str(args.wrapper)]
    for dimension in (128, 144):
        # One warmup pair, then three measured pairs with alternating order.
        for pair in range(4):
            order = ('legacy', 'async') if pair % 2 == 0 else ('async', 'legacy')
            for mode in order:
                root = args.root / ('csd%d-pair%d-%s' % (dimension, pair, mode))
                env = dict(os.environ, APPTAINERENV_HD_MPI_PIPELINE=mode,
                           APPTAINERENV_HD_MPI_CONTROL_TEST='1',
                           APPTAINERENV_HD_MPI_PIPELINE_REPLAY='1',
                           APPTAINERENV_HD_MPI_REPLAY_CSD=str(dimension),
                           APPTAINERENV_HD_HBM_BWC_GB='0',
                           APPTAINERENV_HD_HBM_PWC_WRITEBACK='0',
                           APPTAINERENV_HD_MPI_CANDIDATE_QUEUE_KIB='1048576')
                env.pop('APPTAINERENV_HD_MPI_PIPELINE_TEST_DELAY_US', None)
                env.pop('APPTAINERENV_HD_MPI_PIPELINE_TEST_CORRUPT', None)
                command = launch + ['--input', str(args.basis), '--output', str(root/'unused.basis'),
                                    '--checkpoint-root', str(root), '--resume', 'none',
                                    '--mpi-frame-mib', '256', '--task', 'pump', '--ind_l', '0',
                                    '--ind_r', '80', '--SSD', '58', '--MSD', '60', '--DS']
                completed = subprocess.run(command, env=env, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, timeout=1200)
                text = completed.stdout.decode(errors='replace')
                (args.root / (root.name+'.log')).write_text(text)
                matches = list(RESULT.finditer(text))
                if completed.returncode or len(matches) != 1:
                    raise RuntimeError('%s failed: rc=%d\n%s' % (root.name, completed.returncode, text[-5000:]))
                match = matches[0]
                if int(match[1]) != dimension or match[2] != mode:
                    raise RuntimeError('replay identity mismatch')
                results.append({'csd': dimension, 'mode': mode, 'pair': pair, 'warmup': pair == 0,
                                'seconds': float(match[3]), 'bucket_records': int(match[4]),
                                'candidate_records': int(match[5])})
                (args.root/'results.json').write_text(json.dumps(results, indent=2)+'\n')
                print(match[0], flush=True)
    counts = {(r['bucket_records'], r['candidate_records']) for r in results}
    if len(counts) != 1:
        raise RuntimeError('replay workloads differ')
    summary = summarize(results)
    (args.root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)
    if not all(item['passed'] for item in summary.values()):
        raise SystemExit('Replay correctness passed, but the 20% performance gate did not pass.')


if __name__ == '__main__':
    main()
