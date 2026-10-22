#!/usr/bin/env python3
"""Small two-node GPU integration tests. Run only inside a Slurm allocation."""
import argparse
import os
from pathlib import Path
import shutil
import struct
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wrapper', type=Path, required=True)
    parser.add_argument('--basis', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cache-pressure-only', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        parser.error('requires a two-node Slurm allocation')
    args.root.mkdir(parents=True, exist_ok=True)
    launch = ['srun', '--mpi=pmix_v2', '--nodes=2', '--ntasks=2', '--ntasks-per-node=1',
              '--cpus-per-task=128', '--cpu-bind=none', '--kill-on-bad-exit=1', str(args.wrapper)]

    def run(name, options, overrides=None, expected=0, frame_mib=1):
        root = args.root / name
        root.mkdir(exist_ok=True)
        env = dict(os.environ)
        env.update(overrides or {})
        output = [] if '--task' in options and options[options.index('--task')+1] == 'sieve' else [
            '--output', str(root/'out.basis')]
        command = launch + ['--input', str(args.basis)] + output + [
            '--checkpoint-root', str(root), '--mpi-frame-mib', str(frame_mib)] + options
        result = subprocess.run(command, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=1200)
        with (root/'test.log').open('ab') as log:
            log.write(result.stdout)
        if expected is None:
            assert result.returncode != 0, result.stdout.decode(errors='replace')[-4000:]
        else:
            assert result.returncode == expected, (result.returncode, result.stdout.decode(errors='replace')[-4000:])
        print('PASS: %s returncode=%d' % (name, result.returncode), flush=True)
        return root

    pump = ['--task', 'pump', '--ind_l', '0', '--ind_r', '80', '--SSD', '58', '--MSD', '60', '--DS']

    def cache_pressure():
        cache = run('pipeline-cache-pressure', pump, {'APPTAINERENV_HD_MPI_CONTROL_TEST': '1',
                    'APPTAINERENV_HD_MPI_PIPELINE_CACHE_TEST': '1'})
        assert 'PASS: async SWC/UID pressure' in (cache/'test.log').read_text()

    if args.cache_pressure_only:
        if os.environ.get('APPTAINERENV_HD_MPI_PIPELINE') != 'async':
            parser.error('--cache-pressure-only requires the async pipeline')
        cache_pressure()
        return

    def check_snapshot_pool(root, sequence):
        # The cache serializer places scores immediately after the 12-byte
        # header, including in its padded direct-I/O layout.
        for rank in (0, 1):
            shard = root/('generation-%d/rank%d' % (sequence, rank))
            expected = int((shard/'STATE').read_text().split()[5])
            total = 0
            for path in (shard/'pool').rglob('*'):
                if not path.is_file():
                    continue
                with path.open('rb') as stream:
                    header = stream.read(12)
                    scores = struct.unpack('<8192H', stream.read(8192*2))
                count = struct.unpack('<H', header[:2])[0]
                assert sum(score != 0 for score in scores) == count, path
                assert all(scores[i] != 0 for i in range(count)), path
                total += count
            assert total == expected, (rank, total, expected)

    run('control', pump, {'APPTAINERENV_HD_MPI_CONTROL_TEST': '1'})
    run('control-batched', pump, {'APPTAINERENV_HD_MPI_CONTROL_TEST': '1'}, frame_mib=8)
    if os.environ.get('APPTAINERENV_HD_MPI_PIPELINE') == 'async':
        mismatch_wrapper = args.root/'mode-mismatch.sh'
        mismatch_wrapper.write_text('#!/bin/bash\nset -euo pipefail\n'
            'if [[ "${SLURM_PROCID}" == 1 ]]; then export APPTAINERENV_HD_MPI_PIPELINE=legacy; fi\n'
            'exec "${MPI_TEST_WRAPPER:?}" "$@"\n')
        mismatch_wrapper.chmod(0o755)
        original_wrapper = launch[-1]
        launch[-1] = str(mismatch_wrapper)
        try:
            mismatch = run('pipeline-mode-mismatch', pump, {'MPI_TEST_WRAPPER': str(args.wrapper)}, expected=None)
            assert 'pipeline version/frame settings differ' in (mismatch/'test.log').read_text()
        finally:
            launch[-1] = original_wrapper
        cache_pressure()
        run('control-delayed', pump, {'APPTAINERENV_HD_MPI_CONTROL_TEST': '1',
                                     'APPTAINERENV_HD_MPI_PIPELINE_TEST_DELAY_US': '2000'})
        corrupted = run('control-corrupt-frame', pump, {
            'APPTAINERENV_HD_MPI_CONTROL_TEST': '1',
            'APPTAINERENV_HD_MPI_PIPELINE_TEST_CORRUPT': '1'}, expected=None)
        logs = (corrupted/'test.log').read_text()
        for diagnostic in corrupted.glob('rank*/rank.err'):
            logs += diagnostic.read_text()
        assert 'checksum mismatch' in logs, logs[-4000:]
    run('control-mismatch', pump, {'APPTAINERENV_HD_MPI_CONTROL_TEST': 'mismatch'}, expected=None)
    pressure = run('candidate-backpressure', pump, {
        'APPTAINERENV_HD_MPI_CANDIDATE_QUEUE_KIB': '1',
        'APPTAINERENV_HD_MPI_BKZ_STOP_AFTER': 'top'}, 75)
    diagnostics = (pressure/'test.log').read_text()
    for diagnostic in pressure.glob('rank*/rank.err'):
        diagnostics += diagnostic.read_text()
    assert '[MPI-CANDIDATE-BACKPRESSURE]' in diagnostics
    for kind in ('pool', 'bucket', 'sol', 'uid'):
        forbidden = run('forbidden-' + kind, pump, {'APPTAINERENV_HD_MPI_SPILL_TEST': kind}, expected=None)
        logs = (forbidden/'test.log').read_text()
        for diagnostic in forbidden.glob('rank*/rank.err'):
            logs += diagnostic.read_text()
        assert '[CACHE-SPILL-FORBIDDEN]' in logs
        for directory in forbidden.glob('rank*/.*'):
            if directory.is_dir():
                assert not any(p.is_file() for p in directory.rglob('*')), directory
    root = run('phase-resume', pump, {'APPTAINERENV_HD_MPI_BKZ_STOP_AFTER': 'top'}, 75)
    seq = int((root/'LATEST-BKZ').read_text())
    check_snapshot_pool(root, seq)
    state = (root/('generation-%d/rank0/STATE' % seq)).read_text().split()
    assert int(state[9]) == 60 and int(state[11]) == 2
    run('phase-resume', pump, {'APPTAINERENV_HD_MPI_BKZ_MAX_PHASES': '1'}, 75)
    seq2 = int((root/'LATEST-BKZ').read_text())
    check_snapshot_pool(root, seq2)
    assert seq2 > seq
    state = (root/('generation-%d/rank0/STATE' % seq2)).read_text().split()
    assert int(state[9]) == 59 and int(state[11]) == 3
    run('phase-resume', pump)
    assert (root/'out.basis').is_file()
    # Replay a completed checkpoint without repeating any pump phases.
    run('phase-resume', pump)
    # Mutated input/options must not reinterpret an existing checkpoint.
    changed = list(pump); changed[changed.index('--MSD')+1] = '61'
    run('phase-resume', changed, expected=None)
    run('bad-dual-hash', ['--task', 'bkz', '--BSD', '60', '--D4F', '20', '--BDH', '300'], expected=None)
    # Exercise genuine multi-pump control with an exact lattice check per pump.
    run('bkz-two-pumps', ['--task', 'bkz', '--BSD', '60', '--D4F', '20', '--BDH', '0',
                         '--JUMP', '9', '--STI', '0', '--STOP_INDEX', '18'])
    # A crash before publication must leave the earlier committed generation usable.
    for fault in ('after-rank-snapshot', 'before-publish'):
        name = 'fault-' + fault
        bad = run(name, pump, {'APPTAINERENV_HD_MPI_BKZ_FAULT': fault}, expected=None)
        assert (bad/'LATEST-BKZ').is_file()
        run(name, pump)
    # Missing or corrupt rank shards must fail before pool loading.
    for corruption in ('missing', 'corrupt'):
        name = 'bad-' + corruption
        bad = args.root/name
        shutil.copytree(root, bad)
        seq = int((bad/'LATEST-BKZ').read_text())
        snapshot = bad/('generation-%d/rank1' % seq)
        if corruption == 'missing': shutil.rmtree(snapshot)
        else:
            with (snapshot/'full.qp').open('ab') as f: f.write(b'corrupt')
        run(name, pump, expected=None)
    # Existing final-sieve entrypoint remains usable with immutable v1 checkpoints.
    run('final-sieve', ['--task', 'sieve', '--TSD', '61', '--MLD', '80'])
    print('PASS: all MPI BKZ integration tests', flush=True)


if __name__ == '__main__':
    main()
