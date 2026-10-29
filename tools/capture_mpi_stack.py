#!/usr/bin/env python3
"""Capture application stacks for a rank launched through an explicit loader.

Run on the rank's compute node, with the same UID as the rank process.
"""
import argparse
from pathlib import Path
import subprocess
import re


def load_bias(mappings, binary):
    for line in mappings.splitlines():
        fields = line.split(None, 5)
        if len(fields) == 6 and fields[5] == str(binary) and int(fields[2], 16) == 0:
            return int(fields[0].split('-')[0], 16)
    raise RuntimeError('application mapping at file offset zero not found')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pid', type=int)
    parser.add_argument('binary', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    mappings = Path('/proc/%d/maps' % args.pid).read_text()
    bias = load_bias(mappings, binary)
    args.output.with_suffix(args.output.suffix + '.maps').write_text(mappings)
    # /proc/PID/exe points at ld-linux for this launch method. Loading FILE
    # without relocation makes GDB look for application frames at wrong PCs.
    quoted = str(binary).replace('\\', '\\\\').replace('"', '\\"')
    commands = ['set pagination off', 'set print elements 12',
                'attach %d' % args.pid,
                'add-symbol-file "%s" -o 0x%x' % (quoted, bias),
                'info address g_pipeline_pending',
                'p g_pipeline_pending', 'p g_ready_batches',
                'p g_candidate_queued_bytes', 'thread apply all bt 24', 'detach']
    command = ['gdb', '-q', '-n', '-batch']
    for expression in commands:
        command += ['-ex', expression]
    with args.output.open('w') as output:
        subprocess.run(command, stdout=output, stderr=subprocess.STDOUT,
                       check=True, timeout=45)
    trace = args.output.read_text()
    swc = re.search(r'mpi_sieve_exchange_buckets \(bwc=0x[0-9a-f]+, swc=(0x[0-9a-f]+)', trace)
    bucketer = re.search(r'Bucketer_t::run \(this=(0x[0-9a-f]+)', trace)
    extra = commands[:4]
    if swc:
        for field in ('_max_cached_chunks', '_mpi_live_chunks', '_num_chunks',
                      '_num_deleted_ids', '_num_writing', '_num_wp', '_num_wl',
                      '_num_ready', '_num_rp', '_num_rl'):
            extra.append('p ((swc_manager_tmpl<int>*)%s)->%s' % (swc.group(1), field))
    if bucketer:
        for field in ('_mpi_insert_workers', '_mpi_service_ticks'):
            extra.append('p ((Bucketer_t*)%s)->%s' % (bucketer.group(1), field))
    extra.append('detach')
    command = ['gdb', '-q', '-n', '-batch']
    for expression in extra:
        command += ['-ex', expression]
    with args.output.open('a') as output:
        output.write('\nSWC and insertion state (fields in command order):\n'); output.flush()
        subprocess.run(command, stdout=output, stderr=subprocess.STDOUT,
                       check=True, timeout=45)


if __name__ == '__main__':
    main()
