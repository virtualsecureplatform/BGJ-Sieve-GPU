#!/usr/bin/env python3
"""Gate paired checkpoint measurements without conflating bytes and wall time."""
import argparse
import json
import re
import statistics
from pathlib import Path

RESULT = re.compile(r'\[MPI-CHECKPOINT-BENCH\] pair=(\d) writer=(legacy|direct) '
                    r'seconds=([0-9.]+) pool_write_bytes_per_rank=(\d+)')

def summarize(text):
    records = [(int(pair),writer,float(seconds),int(size))
               for pair,writer,seconds,size in RESULT.findall(text)]
    expected = {(pair,writer) for pair in range(4) for writer in ('legacy','direct')}
    if len(records) != 8 or {(r[0],r[1]) for r in records} != expected:
        raise ValueError('one warmup and three alternating checkpoint pairs required')
    times = {writer: statistics.median(r[2] for r in records if r[0] and r[1] == writer)
             for writer in ('legacy','direct')}
    sizes = {writer: {r[3] for r in records if r[1] == writer} for writer in ('legacy','direct')}
    if any(len(values) != 1 for values in sizes.values()):
        raise ValueError('checkpoint workload sizes differ')
    legacy, direct = (next(iter(sizes[w])) for w in ('legacy','direct'))
    if times['legacy'] <= 0 or legacy <= 0:
        raise ValueError('empty checkpoint benchmark')
    return {'legacy_median_seconds': times['legacy'], 'direct_median_seconds': times['direct'],
            'time_improvement': 1-times['direct']/times['legacy'],
            'serialized_pool_write_reduction': 1-direct/legacy,
            'passed': times['direct'] <= .85*times['legacy'] and direct <= .6*legacy}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = summarize(args.log.read_text())
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not result['passed']: raise SystemExit('Checkpoint performance gate did not pass')

if __name__ == '__main__': main()
