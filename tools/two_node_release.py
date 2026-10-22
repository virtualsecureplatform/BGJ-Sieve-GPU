#!/usr/bin/env python3
"""Produce a fail-closed, source-bound release report from Slurm test artifacts."""
import argparse
import json
import math
from pathlib import Path
import re

from summarize_checkpoint_bench import summarize as checkpoint_summary


def cold_metrics(root):
    text = (root/'mpi-128.log').read_text()
    matches = re.findall(r'\[PHASE-BENCH\] ranks=2 csd=128 sampling_seconds=([0-9.]+) '
                         r'sieve_seconds=([0-9.]+) vectors=(\d+) sieve_rc=(-?\d+)',text)
    if len(matches) != 1 or int(matches[0][3]) not in (0,-1):
        raise ValueError('cold CSD128 did not finish successfully')
    if '[CACHE-SPILL-FORBIDDEN]' in text or 'CUDA error' in text:
        raise ValueError('cold CSD128 has a cache/GPU failure')
    peaks = []
    for rank in (0,1):
        samples = re.findall(r'VmHWM:\s+(\d+)\s+kB',
            (root/('mpi-128.rank%d.monitor.log' % rank)).read_text())
        if not samples: raise ValueError('missing per-rank peak memory samples')
        peaks.append(max(map(int,samples))/(1024*1024))
    sampling,sieving,vectors,_ = matches[0]
    target = re.findall(r'sampling_start ranks=2 csd=128 target_vectors=(\d+)',text)
    if len(target) != 1 or int(target[0]) != int(3.2*(4./3.)**64-5):
        raise ValueError('cold CSD128 pool size ratio/workload changed')
    if not int(target[0]) <= int(vectors) < int(target[0])+65536:
        raise ValueError('cold CSD128 pool lost records or exceeded its size bound')
    return {'sampling_seconds':float(sampling),'sieve_seconds':float(sieving),
            'vectors':int(vectors),'peak_host_gib':peaks,'sieve_rc':int(matches[0][3])}


def choose_profile(replay,baseline,optimized):
    time_ok = optimized['elapsed_seconds'] <= 1.05*baseline['elapsed_seconds']
    memory_ok = all(new <= old+2 for old,new in zip(baseline['peak_host_gib'],optimized['peak_host_gib']))
    replay_ok = set(replay) == {'128','144'} and all(replay[d]['passed'] for d in ('128','144'))
    return ('optimized' if replay_ok and time_ok and memory_ok else 'baseline',time_ok,memory_ok)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit',required=True)
    for name in ('safety-log','integration-log','phase-log','phase-root','replay-summary',
                 'checkpoint-log','cold-log','baseline-root','optimized-root','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    checks = ((args.safety_log,'PASS: memory-safety validation; source='),
              (args.integration_log,'PASS: distributed BKZ validation source='),
              (args.phase_log,'PASS: CSD133 source='))
    for path,prefix in checks:
        if prefix+args.source_commit not in path.read_text():
            raise ValueError('missing source-matched validation: '+str(path))
    sequence = int((args.phase_root/'LATEST-BKZ').read_text())
    states = [(args.phase_root/('generation-%d/rank%d/STATE' % (sequence,r))).read_text().split()
              for r in (0,1)]
    if not all(int(s[9]) == 133 and int(s[11]) == 1 for s in states) or states[0][6:] != states[1][6:]:
        raise ValueError('CSD133 phase checkpoint is missing/inconsistent')
    baseline,optimized = cold_metrics(args.baseline_root),cold_metrics(args.optimized_root)
    times = re.findall(r'\[MPI-COLD-COMPARE\] profile=(baseline|optimized) elapsed_seconds=(\d+)',
                       args.cold_log.read_text())
    if len(times) != 2 or {p for p,_ in times} != {'baseline','optimized'}:
        raise ValueError('missing paired end-to-end cold timings')
    for profile,seconds in times:
        if int(seconds) <= 0: raise ValueError('invalid cold elapsed time')
        (baseline if profile == 'baseline' else optimized)['elapsed_seconds'] = int(seconds)
    replay = json.loads(args.replay_summary.read_text())
    if set(replay) != {'128','144'} or any(
        not isinstance(replay[d].get('passed'),bool) or any(
            not math.isfinite(replay[d].get(field,0)) or replay[d].get(field,0) <= 0
            for field in ('baseline_median_seconds','optimized_median_seconds'))
        for d in ('128','144')):
        raise ValueError('incomplete async replay evidence')
    profile,time_ok,memory_ok = choose_profile(replay,baseline,optimized)
    checkpoints = checkpoint_summary(args.checkpoint_log.read_text())
    result = {'source_commit':args.source_commit,'passed':True,
              'async_profile':profile,'checkpoint_writer':'direct' if checkpoints['passed'] else 'legacy',
              'checkpoint_root':str(args.phase_root),'checkpoint_generation':sequence,
              'baseline_cold':baseline,'optimized_cold':optimized,
              'optimized_cold_time_passed':time_ok,'optimized_memory_passed':memory_ok,
              'replay':replay,'checkpoints':checkpoints,
              'evidence':{name:str(getattr(args,name)) for name in
                          ('safety_log','integration_log','phase_log','replay_summary','checkpoint_log','cold_log')}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__ == '__main__': main()
