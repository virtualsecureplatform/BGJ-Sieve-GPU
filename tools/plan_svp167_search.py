#!/usr/bin/env python3
"""Screen PotLLL GSO profiles with the same heuristic used for SVP164.

No runtime or success guarantee: effective block size is an approximation.
All emitted schedules use BSD <= 143 and are executable by bkz_scheduler.py.
"""
import argparse
import ast
import contextlib
import copy
import hashlib
import io
import json
import math
from pathlib import Path


def simulate(profile, simulator, jump, target=3405.0, max_tours=8):
    n = len(profile)
    if n != 167 or not all(math.isfinite(x) for x in profile):
        raise ValueError("expected 167 finite natural-log GS lengths")
    current = list(profile)
    stages, predictions = [], []
    # Repeat the highest permitted dimension to assess continuation as well.
    progression = [60, 80, 100, 112, 120, 128, 138] + [143] * max_tours
    tours = {}
    for bsd in progression:
        centre = (n - 1) / 2
        slope = sum((i-centre)*x for i,x in enumerate(current)) / sum(
            (i-centre)**2 for i in range(n))
        bonus = max(0, int(math.log(math.sqrt(4/3)) / max(-slope/2, 1e-9)))
        beta = min(n-1, bsd + min(bonus, n-bsd))
        with contextlib.redirect_stdout(io.StringIO()):
            current = simulator(current, beta, 1, n, jump)
        norm = math.exp(current[0])
        tours[bsd] = tours.get(bsd, 0) + 1
        predictions.append(dict(bsd=bsd, tour=tours[bsd], effective_beta=beta,
                                predicted_norm=norm))
        last_start = jump * math.ceil((n-bsd-2)/jump)
        starts = list(range(0, last_start+1, jump))
        for pos, start in enumerate(starts):
            stage = dict(sieve_dimension=bsd, d4f=min(26,n-bsd), start_index=start)
            if pos+1 < len(starts): stage['stop_index'] = starts[pos+1]
            stages.append(stage)
        if norm <= target: break
    return dict(jump=jump, predicted_norm=norm, eligible=norm <= target,
                predictions=predictions, pump_count=len(stages),
                schedule=dict(name='svp167-potlll-progressive-j%d' % jump,
                              target_norm2=math.floor(target*target),
                              defaults=dict(jump=jump, dual_hash_ratio=0,tours=1),
                              stages=stages))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_root', type=Path)
    parser.add_argument('simulator_source', type=Path)
    parser.add_argument('--last-seed', type=int, default=7)
    args = parser.parse_args()
    if not 0 <= args.last_seed <= 63:
        parser.error('--last-seed must be between 0 and 63')
    source = args.simulator_source.read_text()
    node = next(x for x in ast.parse(source).body
                if isinstance(x, ast.FunctionDef) and x.name == 'pnjBKZ_simulator')
    ns = dict(copy=copy, math=math, log=math.log, sqrt=math.sqrt,
              lgamma=math.lgamma, pi=math.pi)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'reference-simulator','exec'),ns)
    reports, candidates = [], []
    for seed in range(args.last_seed+1):
        profile_path = args.run_root / ('seed-%d' % seed) / 'potlll.gso'
        profile = list(map(float,profile_path.read_text().split()))
        paths = [simulate(profile,ns['pnjBKZ_simulator'],j) for j in (9,3,1)]
        report = dict(seed=seed, dimension=167, target_norm=3405, max_bsd=143,
                      profile_sha256=hashlib.sha256(profile_path.read_bytes()).hexdigest(),
                      simulator_sha256=hashlib.sha256(source.encode()).hexdigest(),
                      caveat='Heuristic effective-beta mapping; not a success/runtime guarantee.',
                      paths=paths)
        reports.append(report)
        for path in paths:
            if path['eligible']:
                max_bsd = max(x['bsd'] for x in path['predictions'])
                candidates.append((max_bsd,path['pump_count'],path['predicted_norm'],seed,path))
        (profile_path.parent / 'simulation.json').write_text(json.dumps(report,indent=2)+'\n')
        print('seed %d: %s' % (seed, ', '.join(
            'J%d %.3f (%s)' % (x['jump'],x['predicted_norm'],
                              'candidate' if x['eligible'] else 'above target') for x in paths)))
    selected = None
    if candidates:
        # Pump count is only a coarse work proxy, not a measured runtime model.
        max_bsd,_,_,seed,path = min(candidates,key=lambda x:x[:4])
        schedule = args.run_root / ('seed-%d' % seed) / 'schedule.json'
        schedule.write_text(json.dumps(path['schedule'],indent=2)+'\n')
        selected = dict(seed=seed,jump=path['jump'],max_bsd=max_bsd,predicted_norm=path['predicted_norm'],
                        schedule=str(schedule),pump_count=path['pump_count'])
    summary = dict(target_norm=3405,selected=selected,reports=reports,
                   selection_rule='Smallest maximum BSD, then fewest pumps, then shortest predicted norm; heuristic.')
    (args.run_root / 'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Selected: '+json.dumps(selected),flush=True)


if __name__ == '__main__': main()
