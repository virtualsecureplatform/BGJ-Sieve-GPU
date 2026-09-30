#!/usr/bin/env python3
"""Select a capped progressive schedule from a PotLLL GSO profile.

Uses the reference simulator without importing its plotting dependencies.
Predictions are heuristic, not a certificate of challenge success.
"""
import ast
import contextlib
import copy
import hashlib
import io
import json
import math
import pathlib
import sys


def plan(profile, simulator_source):
    if len(profile) != 164 or not all(math.isfinite(x) for x in profile):
        raise ValueError("expected 164 finite natural-log GS norms")
    node = next(n for n in ast.parse(simulator_source).body
                if isinstance(n, ast.FunctionDef) and n.name == "pnjBKZ_simulator")
    ns = dict(copy=copy, math=math, log=math.log, sqrt=math.sqrt,
              lgamma=math.lgamma, pi=math.pi)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "reference-simulator", "exec"), ns)
    target = 1.05 * math.exp(sum(profile) / 164 + math.lgamma(83) / 164
                            - math.log(math.pi) / 2)
    stages, predictions = [], []
    current = list(profile)
    # Sparse progression avoids paying for every adjacent high-dimensional tour.
    for bsd in [60, 80, 100, 112, 120, 128, 138, 143, 143]:
        slope = sum((i - 81.5) * x for i, x in enumerate(current)) / sum(
            (i - 81.5) ** 2 for i in range(164))
        bonus = max(0, int(math.log(math.sqrt(4 / 3)) / max(-slope / 2, 1e-9)))
        beta = min(163, bsd + min(bonus, 164 - bsd))
        with contextlib.redirect_stdout(io.StringIO()):
            current = ns["pnjBKZ_simulator"](current, beta, 1, 164, 9)
        predicted = math.exp(current[0])
        predictions.append(dict(bsd=bsd, effective_beta=beta, predicted_norm=predicted))
        last_start = 9 * math.ceil((164 - bsd - 2) / 9)
        starts = list(range(0, last_start + 1, 9))
        for pos, start in enumerate(starts):
            stage = dict(sieve_dimension=bsd, d4f=min(26, 164-bsd),
                         start_index=start)
            if pos + 1 < len(starts):
                stage["stop_index"] = starts[pos + 1]
            stages.append(stage)
        if predicted < target:
            break
    return dict(name="svp164-seed1-potlll-simulator-progressive",
                target_norm2=math.ceil(target * target) - 1,
                defaults=dict(jump=9, dual_hash_ratio=0, tours=1), stages=stages), dict(
                    target_norm=target, predictions=predictions,
                    simulator_sha256=hashlib.sha256(simulator_source.encode()).hexdigest(),
                    caveat="Heuristic effective-beta mapping; no success guarantee; BSD capped at 143.")


if __name__ == "__main__":
    profile_path, source_path, output_path = map(pathlib.Path, sys.argv[1:])
    schedule, report = plan(list(map(float, profile_path.read_text().split())),
                            source_path.read_text())
    report["profile_sha256"] = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    output_path.write_text(json.dumps(schedule, indent=2) + "\n")
    output_path.with_suffix(".simulation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
