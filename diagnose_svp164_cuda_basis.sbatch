#!/bin/bash
#SBATCH --job-name=svp164-detcheck
#SBATCH --partition=gr20100g
#SBATCH --rsc=g=1
#SBATCH --time=00:10:00
#SBATCH --output=/LARGE0/gr20116/%u/BGJ-Sieve-GPU/logs/svp163-164/slurm-%x-%j.out

set -euo pipefail

BASE="/LARGE0/gr20116/${USER}/BGJ-Sieve-GPU/inputs/svp164-seed0"
export INPUT_A="${BASE}/L_164_0.potlll-heuristic-mpfr256"
export INPUT_B="${BASE}/L_164_0.potlll-heuristic-mpfr256.bkz.p72l8.pre"
export INPUT_C="${BASE}/L_164_0.potlll-heuristic-mpfr256.progressive-bkz74-cuda.pre"

# Exact modulo each prime: equal determinants for several independent primes
# provide a robust lattice-preservation check without numerically unstable GSO.
python3 - <<'PY'
import os

def load(path):
    with open(path) as f:
        rows = []
        for line in f:
            line = line.strip()
            if line.startswith('[['):
                line = line[2:]
            elif line.startswith('['):
                line = line[1:]
            if line.endswith(']]'):
                line = line[:-2]
            elif line.endswith(']'):
                line = line[:-1]
            if line:
                rows.append([int(x) for x in line.split()])
        return rows

def det_mod(a, p):
    a = [[x % p for x in row] for row in a]
    n = len(a)
    d = 1
    for col in range(n):
        pivot = next((r for r in range(col, n) if a[r][col]), None)
        if pivot is None:
            return 0
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
            d = -d
        v = a[col][col]
        d = d * v % p
        inv = pow(v, p - 2, p)
        for r in range(col + 1, n):
            if not a[r][col]:
                continue
            q = a[r][col] * inv % p
            a[r][col:] = [(x - q*y) % p for x, y in zip(a[r][col:], a[col][col:])]
    return d % p

bases = [(name, load(os.environ[name])) for name in ('INPUT_A', 'INPUT_B', 'INPUT_C')]
for name, a in bases:
    print(f'{name} rows={len(a)} cols={len(a[0])}')
for p in (1000000007, 1000000009, 998244353, 2147483647):
    print('prime', p, *(f'{name}={det_mod(a,p)}' for name, a in bases))

def primes(limit):
    out = []
    for n in range(2, limit + 1):
        if all(n % q for q in out if q*q <= n):
            out.append(n)
    return out

for p in primes(251):
    da = det_mod(bases[0][1], p)
    dc = det_mod(bases[2][1], p)
    if da and not dc:
        print(f'index_factor_candidate={p} input_det={da} cuda_det={dc}')
PY
