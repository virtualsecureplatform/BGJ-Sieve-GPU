# Reproduce SVP-138, seed 2

This recipe recreates the August 6, 2026 run that found a vector of squared
norm `8496181` (Euclidean norm `2914.8209207428163`) in the Darmstadt SVP
challenge lattice of dimension 138 with lattice seed 2.

The complete pipeline is:

1. Download dimension 138/seed 2 from the official online generator.
2. Check the raw lattice byte-for-byte.
3. Run `fplll 5.5.0` LLL.
4. Run pruned BKZ-60 with the pinned fplll strategy, at most eight tours.
5. Build the solver with the 47/32/8 GiB SVP-140 cache profile.
6. Sieve with `HD_SIEVE_SEED=0`, `TSD=128`, and `MLD=114`.
7. Verify exact membership in the downloaded lattice and squared norm at most
   `8496181` before declaring success.

Print the pinned commands and hashes without doing any work:

```bash
python3 reproduce/svp138-seed2/reproduce.py plan
```

Run each expensive stage separately:

```bash
python3 reproduce/svp138-seed2/reproduce.py prepare
python3 reproduce/svp138-seed2/reproduce.py build
python3 reproduce/svp138-seed2/reproduce.py run
```

Or run the whole workflow. `all` deliberately starts by acquiring the lattice:

```bash
python3 reproduce/svp138-seed2/reproduce.py all
```

An existing cache-profile binary may be supplied to avoid rebuilding:

```bash
python3 reproduce/svp138-seed2/reproduce.py all --binary app/hd_sieve_140P
```

## Requirements

- `fplll 5.5.0` and
  `/usr/share/libfplll9/strategies/default.json`. The script checks the version
  and the strategy SHA-256 before preprocessing.
- `make`, `g++`, `nvcc`, NTL, GMP, libnuma, and the CUDA driver library to build.
  The recipe looks for `nvcc` on `PATH` and under `/usr/local/cuda*`.
- Two NVIDIA A100 40 GB GPUs. The checked-in hardware configuration addresses
  GPU IDs 0 and 1.
- About 125 GiB system RAM and ample fast local storage. The cache-profile
  build reserves 47/32/8 GiB for the pool/bucket/solution caches. Run data is
  preserved under `work/runs/`; remove it manually after retaining the log and
  `result.json`.

The successful run used source commit `c586b64` and took about 3852 seconds to
reach the vector on two A100-PCIE-40GB GPUs. The current compatibility wrapper
sets `HD_CUDA_BLOCKING_SYNC=0` and `HD_BGJ2_REDUCER_THREADS=32` to retain that
commit's runtime behavior after the later CPU-bottleneck defaults changed.
GPU scheduling, drivers, compiler versions, and hardware can still affect
timing; the recipe pins inputs, algorithm parameters, and RNG selection, but
does not promise identical wall-clock time.

A fresh run on August 9, 2026 replayed the same 138-coordinate vector after
2554.46 seconds of sieving. Its trajectory differed from the
original run, confirming that the GPU/thread schedule is not bit-for-bit
deterministic even with `HD_SIEVE_SEED=0`. The solver displayed this candidate's
approximate length as `2914.85`; the recipe checks every reported leading
vector using integer arithmetic rather than trusting that floating estimate.

The known result can be checked independently (this downloads only the raw
lattice and does not run BKZ or the sieve):

```bash
python3 reproduce/svp138-seed2/reproduce.py verify-known
```

The official challenge site documents that arbitrary seeds can be obtained
through its online generator. Local generation requires NTL older than 9.4
because newer NTL releases changed their pseudorandom generator:
https://www.latticechallenge.org/svp-challenge/
