# SVP-142, lattice seed 0

This recipe targets the official seed-0 vector with exact squared norm
`9075417` (displayed norm `3013`). The preprocessing path is deliberately
LLL followed by fplll BKZ-60 with its pruned default strategy and at most eight
loops. The final sieve uses seed 0, MLD 118, and TSD 132 by default.

Submit from the repository root with:

```sh
./submit_svp142_seed0_a100x4.sh
```

The wrapper first submits a one-GPU preparation/build job, then a dependent
four-GPU solve job. All generated bases, binaries, and sieve files are stored
under `/LARGE0/gr20116/$USER/BGJ-Sieve-GPU`; no generated data is put in home.
