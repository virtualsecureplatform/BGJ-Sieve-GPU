# SVP-164 seed-0 scheduling simulation

The input is the determinant-verified HD-BKZ-128 basis. Its 164 natural-log
Gram--Schmidt norms were extracted with `tools/dump_gso.cpp`. The standalone
pnj-BKZ simulator in `pro-pnj-bkz/PnJBKZ_Simulator` was applied using the
ASIA CCS 2023 strategy-5 adjustment for jump values at most nine:

```text
effective simulator beta = BSD + floor(log(sqrt(4/3)) / (-GSO slope / 2))
```

The mapping was checked against the observed HD-BKZ-120 to HD-BKZ-128
transition. Effective beta 148 gave the lowest profile RMSE among the nearby
candidates. It predicted first-vector norm 3634; the observed value was 3545,
so the simulator was conservative by about 89 in this transition.

One-tour candidates from the actual BKZ-128 profile were:

| BSD | Effective beta | Simulated first norm | Calibrated norm | Relative list-size proxy |
|---:|---:|---:|---:|---:|
| 132 | 148 | 3545 | 3456 | 1.00 |
| 136 | 152 | 3521 | 3432 | 1.78 |
| 138 | 154 | 3470 | 3381 | 2.37 |
| 140 | 156 | 3417 | 3328 | 3.16 |
| 142 | 158 | 3366 | 3277 | 4.21 |

The additive calibration is only a diagnostic: the observed transition also
shows that first-vector prediction has substantial error. In particular, BSD
140 is around the strict target (3318.84), while BSD 142 has clearer direct-BKZ
margin but approaches the final-sieve resource cost.

The selected compromise is BSD 138, D4F 26, jump 9. This is a full-width
164-dimensional block and is the lowest candidate that materially changes the
simulated leading profile without paying the BSD-140/142 cost. The subsequent
CSD-138 and CSD-142 jobs remain necessary; this simulation does not claim that
BKZ-138 alone will solve the instance.

No enumeration, GPU kernel, or sieve was run for this analysis.

Operational recovery note (2026-09-23): the CSD-138 sieve pool from the first
BKZ pump completed, but the first dual-hash pump-down insertion failed with a
CUDA illegal-memory-access error. The recovery schedule sets `BDH=0` so that
pump-down uses the ordinary insertion path. This changes the actual BKZ tour
and has not been validated by the simulation above. `RESUME_CSD=138` reuses the
saved first-pump pool only after checking its context and basis hash; later
pumps start normally. The scheduler still checkpoints only completed tours.
