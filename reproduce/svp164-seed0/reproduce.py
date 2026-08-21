#!/usr/bin/env python3
"""Solve the official SVP-164 seed-0 challenge at the 1.05-GH threshold."""

import os
from pathlib import Path
import runpy


HERE = Path(__file__).resolve().parent
seed = os.environ.get("SVP142_LATTICE_SEED", "0")
if int(seed) != 0:
    raise SystemExit("SVP-164 seed-0 recipe requires SVP142_LATTICE_SEED=0")

os.environ.setdefault("SVP_DIMENSION", "164")
# For the official seed-0 lattice, 1.05 * GH is 3318.842..., hence a vector
# with integral squared norm <= 3318^2 meets the strict challenge threshold.
os.environ.setdefault("SVP142_TARGET_NORM2", "11009124")
os.environ.setdefault("SVP142_RAW_SHA256", "GENERATE")
os.environ.setdefault("SVP142_KNOWN_VECTOR", str(HERE.parent / "svp158-search" / "no-known-vector.txt"))
runpy.run_path(str(HERE.parent / "svp142-seed0" / "reproduce.py"), run_name="__main__")
