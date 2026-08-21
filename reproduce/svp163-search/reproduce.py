#!/usr/bin/env python3
"""Search SVP-163 lattice seeds 0 through 6 for a strict record improvement."""

import os
from pathlib import Path
import runpy


HERE = Path(__file__).resolve().parent
seed = os.environ.get("SVP142_LATTICE_SEED")
if seed is None or not (0 <= int(seed) <= 6):
    raise SystemExit("SVP142_LATTICE_SEED must be in [0, 6]")

os.environ.setdefault("SVP_DIMENSION", "163")
# The official SVP Challenge record is norm 3333.  Seed hunting must only
# stop on a strictly shorter vector.
os.environ.setdefault("SVP142_TARGET_NORM2", "11108888")
os.environ.setdefault("SVP142_RAW_SHA256", "GENERATE")
os.environ.setdefault("SVP142_KNOWN_VECTOR", str(HERE.parent / "svp158-search" / "no-known-vector.txt"))
runpy.run_path(str(HERE.parent / "svp142-seed0" / "reproduce.py"), run_name="__main__")
