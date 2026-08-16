#!/usr/bin/env python3
"""Explore official SVP-158 lattice seeds 1 through 8."""

import os
from pathlib import Path
import runpy


HERE = Path(__file__).resolve().parent
seed = os.environ.get("SVP142_LATTICE_SEED")
if seed is None or not (1 <= int(seed) <= 8):
    raise SystemExit("SVP142_LATTICE_SEED must be in [1, 8]")

os.environ.setdefault("SVP_DIMENSION", "158")
# The published seed-0 record has norm^2 9,799,278. A seed hunt must require
# a strict improvement rather than stopping on a tie.
os.environ.setdefault("SVP142_TARGET_NORM2", "9799277")
os.environ.setdefault("SVP142_RAW_SHA256", "GENERATE")
os.environ.setdefault("SVP142_KNOWN_VECTOR", str(HERE / "no-known-vector.txt"))
runpy.run_path(str(HERE.parent / "svp142-seed0" / "reproduce.py"), run_name="__main__")
