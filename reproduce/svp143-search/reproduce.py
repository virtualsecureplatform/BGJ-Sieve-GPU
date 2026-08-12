#!/usr/bin/env python3
"""Explore official SVP-143 lattice seeds 15 through 23."""

import os
from pathlib import Path
import runpy


HERE = Path(__file__).resolve().parent
seed = os.environ.get("SVP142_LATTICE_SEED")
if seed is None or not (15 <= int(seed) <= 23):
    raise SystemExit("SVP142_LATTICE_SEED must be in [15, 23]")

os.environ.setdefault("SVP_DIMENSION", "143")
os.environ.setdefault("SVP142_TARGET_NORM2", "8793148")
os.environ.setdefault("SVP142_RAW_SHA256", "GENERATE")
os.environ.setdefault("SVP142_KNOWN_VECTOR", str(HERE / "no-known-vector.txt"))
runpy.run_path(str(HERE.parent / "svp142-seed0" / "reproduce.py"), run_name="__main__")
