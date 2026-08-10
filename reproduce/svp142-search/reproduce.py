#!/usr/bin/env python3
"""Explore an official SVP-142 lattice seed with a strict record target."""

import os
from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent
seed = os.environ.get("SVP142_LATTICE_SEED")
if seed is None or not (43 <= int(seed) <= 52):
    raise SystemExit("SVP142_LATTICE_SEED must be in [43, 52]")
os.environ.setdefault("SVP142_TARGET_NORM2", "8949278")
os.environ.setdefault("SVP142_RAW_SHA256", "GENERATE")
os.environ.setdefault("SVP142_KNOWN_VECTOR", str(HERE / "no-known-vector.txt"))
runpy.run_path(str(HERE.parent / "svp142-seed0" / "reproduce.py"), run_name="__main__")
