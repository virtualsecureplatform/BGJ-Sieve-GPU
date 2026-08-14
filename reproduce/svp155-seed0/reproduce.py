#!/usr/bin/env python3
"""Run the shared recipe for official SVP-155 lattice seed 0."""

import os
from pathlib import Path
import runpy


HERE = Path(__file__).resolve().parent
os.environ.setdefault("SVP_DIMENSION", "155")
os.environ.setdefault("SVP142_LATTICE_SEED", "0")
os.environ.setdefault("SVP142_TARGET_NORM2", "10018742")
os.environ.setdefault("SVP142_RAW_SHA256", "GENERATE")
os.environ.setdefault("SVP142_KNOWN_VECTOR", str(HERE / "known-vector.txt"))
runpy.run_path(str(HERE.parent / "svp142-seed0" / "reproduce.py"), run_name="__main__")
