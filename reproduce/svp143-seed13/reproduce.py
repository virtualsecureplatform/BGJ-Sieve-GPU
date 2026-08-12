#!/usr/bin/env python3
"""Test official SVP-143 lattice seed 13 with integrated PotLLLBKZ."""

import os
from pathlib import Path
import runpy


HERE = Path(__file__).resolve().parent
os.environ.setdefault("SVP_DIMENSION", "143")
os.environ.setdefault("SVP142_LATTICE_SEED", "13")
os.environ.setdefault("SVP142_TARGET_NORM2", "8793148")
os.environ.setdefault("SVP142_RAW_SHA256", "GENERATE")
os.environ.setdefault("SVP142_KNOWN_VECTOR", str(HERE / "no-known-vector.txt"))
runpy.run_path(str(HERE.parent / "svp142-seed0" / "reproduce.py"), run_name="__main__")
