#!/usr/bin/env python3
"""Run the shared SVP-142 recipe for official lattice seed 42."""

import os
from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent
os.environ.setdefault("SVP142_LATTICE_SEED", "42")
os.environ.setdefault("SVP142_TARGET_NORM2", "8949279")
os.environ.setdefault(
    "SVP142_RAW_SHA256",
    "e8ca102d4fe3282731d009a3ac8b36219fad5cf0b8bc64e1affa256fac0f2962",
)
os.environ.setdefault("SVP142_KNOWN_VECTOR", str(HERE / "known-vector.txt"))
runpy.run_path(str(HERE.parent / "svp142-seed0" / "reproduce.py"), run_name="__main__")
