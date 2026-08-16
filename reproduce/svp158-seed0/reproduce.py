#!/usr/bin/env python3
"""Run the shared recipe for the official SVP-158 seed-0 instance."""

import os
from pathlib import Path
import runpy


HERE = Path(__file__).resolve().parent
os.environ.setdefault("SVP_DIMENSION", "158")
os.environ.setdefault("SVP142_LATTICE_SEED", "0")
os.environ.setdefault("SVP142_TARGET_NORM2", "9799278")
os.environ.setdefault(
    "SVP142_RAW_SHA256",
    "e9385c3dab0f8e15ad2e1cdbfab9ff9705ebb5ef140bbdec58fc55196585eeee",
)
os.environ.setdefault("SVP142_KNOWN_VECTOR", str(HERE / "known-vector.txt"))
runpy.run_path(str(HERE.parent / "svp142-seed0" / "reproduce.py"), run_name="__main__")
