#!/usr/bin/env python3
"""Run the shared recipe for the official SVP-157 seed-0 instance."""

import os
from pathlib import Path
import runpy


HERE = Path(__file__).resolve().parent
os.environ.setdefault("SVP_DIMENSION", "157")
os.environ.setdefault("SVP142_LATTICE_SEED", "0")
os.environ.setdefault("SVP142_TARGET_NORM2", "10701947")
os.environ.setdefault(
    "SVP142_RAW_SHA256",
    "a670a17904355c08f5396dc6c006f74365d2b64f7a6a4a9a2ea98d11c4fe79c9",
)
os.environ.setdefault("SVP142_KNOWN_VECTOR", str(HERE / "known-vector.txt"))
runpy.run_path(str(HERE.parent / "svp142-seed0" / "reproduce.py"), run_name="__main__")
