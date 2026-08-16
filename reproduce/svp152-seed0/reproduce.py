#!/usr/bin/env python3
"""Run the shared recipe for the official SVP-152 seed-0 instance."""

import os
from pathlib import Path
import runpy


HERE = Path(__file__).resolve().parent
os.environ.setdefault("SVP_DIMENSION", "152")
os.environ.setdefault("SVP142_LATTICE_SEED", "0")
os.environ.setdefault("SVP142_TARGET_NORM2", "9920828")
os.environ.setdefault(
    "SVP142_RAW_SHA256",
    "eeb892b80e4da321dc3c1787569803485580ac73f879045f971e53248f3f9daf",
)
os.environ.setdefault("SVP142_KNOWN_VECTOR", str(HERE / "known-vector.txt"))
runpy.run_path(str(HERE.parent / "svp142-seed0" / "reproduce.py"), run_name="__main__")
