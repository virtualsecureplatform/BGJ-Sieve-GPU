#!/usr/bin/env python3
"""Validate a copied HD pool without loading vectors into memory."""

import argparse
import os
from pathlib import Path
import re
import struct


CHUNK_NAME = re.compile(r"^\..+_([0-9a-f]{6})$")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--left", type=int, required=True)
    parser.add_argument("--right", type=int, required=True)
    parser.add_argument("--hash", type=lambda value: int(value, 0), required=True)
    parser.add_argument("--chunks", type=int)
    parser.add_argument("--vectors", type=int)
    args = parser.parse_args()

    ids = set()
    vectors = 0
    for device in range(4):
        directory = args.root / ".pool" / str(device)
        if not directory.is_dir():
            raise ValueError(f"missing pool directory: {directory}")
        with os.scandir(directory) as entries:
            for entry in entries:
                match = CHUNK_NAME.fullmatch(entry.name)
                if not match or not entry.is_file(follow_symlinks=False):
                    raise ValueError(f"unexpected pool entry: {entry.path}")
                chunk_id = int(match.group(1), 16)
                if chunk_id in ids:
                    raise ValueError(f"duplicate chunk ID: {chunk_id}")
                ids.add(chunk_id)
                with open(entry.path, "rb") as handle:
                    header = handle.read(12)
                if len(header) != 12:
                    raise ValueError(f"short header: {entry.path}")
                size, basis_hash, left, right = struct.unpack("<HQBB", header)
                expected_bytes = 4096 + 8192 * (14 + args.right - args.left)
                if entry.stat(follow_symlinks=False).st_size != expected_bytes:
                    raise ValueError(f"wrong file size: {entry.path}")
                if not (0 < size <= 8192):
                    raise ValueError(f"invalid vector count {size}: {entry.path}")
                if (basis_hash, left, right) != (args.hash, args.left, args.right):
                    raise ValueError(
                        f"wrong metadata {(basis_hash, left, right)}: {entry.path}"
                    )
                vectors += size

    if ((args.chunks is not None and len(ids) != args.chunks) or
            (args.vectors is not None and vectors != args.vectors)):
        raise ValueError(
            f"got {len(ids)} chunks and {vectors} vectors; "
            f"expected {args.chunks} and {args.vectors}"
        )
    if not (args.root / ".bkz-scheduler-input.basis_0").is_file():
        raise ValueError("matching input basis is missing")
    print(f"validated CSD {args.right - args.left}: {len(ids)} chunks, {vectors} vectors")


if __name__ == "__main__":
    main()
