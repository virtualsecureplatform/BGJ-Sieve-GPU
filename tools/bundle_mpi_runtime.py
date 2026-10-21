#!/usr/bin/env python3
"""Extract this image's ELF dependencies for a host-namespace MPI launcher.

Run inside the trusted image on a compute node. Host MPI / GPU driver libraries
remain host-provided; only image userspace libraries are copied to a fresh root.
"""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


def main():
    binary, destination = sys.argv[1:]
    root = Path(destination)
    root.mkdir()  # Refuse to overwrite an existing runtime bundle.
    output = subprocess.check_output(['ldd', binary], universal_newlines=True)
    if 'not found' in output:
        raise RuntimeError(output)
    paths = set(re.findall(r'(?:=>\s+|^\s*)(/\S+)\s+\(', output, re.MULTILINE))
    manifest = []
    for path in sorted(paths):
        if path.startswith(('/opt/system/', '/.singularity.d/')):
            continue
        source = Path(path)
        target = root / source.name
        shutil.copy2(str(source), str(target), follow_symlinks=True)
        manifest.append({'source': path, 'name': target.name,
                         'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
    for name in ('ld-linux-x86-64.so.2', 'libc.so.6', 'libcudart.so.13'):
        if not (root / name).is_file():
            raise RuntimeError('missing required runtime library: ' + name)
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Runtime bundle:', root, 'libraries:', len(manifest), flush=True)


if __name__ == '__main__':
    main()
