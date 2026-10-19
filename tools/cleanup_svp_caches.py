#!/usr/bin/env python3
"""Conservative, audited cleanup of obsolete SVP binary cache payloads only.

Logs, bases, vectors, inputs, code, builds and MPI snapshots are never targets.
No cache contents are read; each unlink is restricted by path, name, age and UID.
"""
import argparse
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import time
from urllib.request import urlopen

BASE = Path('/LARGE0/gr20116/kotaro982matsuoka/BGJ-Sieve-GPU')
TOPS = ('runs', 'diagnostics', 'cache-diag', 'c586-a100x4-repro',
        'c586-repro', 'c586-variant', 'mpi-probe')
CACHES = {'.pool', '.bucket', '.sol', '.uid', '.mpi-active-pool'}
PAYLOAD = re.compile(r'(?:\.[A-Za-z0-9&@]{8}_[0-9a-f]{6}|\.Ufile-[0-9]{4,})\Z')
HALL = 'https://latticechallenge.org/svp-challenge/halloffame.php'


class Records(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self, convert_charrefs=True)
        self.rows = []; self.row = None; self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == 'tr': self.row = {'notes': dict(attrs).get('title', ''), 'cells': []}
        if tag == 'td' and self.row is not None: self.cell = []

    def handle_data(self, data):
        if self.cell is not None: self.cell.append(data)

    def handle_endtag(self, tag):
        if tag == 'td' and self.cell is not None:
            self.row['cells'].append(''.join(self.cell).strip()); self.cell = None
        if tag == 'tr' and self.row is not None:
            cells = self.row['cells']
            if len(cells) >= 9 and cells[4] in ('Kotaro Matsuoka', 'Kotark Matsuoka'):
                self.rows.append(self.row)
            self.row = None


def jobs():
    output = subprocess.check_output(['squeue', '-h', '-u', str(os.getuid()), '-o', '%i|%T|%o'],
                                     universal_newlines=True)
    lines = [line for line in output.splitlines() if line.strip()]
    # Known active workflow is protected in its entirety, including seed search
    # inputs. Refuse to clean alongside another, unidentified user workflow.
    if any('mpi_bkz144_pilot.sbatch' not in line for line in lines):
        raise RuntimeError('unrecognized queued/running user job; review exclusions first: '+repr(lines))
    return lines


def eligible(path):
    relative = path.relative_to(BASE)
    return (relative.parts[0] in TOPS and
            not any('svp167' in part or 'seed38' in part for part in relative.parts) and
            path.name in CACHES and not path.is_symlink() and path.resolve() == path)


def discover():
    for top in TOPS:
        if not (BASE/top).exists(): continue
        for root, dirs, files in os.walk(str(BASE/top), followlinks=False):
            path = Path(root)
            dirs[:] = sorted(d for d in dirs if not (path/d).is_symlink() and
                             'svp167' not in d and 'seed38' not in d and d != '.git')
            if path.name in CACHES:
                dirs[:] = []
                if eligible(path): yield str(path)


def save_json(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def plan(manifest):
    if manifest.exists(): raise RuntimeError('manifest already exists')
    current_jobs = jobs()
    parser = Records()
    parser.feed(urlopen(HALL, timeout=45).read().decode('utf-8', errors='replace'))
    if not parser.rows or not any(r['cells'][4] == 'Kotark Matsuoka' for r in parser.rows):
        raise RuntimeError('official record lookup incomplete; no cleanup plan written')
    manifest.parent.mkdir(parents=True, exist_ok=True)
    for row in parser.rows:
        row['solution_url'] = HALL.rsplit('/', 1)[0]+'/solution.php?id='+row['cells'][0]
        row['solution_html'] = urlopen(row['solution_url'], timeout=45).read().decode('utf-8', errors='replace')
    caches = sorted(discover())
    data = {'version': 1, 'base': str(BASE), 'cutoff_ns': int(time.time()*10**9),
            'created_utc': datetime.now(timezone.utc).isoformat(), 'jobs': current_jobs,
            'official_url': HALL, 'official_records': parser.rows,
            'preserve': ['ALL logs, bases, vectors, inputs, sources, builds and non-cache files',
                         'ALL mpi-runs including previous CSD124 snapshots',
                         'ALL svp167/seed38 directories and active pilot data'],
            'cache_directories': caches, 'payload_regex': PAYLOAD.pattern}
    save_json(manifest, data)
    print('Official records: '+repr([(r['cells'][1], r['cells'][3], r['cells'][4]) for r in parser.rows]), flush=True)
    print('Planned cache directories: %d; manifest=%s' % (len(caches), manifest), flush=True)


def apply(manifest):
    with manifest.open() as stream: data = json.load(stream)
    if data['version'] != 1 or data['base'] != str(BASE) or data['payload_regex'] != PAYLOAD.pattern:
        raise RuntimeError('incompatible cleanup manifest')
    jobs()
    report = manifest.with_name('removed-cache-summary.jsonl')
    removed = allocated = logical = kept = 0
    last_progress = time.monotonic()
    # Exclusive audit creation prevents accidentally replaying a partial pass.
    with report.open('x') as audit:
        for index, target in enumerate(data['cache_directories']):
            path = Path(target)
            if not eligible(path): raise RuntimeError('unsafe target: '+target)
            counts = {'path': target, 'files_removed': 0, 'allocated_bytes_removed': 0,
                      'logical_bytes_removed': 0, 'unrecognized_files_kept': 0}
            for root, dirs, files in os.walk(target, followlinks=False):
                dirs[:] = [d for d in dirs if not (Path(root)/d).is_symlink()]
                if Path(root).resolve() != Path(root): raise RuntimeError('symlink ancestor')
                # dir_fd keeps every unlink confined to the checked directory.
                fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                try:
                    for name in files:
                        if not PAYLOAD.fullmatch(name):
                            counts['unrecognized_files_kept'] += 1; continue
                        try: info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                        except FileNotFoundError: continue
                        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mtime_ns >= data['cutoff_ns']:
                            counts['unrecognized_files_kept'] += 1; continue
                        os.unlink(name, dir_fd=fd)
                        counts['files_removed'] += 1
                        if info.st_nlink == 1:
                            counts['allocated_bytes_removed'] += info.st_blocks*512
                        counts['logical_bytes_removed'] += info.st_size
                        if counts['files_removed'] % 1000 == 0:
                            time.sleep(.05)  # Rate-limit metadata operations.
                            if time.monotonic()-last_progress >= 30:
                                print('IN-PROGRESS files=%d freed=%.3f TiB current=%s' %
                                      (removed+counts['files_removed'],
                                       (allocated+counts['allocated_bytes_removed'])/2**40, target), flush=True)
                                last_progress = time.monotonic()
                finally: os.close(fd)
            audit.write(json.dumps(counts)+'\n'); audit.flush(); os.fsync(audit.fileno())
            removed += counts['files_removed']; allocated += counts['allocated_bytes_removed']
            logical += counts['logical_bytes_removed']; kept += counts['unrecognized_files_kept']
            print('%d/%d files=%d freed=%.3f TiB current=%s' %
                  (index+1, len(data['cache_directories']), removed, allocated/2**40, target), flush=True)
    summary = {'files_removed': removed, 'allocated_bytes_removed': allocated,
               'logical_bytes_removed': logical, 'unrecognized_files_kept': kept,
               'finished_utc': datetime.now(timezone.utc).isoformat()}
    save_json(manifest.with_name('cleanup-result.json'), summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'apply'))
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    {'plan': plan, 'apply': apply}[args.mode](args.manifest.resolve())
