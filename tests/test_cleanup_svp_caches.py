import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location('cleanup_svp_caches',
    str(Path(__file__).resolve().parents[1]/'tools/cleanup_svp_caches.py'))
CLEANUP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLEANUP)


class CleanupTests(unittest.TestCase):
    def test_recognizes_misspelled_record(self):
        parser = CLEANUP.Records()
        parser.feed('<tr title="notes">'+''.join('<td>'+value+'</td>' for value in
            ['55', '152', '3093', '0', 'Kotark Matsuoka', 'vec', 'Sieving', 'date', '1'])+'</tr>')
        self.assertEqual(len(parser.rows), 1)
        self.assertEqual(parser.rows[0]['cells'][4], 'Kotark Matsuoka')

    def test_payload_names_exclude_logs_and_results(self):
        self.assertTrue(CLEANUP.PAYLOAD.fullmatch('.x@nV&zSP_00031e'))
        self.assertTrue(CLEANUP.PAYLOAD.fullmatch('.Ufile-0001'))
        for value in ['rank.log', 'result.basis', 'STATE', 'full.qp', 'submission-note.txt']:
            self.assertFalse(CLEANUP.PAYLOAD.fullmatch(value))

    def test_cleanup_preserves_logs_results_links_and_active_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            cache = base/'runs/svp160-seed2/job-old/.pool/0'
            cache.mkdir(parents=True)
            payload = cache/'.abcdefgh_000001'
            payload.write_bytes(b'cache')
            log = cache/'rank.log'; log.write_text('keep')
            basis = cache/'result.basis'; basis.write_text('keep')
            link = cache/'.abcdefgh_000002'; link.symlink_to(basis)
            active = base/'runs/svp167-search/seed38/.pool'
            active.mkdir(parents=True)
            mpi = base/'mpi-runs/active/rank0/.pool'; mpi.mkdir(parents=True)
            manifest = base/'manifest.json'
            data = {'version': 1, 'base': str(base), 'payload_regex': CLEANUP.PAYLOAD.pattern,
                    'cutoff_ns': int((time.time()+2)*10**9), 'cache_directories': [str(cache.parent)]}
            manifest.write_text(json.dumps(data))
            with mock.patch.object(CLEANUP, 'BASE', base), mock.patch.object(CLEANUP, 'jobs', return_value=[]):
                self.assertEqual(list(CLEANUP.discover()), [str(cache.parent)])
                CLEANUP.apply(manifest)
                self.assertFalse(CLEANUP.eligible(active))
                self.assertFalse(CLEANUP.eligible(mpi))
            self.assertFalse(payload.exists())
            self.assertTrue(log.exists()); self.assertTrue(basis.exists()); self.assertTrue(link.is_symlink())

    def test_refuses_symlink_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            actual = base/'outside'; actual.mkdir()
            run = base/'runs/old'; run.mkdir(parents=True)
            link = run/'.pool'; link.symlink_to(actual, target_is_directory=True)
            with mock.patch.object(CLEANUP, 'BASE', base):
                self.assertFalse(CLEANUP.eligible(link))


if __name__ == '__main__':
    unittest.main()
