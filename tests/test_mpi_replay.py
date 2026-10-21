import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('replay', ROOT/'tools/test_mpi_pipeline_replay.py')
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


class ReplaySummaryTests(unittest.TestCase):
    def measurements(self, asynchronous=7):
        return [{'csd': dim, 'mode': mode, 'warmup': pair == 0,
                 'seconds': 1000 if pair == 0 else value}
                for dim in (128, 144) for pair in range(4)
                for mode, value in [('legacy', 10), ('async', asynchronous)]]

    def test_warmups_excluded_and_threshold_enforced(self):
        passed = REPLAY.summarize(self.measurements())
        self.assertTrue(all(row['passed'] for row in passed.values()))
        self.assertEqual(passed['128']['legacy_median_seconds'], 10)
        self.assertAlmostEqual(passed['144']['improvement'], .3)
        failed = REPLAY.summarize(self.measurements(9))
        self.assertFalse(failed['128']['passed'])

    def test_missing_measurements_rejected(self):
        with self.assertRaises(ValueError):
            REPLAY.summarize(self.measurements()[:-1])

    def test_exact_result_parser(self):
        match = REPLAY.RESULT.fullmatch('[MPI-REPLAY] csd=144 mode=async elapsed_seconds=1.2 '
                                       'bucket_records=33555520 candidate_records=524350')
        self.assertIsNotNone(match)
        self.assertEqual(match[2], 'async')

    def test_sequential_submission_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root/'sbatch'
            fake.write_text('#!%s\n' % sys.executable + '''
import json, os, sys
path = os.environ['MOCK_CALLS']
try:
    with open(path) as stream: calls = json.load(stream)
except FileNotFoundError:
    calls = []
calls.append(sys.argv[1:])
with open(path, 'w') as stream: json.dump(calls, stream)
print(str(710000+len(calls))+';cluster')
''')
            fake.chmod(0o755)
            env = dict(os.environ, PATH=directory+os.pathsep+os.environ['PATH'],
                       SOURCE_COMMIT='test-source', MPI_HOST_RUNTIME='/test/runtime',
                       MOCK_CALLS=str(root/'calls.json'))
            result = subprocess.run(['bash', str(ROOT/'reproduce/svp167-search/submit_mpi_pipeline_validation.sh')],
                                    env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    universal_newlines=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = json.loads((root/'calls.json').read_text())
            self.assertEqual(len(calls), 4)
            self.assertIn('--rsc=g=1', calls[0])
            for index in (1, 2, 3):
                self.assertIn('--dependency=afterok:%d' % (710000+index), calls[index])
                self.assertIn('--kill-on-invalid-dep=yes', calls[index])
            self.assertIn('--export=ALL,BENCH_BACKEND=mpi', calls[3])


if __name__ == '__main__':
    unittest.main()
