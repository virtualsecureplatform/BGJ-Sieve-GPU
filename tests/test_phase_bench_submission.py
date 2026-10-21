import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PhaseBenchSubmissionTests(unittest.TestCase):
    def submit_with_mock(self, fail_mpi=False, validation_job=''):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / 'sbatch'
            calls = root / 'calls.json'
            fake.write_text('#!%s\n' % sys.executable + '''
import json, os, sys
path = os.environ['MOCK_CALLS']
try:
    with open(path) as stream: calls = json.load(stream)
except FileNotFoundError:
    calls = []
calls.append(sys.argv[1:])
with open(path, 'w') as stream: json.dump(calls, stream)
if len(calls) == 2 and os.environ.get('FAIL_MPI') == '1':
    sys.exit(1)
print(str(700000 + len(calls)) + ';test-cluster')
''')
            fake.chmod(0o755)
            env = dict(os.environ, PATH=directory + os.pathsep + os.environ['PATH'],
                       SOURCE_COMMIT='validated-source', VALIDATED_BUILD='/validated/build',
                       MOCK_CALLS=str(calls), FAIL_MPI='1' if fail_mpi else '0',
                       BENCH_VALIDATION_JOB=validation_job)
            result = subprocess.run(['bash', str(ROOT / 'reproduce' / 'svp167-search' /
                                                'submit_phase_bench.sh')],
                                    env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    universal_newlines=True, timeout=10)
            return result, json.loads(calls.read_text())

    def test_separate_resources_and_completion_dependency(self):
        result, calls = self.submit_with_mock()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 2)
        self.assertIn('--rsc=g=4', calls[0])
        self.assertIn('--export=ALL,BENCH_BACKEND=serial', calls[0])
        self.assertIn('--rsc=g=8', calls[1])
        self.assertIn('--export=ALL,BENCH_BACKEND=mpi', calls[1])
        self.assertIn('--dependency=afterany:700001', calls[1])
        self.assertIn('MPI benchmark: 700002', result.stdout)

    def test_failed_mpi_submission_reports_existing_serial_job(self):
        result, calls = self.submit_with_mock(fail_mpi=True)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(len(calls), 2)
        self.assertIn('serial job 700001 remains queued', result.stderr)

    def test_validation_gates_both_but_serial_success_does_not_gate_mpi(self):
        result, calls = self.submit_with_mock(validation_job='600001')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--dependency=afterok:600001', calls[0])
        self.assertIn('--dependency=afterok:600001,afterany:700001', calls[1])


if __name__ == '__main__':
    unittest.main()
