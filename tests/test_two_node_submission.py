import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class RolloutSubmissionTests(unittest.TestCase):
    def invoke(self,fail=0):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            queue = temp/'squeue'
            queue.write_text('#!/bin/sh\nprintf "306526\\n306527\\n"\n')
            queue.chmod(0o755)
            batch = temp/'sbatch'
            batch.write_text('#!'+sys.executable+'\n'+'''
import json, os, sys
path = os.environ['MOCK_CALLS']
try: calls = json.load(open(path))
except FileNotFoundError: calls = []
calls.append(sys.argv[1:])
with open(path,'w') as out: json.dump(calls,out)
if len(calls) == int(os.environ['MOCK_FAIL']): sys.exit(1)
print(str(710000+len(calls)))
''')
            batch.chmod(0o755)
            env = dict(os.environ,PATH=directory+os.pathsep+os.environ['PATH'],
                SOURCE_COMMIT='test-source',MPI_HOST_RUNTIME='/test/runtime',
                MOCK_CALLS=str(temp/'calls.json'),MOCK_FAIL=str(fail))
            result = subprocess.run(['bash',str(ROOT/'reproduce/svp167-search/submit_two_node_rollout.sh')],
                env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=10)
            return result,json.loads((temp/'calls.json').read_text())
    def test_sequential_source_bound_chain(self):
        result,calls = self.invoke()
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(len(calls),8)
        self.assertIn('--dependency=afterany:306526:306527',calls[0])
        for index in range(1,8):
            self.assertIn('--dependency=afterok:%d' % (710000+index),calls[index])
            self.assertIn('--kill-on-invalid-dep=yes',calls[index])
        self.assertIn('--rsc=g=1',calls[1])
        self.assertIn('RUN_MODE=tour',' '.join(calls[-1]))
    def test_submission_failure_stops_chain(self):
        result,calls = self.invoke(3)
        self.assertNotEqual(result.returncode,0)
        self.assertEqual(len(calls),3)

if __name__ == '__main__': unittest.main()
