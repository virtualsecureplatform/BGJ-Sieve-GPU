import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('checkpoint_bench',
    Path(__file__).resolve().parents[1]/'tools/summarize_checkpoint_bench.py')
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)

class CheckpointBenchmarkTests(unittest.TestCase):
    def records(self,direct=8):
        return '\n'.join('[MPI-CHECKPOINT-BENCH] pair=%d writer=%s seconds=%s pool_write_bytes_per_rank=%d' %
            (pair,writer,time,size) for pair in range(4)
            for writer,time,size in [('legacy',10,200),('direct',direct,100)])
    def test_threshold_and_warmup(self):
        self.assertTrue(BENCH.summarize(self.records())['passed'])
        self.assertFalse(BENCH.summarize(self.records(9))['passed'])
    def test_missing_or_duplicate_pairs(self):
        with self.assertRaises(ValueError): BENCH.summarize(self.records().split('\n',1)[1])
        with self.assertRaises(ValueError): BENCH.summarize(self.records()+'\n'+self.records())

if __name__ == '__main__': unittest.main()
