from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

class MemorySafetyGuards(unittest.TestCase):
    def test_inactive_center_warps_do_not_load(self):
        source = (ROOT/'src/kernel_buc.cu').read_text()
        loader = source.split('void _center_load(')[1].split('void _buffer_init(')[0]
        self.assertIn('if (active)',loader)
        self.assertIn('wmma::fill_fragment',loader)
    def test_no_unsafe_clock_write_branches(self):
        for name in ('kernel_buc.cu','kernel_red.cu','kernel_mred.cu'):
            source = (ROOT/'src'/name).read_text()
            self.assertNotIn('RECORD_CLK',source)
            self.assertNotIn('end_clk',source)
    def test_queue_completion_is_synchronized(self):
        source = (ROOT/'src/bgj_hd_device.cu').read_text().split('int Reducer_t::_ld_lbuc(')[1]
        self.assertIn('tasks.pop(task)',source)
        self.assertIn('tasks.close()',source)
        self.assertIn('num_chunks > published_chunks',source)
        self.assertNotIn('num_task_vol',source)
    def test_import_does_not_weaken_resume(self):
        source = (ROOT/'app/mpi_bkz_driver.inc').read_text()
        normal = source.split('int mpi_bkz_run(')[1]
        self.assertIn('saved.identity == s.identity',normal)
        self.assertIn('unreviewed checkpoint source revision',source)
        self.assertIn('source checkpoint is in use',source)

if __name__ == '__main__': unittest.main()
