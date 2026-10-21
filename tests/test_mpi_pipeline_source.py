"""Cheap source guardrails; real transport/cache tests run under Slurm."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MpiPipelineSourceTests(unittest.TestCase):
    def test_nonblocking_owner_and_ack_lifetime(self):
        source = (ROOT / 'app/mpi_sieve_pipeline.inc').read_text()
        self.assertNotIn('MPI_Sendrecv', source)
        self.assertNotIn('MPI_Allreduce', source)
        self.assertIn('MPI_Isend', source)
        self.assertIn('MPI_Irecv', source)
        self.assertIn('if (!data_done || !ack_done) return;', source)
        self.assertIn('g_pipeline_pending.fetch_sub(slot.charged)', source)
        self.assertIn('(!swc || bwc->num_active() == 0)', source)

    def test_bounded_buffers_and_no_final_cleanup_in_pressure_pass(self):
        source = (ROOT / 'src/bgj_hd_device.cu').read_text()
        self.assertIn('_batch(tid, threshold, 0, false, 8)', source)
        self.assertIn('if (!batch0 && final_cleanup)', source)
        self.assertIn('if (!mpi_sieve_async() && mpi_sieve_flush_candidates', source)
        for frame_mib in (1, 8, 256, 1024):
            frame = frame_mib << 20
            pinned = 4 * (min(64 << 20, frame//4) + min(16 << 20, frame//16))
            self.assertLessEqual(pinned + 2*min(1 << 20, frame//4), 2*frame)

    def test_legacy_default_and_protocol_agreement(self):
        source = (ROOT / 'app/mpi_sieve_runtime.cpp').read_text()
        self.assertIn('pipeline && !strcmp(pipeline, "async")', source)
        self.assertIn('pipeline version/frame settings differ between ranks', source)
        self.assertIn('pipeline_release_buffers()', source)


if __name__ == '__main__':
    unittest.main()
