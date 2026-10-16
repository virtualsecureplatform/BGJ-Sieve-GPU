from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CudaSourceRegressionTests(unittest.TestCase):
    def test_lossless_compaction_bypasses_histogram_cutoff(self):
        source = (ROOT / "src" / "pool_hd.cpp").read_text()
        shrink = source.split("int Pool_hd_t::shrink(long N)", 1)[1].split("int Pool_hd_t::", 1)[0]
        self.assertIn("if (N != num_vec)", shrink)
        self.assertLess(shrink.index("if (score_th == 65536"),
                        shrink.index("score_stat[score_th] -= th_max_del"))

    def test_dual_hash_bucket_writes_are_saturated(self):
        source = (ROOT / "src" / "kernel_dhb.cu").read_text()
        self.assertNotIn("min(t_sbuc_num, out_max_size - pos)", source)
        self.assertEqual(source.count("pos < out_max_size ?"), 2)


if __name__ == "__main__":
    unittest.main()
