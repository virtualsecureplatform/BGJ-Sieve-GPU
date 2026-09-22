from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CudaSourceRegressionTests(unittest.TestCase):
    def test_dual_hash_bucket_writes_are_saturated(self):
        source = (ROOT / "src" / "kernel_dhb.cu").read_text()
        self.assertNotIn("min(t_sbuc_num, out_max_size - pos)", source)
        self.assertEqual(source.count("pos < out_max_size ?"), 2)


if __name__ == "__main__":
    unittest.main()
