from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CudaSourceRegressionTests(unittest.TestCase):
    def test_received_candidates_pack_reused_swc_chunks(self):
        source = (ROOT / "app" / "mpi_sieve_runtime.cpp").read_text()
        commit = source.split("int commit_candidates(", 1)[1].split("int make_dir(", 1)[0]
        self.assertLess(commit.index("_normalize_chunk(dst, csd)"),
                        commit.index("const int p = dst->size++"))
        self.assertIn("memset(dst->score + dst->size", commit)

    def test_mpi_candidates_wait_instead_of_aborting_on_capacity(self):
        source = (ROOT / "app" / "mpi_sieve_runtime.cpp").read_text()
        filtering = source.split("int mpi_sieve_filter_candidates(", 1)[1].split("int mpi_sieve_flush_candidates(", 1)[0]
        self.assertIn("g_candidate_space.wait(lock", filtering)
        self.assertNotIn("abort()", filtering)
        self.assertIn("g_candidate_space.notify_all()", source)
        self.assertIn("mpi_sieve_drain_buckets", source)

    def test_mpi_redistribution_packs_sparse_chunks(self):
        source = (ROOT / "app" / "mpi_sieve_runtime.cpp").read_text()
        redistribute = source.split("int mpi_sieve_redistribute_pool(", 1)[1]
        self.assertLess(redistribute.index("_normalize_chunk(chunk, pool->CSD)"),
                        redistribute.index("const int old_size = chunk->size"))

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
