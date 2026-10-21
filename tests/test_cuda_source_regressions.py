from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CudaSourceRegressionTests(unittest.TestCase):
    def test_phase_scratch_is_claimed_before_mpi_initialization(self):
        source = (ROOT / "app" / "bin_hd_phase_bench.cu").read_text()
        self.assertLess(source.index('fs::create_directory(scratch, scratch_error)'),
                        source.index('mpi_sieve_runtime_init('))
        self.assertNotIn('fs::exists(scratch)', source)
        self.assertLess(source.index('const auto finished ='),
                        source.index('phase_bench_pool_access_t::discard(pool)'))
        manager = (ROOT / 'include' / 'pool_hd.h').read_text()
        self.assertIn('friend struct phase_bench_pool_access_t;', manager)
        access = source.split('struct phase_bench_pool_access_t {', 1)[1].split('int main(', 1)[0]
        self.assertIn('pool.pwc_manager->release_del(id)', access)
        smoke = (ROOT / 'tools' / 'test_phase_bench.py').read_text()
        self.assertIn("('mpi', 2, 'hd_phase_bench_mpi')", smoke)
        self.assertIn('assert repeated.returncode != 0', smoke)
        self.assertIn("marker.read_text()", smoke)

    def test_phase_timeout_defaults_to_three_hours(self):
        source = (ROOT / "reproduce" / "svp167-search" / "mpi_phase_bench.sbatch").read_text()
        self.assertIn('BENCH_TIMEOUT_SECONDS:-10800', source)
        self.assertIn('--kill-after=60 "${TEST_TIMEOUT_SECONDS}" srun', source)

    def test_phase_diagnostics_run_inside_each_rank(self):
        source = (ROOT / "reproduce" / "svp167-search" / "mpi_phase_bench.sbatch").read_text()
        self.assertIn('bash "${RANK_WRAPPER}"', source)
        self.assertIn('archive "${LAUNCH_COMMIT}"', source)
        monitor = (ROOT / "reproduce" / "svp167-search" / "phase_bench_rank.sh").read_text()
        self.assertIn('nvidia-smi --query-gpu=', monitor)
        self.assertIn('ps -L -p "${native}"', monitor)
        self.assertIn('thread apply all bt 8', monitor)
        self.assertIn('1200', monitor)
        self.assertIn('kill "${monitor_pid}"', monitor)
        self.assertNotIn('/environ', monitor)

    def test_no_spill_swc_borrows_only_unused_pool_budget(self):
        policy = (ROOT / "include" / "sieve_cache_budget.h").read_text()
        self.assertIn('pool_capacity - phase_pool_chunks', policy)
        self.assertIn('std::min(unused_pool,', policy)
        source = (ROOT / "src" / "bgj_hd.cpp").read_text()
        self.assertIn('base = no_spill_swc_cache_chunks(', source)
        self.assertIn('1000 + RED_MAX_NUM_THREADS +', source)
        self.assertIn('host_pool + requested_chunks + 256', source)
        self.assertNotIn('PWC_DEFAULT_MAX_CACHED_CHUNKS +\n                                       requested_chunks', source)
        native = (ROOT / "app" / "mpi_sieve_runtime.cpp").read_text()
        self.assertIn('PASS: bounded no-spill SWC cache borrowing', native)
        self.assertIn('phase_pool + slots > pool_capacity + base_swc', native)

    def test_cache_timeout_identifies_manager(self):
        source = (ROOT / "include" / "pool_hd.h").read_text()
        self.assertIn('[cache-wait-timeout] manager=%s kind=%s', source)

    def test_phase_benchmark_reports_sampling_boundaries(self):
        source = (ROOT / "app" / "bin_hd_phase_bench.cu").read_text()
        self.assertIn('[PHASE-BENCH] sampling_start', source)
        self.assertIn('[PHASE-BENCH] sampling_done', source)

    def test_phase_benchmark_uses_separate_allocations(self):
        source = (ROOT / "reproduce" / "svp167-search" / "mpi_phase_bench.sbatch").read_text()
        self.assertIn('#SBATCH --rsc=g=4', source)
        self.assertIn('serial) BACKEND=serial; NODES=1; GPUS=4;', source)
        self.assertIn('mpi) BACKEND=mpi; NODES=2; GPUS=8;', source)
        self.assertIn('requires exactly ${NODES} allocated node(s)', source)
        self.assertIn('--gpus="${GPUS}" --gres=gpu:4', source)
        self.assertIn('--nodes="${NODES}" --ntasks="${NODES}"', source)
        self.assertIn('--ntasks-per-node=1', source)
        self.assertIn('--cpus-per-task=128', source)
        self.assertIn('--immediate=60', source)

    def test_mpi_bucket_transport_batches_columnar_blocks(self):
        source = (ROOT / "app" / "mpi_sieve_runtime.cpp").read_text()
        exchange = source.split("int mpi_sieve_exchange_buckets(", 1)[1].split("int mpi_sieve_global_stuck", 1)[0]
        self.assertIn("64ULL<<20", exchange)
        self.assertIn("outgoing.size()+chunk_bytes <= batch_bytes", exchange)
        self.assertIn("mpi_append_bucket_soa", exchange)
        self.assertIn("count > Pool_hd_t::chunk_max_nvecs", exchange)
        self.assertIn("recv_header.version != 2", source)

    def test_checkpoint_parallelism_preserves_durability(self):
        source = (ROOT / "app" / "mpi_bkz_driver.inc").read_text()
        self.assertIn('requested >= 1 && requested <= 8', source)
        self.assertIn('std::rethrow_exception(error)', source)
        self.assertIn('durable(path);', source)
        self.assertIn('copy_pool_tree(".pool", dst/"pool")', source)
        self.assertIn('file_hash(dir/entry.path) == entry.hash', source)

    def test_mpi_compaction_uses_exclusive_two_ended_packing(self):
        source = (ROOT / "src" / "pool_hd.cpp").read_text()
        packing = source.split("int Pool_hd_t::mpi_compact_dense()", 1)[1].split("int Pool_hd_t::mpi_retain_owner", 1)[0]
        self.assertIn("while (left <= right)", packing)
        self.assertNotIn("#pragma omp", packing)
        self.assertIn("pwc_manager->num_vec() == expected", packing)
        driver = (ROOT / "app" / "mpi_bkz_driver.inc").read_text()
        self.assertIn("pool.mpi_compact_dense()", driver)

    def test_insertion_reconstructs_before_redistribution(self):
        source = (ROOT / "app" / "mpi_bkz_driver.inc").read_text()
        insertion = source.split("pool.insert(s.cursor++", 1)[1]
        self.assertLess(insertion.index("pool.canonicalize_records()"),
                        insertion.index("redistribute(pool)"))

    def test_checkpoint_restore_does_not_repair_saved_vectors(self):
        source = (ROOT / "app" / "mpi_bkz_driver.inc").read_text()
        self.assertIn("restore_pool_exact(pool, s);", source)
        self.assertNotIn("pool.load(0)", source)
        restore = source.split("void restore_pool_exact(", 1)[1].split("} // namespace mpi_bkz", 1)[0]
        self.assertIn("duplicate exact-restore UID", restore)
        self.assertIn("exact-restore vector count mismatch", restore)
        self.assertIn("in.read((char *)chunk->vec", restore)

    def test_transform_finalization_recounts_and_packs_live_slots(self):
        source = (ROOT / "src" / "pool_hd.cpp").read_text()
        finalization = source.split("if (_type == type_others) {\n        uint32_t _score_stat", 1)[1]
        self.assertIn("live += _in_check[i].score[j] != 0;", finalization)
        self.assertLess(finalization.index("_normalize_chunk(&_in_check[i], CSD)"),
                        finalization.index("_pwc_manager->release_sync"))

    def test_mpi_bucket_wave_requires_supported_kernel(self):
        source = (ROOT / "src" / "bgj_hd_device.cu").read_text()
        self.assertIn("if (lim <= 0) return 0;", source)
        self.assertIn("[BUCKET-KERNEL] unsupported", source)
        run = source.split("int Bucketer_t::run()", 1)[1]
        self.assertIn("mpi_sieve_drain_buckets(_bwc, _swc, _ut_checker, _pool->CSD, 0)", run)
        self.assertIn("if (batch0 < 16)", run)

    def test_received_candidates_pack_reused_swc_chunks(self):
        source = (ROOT / "app" / "mpi_sieve_runtime.cpp").read_text()
        commit = source.split("int commit_candidates(", 1)[1].split("int make_dir(", 1)[0]
        self.assertLess(commit.index("_normalize_chunk(dst, csd)"),
                        commit.index("const int p = dst->size++"))
        self.assertIn("memset(dst->score + dst->size", commit)

    def test_mpi_candidates_wait_instead_of_aborting_on_capacity(self):
        source = (ROOT / "app" / "mpi_sieve_runtime.cpp").read_text()
        filtering = source.split("int mpi_sieve_filter_candidates(", 1)[1].split("int mpi_sieve_flush_candidates(", 1)[0]
        self.assertIn("while (g_remote_candidates.size() + stride > g_candidate_limit)", filtering)
        self.assertIn("lock.unlock();", filtering)
        self.assertIn("lock.lock();", filtering)
        self.assertNotIn("abort()", filtering)
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
