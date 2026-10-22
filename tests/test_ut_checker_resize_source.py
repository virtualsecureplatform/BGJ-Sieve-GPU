"""Cheap guardrails supplement (but never replace) the compiled sanitizer test."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CheckerResizeSourceTests(unittest.TestCase):
    def test_transactional_resize_and_shared_cleanup(self):
        source = (ROOT/'src/pool_hd.cpp').read_text()
        resizing = source.split('int ut_checker_t::set_max_holding(')[1].split('int ut_checker_t::set_exp_batch(')[0]
        self.assertNotIn('realloc(', resizing)
        self.assertIn('checker_buffer_t staged[4]', resizing)
        self.assertLess(resizing.index('if (!staged[i].base) return -2'), resizing.index('_release_buffers()'))
        self.assertIn('return -1', resizing)
        self.assertIn('std::free(view - 1)', source)
        destructor = source.split('ut_checker_t::~ut_checker_t()')[1].split('int ut_checker_t::set_num_threads')[0]
        self.assertLess(destructor.index('_ut_pool.wait_sleep()'), destructor.index('_release_buffers()'))

    def test_compute_build_gates_on_real_sanitized_regressions(self):
        make = (ROOT/'app/Makefile').read_text()
        self.assertIn('-fsanitize=address,undefined', make)
        self.assertIn('--wrap=calloc', make)
        script = (ROOT/'reproduce/svp167-search/mpi_bkz_validate.sbatch').read_text()
        self.assertLess(script.index('ASAN_OPTIONS=detect_leaks=1'), script.index('if [[ "${BUILD_ONLY:-0}" == 1 ]]'))
        self.assertIn('--cache-pressure-only', script)
        pressure = (ROOT/'app/mpi_sieve_pipeline.inc').read_text().split('int pipeline_cache_pressure(')[1]
        self.assertIn('if (checker.set_max_holding(8))', pressure)


if __name__ == '__main__':
    unittest.main()
