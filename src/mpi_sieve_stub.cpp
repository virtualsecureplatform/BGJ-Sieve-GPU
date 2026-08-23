#include "../include/mpi_sieve.h"

#define BGJ_WEAK __attribute__((weak))

BGJ_WEAK bool mpi_sieve_active() { return false; }
BGJ_WEAK void mpi_sieve_request_stop() {}
BGJ_WEAK bool mpi_sieve_should_stop() { return false; }
BGJ_WEAK int mpi_sieve_rank() { return 0; }
BGJ_WEAK int mpi_sieve_world() { return 1; }
BGJ_WEAK int mpi_sieve_resumed_csd() { return -1; }
BGJ_WEAK int mpi_sieve_sync_centers(int8_t *, std::size_t) { return 0; }
BGJ_WEAK int mpi_sieve_sync_uid_coeffs(uint64_t *, std::size_t) { return 0; }
BGJ_WEAK uint64_t mpi_sieve_global_u64(uint64_t value) { return value; }
BGJ_WEAK int mpi_sieve_min_int(int value) { return value; }
BGJ_WEAK int mpi_sieve_global_score_stat(const uint32_t *local, uint64_t *global,
                                         std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) global[i] = local[i];
    return 0;
}
BGJ_WEAK int mpi_sieve_exchange_buckets(bwc_manager_t *, swc_manager_t *,
                                        ut_checker_t *, const int32_t *, int, int) {
    return 0;
}
BGJ_WEAK int mpi_sieve_filter_candidates(int, int8_t *, int32_t *, uint16_t *,
                                         uint64_t *, int size) { return size; }
BGJ_WEAK int mpi_sieve_flush_candidates(int, swc_manager_t *, ut_checker_t *) { return 0; }
BGJ_WEAK int mpi_sieve_global_stuck(uint64_t checked, uint64_t not_inserted, int csd) {
    (void)csd;
    return not_inserted * 36 < checked && checked > 100;
}
BGJ_WEAK int mpi_sieve_partition_pool(Pool_hd_t *, long) { return 0; }
BGJ_WEAK int mpi_sieve_redistribute_pool(Pool_hd_t *) { return 0; }
BGJ_WEAK int mpi_sieve_prepare_working_pool(Pool_hd_t *) { return 0; }
BGJ_WEAK int mpi_sieve_checkpoint(Pool_hd_t *) { return 0; }
BGJ_WEAK void mpi_sieve_report_dimension(Pool_hd_t *, const char *) {}
