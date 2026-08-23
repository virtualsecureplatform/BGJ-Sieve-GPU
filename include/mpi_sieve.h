#ifndef BGJ_MPI_SIEVE_H
#define BGJ_MPI_SIEVE_H

#include <cstddef>
#include <cstdint>

#include "pool_hd.h"

int mpi_sieve_runtime_init(const char *checkpoint_root, int frame_mib,
                           bool resume);
int mpi_sieve_runtime_finalize();
void mpi_sieve_request_stop();
bool mpi_sieve_should_stop();

// These entry points have weak, serial no-op definitions in libllib.  The
// hd_sieve_mpi executable supplies the strong MPI implementation, keeping the
// normal hd_sieve binary and its ABI unchanged.
bool mpi_sieve_active();
int mpi_sieve_rank();
int mpi_sieve_world();
// Returns the completed checkpoint dimension restored by --resume auto, or -1.
int mpi_sieve_resumed_csd();
int mpi_sieve_sync_centers(int8_t *centers, std::size_t nbytes);
int mpi_sieve_sync_uid_coeffs(uint64_t *coeffs, std::size_t count);
uint64_t mpi_sieve_global_u64(uint64_t value);
int mpi_sieve_min_int(int value);
int mpi_sieve_global_score_stat(const uint32_t *local, uint64_t *global,
                                std::size_t count);
int mpi_sieve_exchange_buckets(bwc_manager_t *bwc, swc_manager_t *swc,
                               ut_checker_t *ut_checker, const int32_t *bucket_ids,
                               int num_buckets, int csd);
int mpi_sieve_filter_candidates(int csd, int8_t *vec, int32_t *norm,
                                uint16_t *score, uint64_t *uid, int size);
int mpi_sieve_flush_candidates(int csd, swc_manager_t *swc,
                               ut_checker_t *ut_checker);
int mpi_sieve_global_stuck(uint64_t checked, uint64_t not_inserted, int csd);
int mpi_sieve_partition_pool(Pool_hd_t *pool, long target_global_size);
int mpi_sieve_redistribute_pool(Pool_hd_t *pool);
int mpi_sieve_checkpoint(Pool_hd_t *pool);
void mpi_sieve_report_dimension(Pool_hd_t *pool, const char *phase);

#endif
