// One real sieve phase, with no BKZ insertion or checkpoint I/O in the timer.
// Compile twice so the serial executable uses the normal weak MPI stubs.
#ifdef HD_PHASE_BENCH_MPI
#include <mpi.h>
#endif
#include <filesystem>
#include <chrono>
#define main bgj_serial_benchmark_unused_main
#include "bin_hd_sieve.cu"
#undef main

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: %s BASIS CSD SCRATCH\n", argv[0]); return 2; }
    int rank = 0;
#ifdef HD_PHASE_BENCH_MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    if (provided < MPI_THREAD_SERIALIZED) MPI_Abort(MPI_COMM_WORLD, 2);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    namespace fs = std::filesystem;
    const fs::path input = fs::absolute(argv[1]);
    const fs::path root = fs::absolute(argv[3]);
    char *end = nullptr; const long dimension = strtol(argv[2], &end, 10);
    if (*end || dimension < 40 || dimension > Pool_hd_t::host_vec_nbytes) return 2;
    fs::create_directories(root);
    const fs::path scratch = root/("rank"+std::to_string(rank));
    // Claim our directory atomically BEFORE runtime initialization creates it
    // for rank-local logs. Never accept a pre-existing run, even if empty.
    std::error_code scratch_error;
    if (!fs::create_directory(scratch, scratch_error)) {
        fprintf(stderr, "benchmark scratch already exists or cannot be created: %s (%s)\n",
                scratch.c_str(), scratch_error ? scratch_error.message().c_str() : "existing directory");
#ifdef HD_PHASE_BENCH_MPI
        MPI_Abort(MPI_COMM_WORLD, 2);
#endif
        return 2;
    }
#ifdef HD_PHASE_BENCH_MPI
    if (mpi_sieve_runtime_init(root.c_str(), 256, false, true)) MPI_Abort(MPI_COMM_WORLD, 2);
#endif
    fs::current_path(scratch);
    for (const char *kind : {".pool", ".bucket", ".sol", ".uid"})
        for (int i = 0; i < hw::ssd_num; ++i) fs::create_directories(fs::path(kind)/hw::ssd_name_list[i]);
    _start_ck_allocator(); srand(1234+rank);
    Lattice_QP basis(input.c_str());
    int result = 0;
    {
        Pool_hd_t pool(&basis);
        if (dimension >= basis.NumRows() || pool.set_sieving_context(basis.NumRows()-dimension, basis.NumRows())) result = 2;
        pool.set_num_threads(configured_pool_threads()); pool.set_boost_depth(0);
        const long target = BGJ1_SIZE_RATIO*pow(4./3., dimension*.5)-5;
        const auto begin = std::chrono::steady_clock::now();
        if (!rank) {
            printf("[PHASE-BENCH] sampling_start ranks=%d csd=%ld target_vectors=%ld\n",
                   mpi_sieve_world(), dimension, target);
            fflush(stdout);
        }
        if (mpi_sieve_active()) {
            if (pool.sampling(target/2+4096) || mpi_sieve_partition_pool(&pool, target)) result = 2;
        } else if (pool.sampling(target)) result = 2;
        if (!mpi_sieve_min_int(result ? 0 : 1)) {
#ifdef HD_PHASE_BENCH_MPI
            MPI_Abort(MPI_COMM_WORLD, 2);
#endif
            return 2;
        }
        const auto sampled = std::chrono::steady_clock::now();
        if (!rank) {
            printf("[PHASE-BENCH] sampling_done seconds=%.6f; sieve_start csd=%ld\n",
                   std::chrono::duration<double>(sampled-begin).count(), dimension);
            fflush(stdout);
        }
        const int sieve_result = result ? -2 : _sieve(&pool);
        const auto finished = std::chrono::steady_clock::now();
        // A stuck/converged return (-1) is part of the sieve stopping criterion.
        if (sieve_result < -1) result = 2;
        const auto vectors = mpi_sieve_global_u64(pool.pwc_manager->num_vec());
        if (!rank) {
            printf("[PHASE-BENCH] ranks=%d csd=%ld sampling_seconds=%.6f sieve_seconds=%.6f vectors=%llu sieve_rc=%d\n",
                   mpi_sieve_world(), dimension,
                   std::chrono::duration<double>(sampled-begin).count(),
                   std::chrono::duration<double>(finished-sampled).count(),
                   (unsigned long long)vectors, sieve_result);
            fflush(stdout);
        }
        // Only this disposable, freshly claimed benchmark pool is discarded.
        // Leaving dirty chunks queued with writeback/spill disabled would make
        // its destructor wait forever. This is outside the timed sieve; real
        // BKZ persistence and checkpoint paths are deliberately unchanged.
        pool.pwc_manager->wait_work();
        for (long id = 0; id < pool.pwc_manager->num_chunks(); ++id)
            if (pool.pwc_manager->chunk_size(id) <= Pool_hd_t::chunk_max_nvecs)
                pool.pwc_manager->release_del(id);
    }
    _destroy_pool_hd_device_buffers();
    _destory_ck_allocator();
#ifdef HD_PHASE_BENCH_MPI
    mpi_sieve_runtime_finalize(); MPI_Finalize();
#endif
    return result;
}
