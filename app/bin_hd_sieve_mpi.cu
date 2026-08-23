#include <mpi.h>
#include <cuda_runtime.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <csignal>
#include <filesystem>

#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../include/mpi_sieve.h"

#define main bgj_serial_main
#include "bin_hd_sieve.cu"
#undef main

static int mkdir_one(const std::string &path) {
    return (!mkdir(path.c_str(), 0755) || errno == EEXIST) ? 0 : -1;
}

static void request_mpi_stop(int) { mpi_sieve_request_stop(); }

int main(int argc, char **argv) {
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided) != MPI_SUCCESS ||
        provided < MPI_THREAD_SERIALIZED) {
        fprintf(stderr, "[MPI] MPI_THREAD_SERIALIZED is required\n");
        return 2;
    }

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::string checkpoint_root;
    int frame_mib = 256;
    bool resume = true;
    std::vector<std::string> kept;
    kept.emplace_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--checkpoint-root")) {
            if (++i >= argc) { MPI_Abort(MPI_COMM_WORLD, 2); }
            checkpoint_root = argv[i];
        } else if (!strcmp(argv[i], "--mpi-frame-mib")) {
            if (++i >= argc) { MPI_Abort(MPI_COMM_WORLD, 2); }
            frame_mib = atoi(argv[i]);
        } else if (!strcmp(argv[i], "--resume")) {
            if (++i >= argc) { MPI_Abort(MPI_COMM_WORLD, 2); }
            if (!strcmp(argv[i], "auto")) resume = true;
            else if (!strcmp(argv[i], "none")) resume = false;
            else { if (!rank) fprintf(stderr, "[MPI] --resume must be auto or none\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
        } else {
            kept.emplace_back(argv[i]);
        }
    }
    if (checkpoint_root.empty()) {
        if (!rank) fprintf(stderr, "[MPI] --checkpoint-root is required\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    if (!resume) {
        const char *job = getenv("SLURM_JOB_ID");
        checkpoint_root += "/run-" + std::string(job ? job : "manual");
    }

    // Make input/output paths independent of the rank-local working directory.
    for (std::size_t i = 1; i < kept.size(); ++i) {
        const bool path_arg = kept[i] == "--input" || kept[i] == "-i" ||
                              kept[i] == "--output" || kept[i] == "-o";
        if (path_arg && i + 1 < kept.size()) {
            char resolved[PATH_MAX];
            if (realpath(kept[i + 1].c_str(), resolved)) kept[i + 1] = resolved;
            ++i;
        }
    }

    if (mpi_sieve_runtime_init(checkpoint_root.c_str(), frame_mib, resume))
        MPI_Abort(MPI_COMM_WORLD, 2);
    signal(SIGUSR1, request_mpi_stop);
    signal(SIGTERM, request_mpi_stop);

    int gpu_count = 0;
    if (cudaGetDeviceCount(&gpu_count) != cudaSuccess || gpu_count != 4) {
        fprintf(stderr, "[MPI rank %d] exactly four visible GPUs are required (got %d)\n",
                rank, gpu_count);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    const std::string rank_dir = checkpoint_root + "/rank" + std::to_string(rank);
    if (mkdir_one(rank_dir) || chdir(rank_dir.c_str())) {
        fprintf(stderr, "[MPI rank %d] cannot enter %s: %s\n", rank,
                rank_dir.c_str(), strerror(errno));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    // These are run-local derived caches.  In particular, retaining .uid while
    // loading a pool causes every restored vector to look like a duplicate.
    for (const char *top : {".bucket", ".sol", ".uid"}) {
        std::error_code ec;
        std::filesystem::remove_all(top, ec);
        if (ec) MPI_Abort(MPI_COMM_WORLD, 2);
    }
    for (const char *top : {".pool", ".bucket", ".sol", ".uid"}) {
        if (mkdir_one(top)) MPI_Abort(MPI_COMM_WORLD, 2);
        for (const char *ssd : {"0", "1"}) {
            if (mkdir_one(std::string(top) + "/" + ssd))
                MPI_Abort(MPI_COMM_WORLD, 2);
        }
    }
    if (rank) {
        freopen("rank.log", "a", stdout);
        freopen("rank.err", "a", stderr);
    }

    std::vector<char *> serial_argv;
    for (std::string &arg : kept) serial_argv.push_back(arg.data());
    int rc = bgj_serial_main((int)serial_argv.size(), serial_argv.data());
    mpi_sieve_runtime_finalize();
    MPI_Finalize();
    return rc;
}
