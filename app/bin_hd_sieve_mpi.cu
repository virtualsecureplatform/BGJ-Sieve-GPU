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
#include <sys/file.h>
#include <unistd.h>
#include <ucontext.h>
#include <dlfcn.h>

#include "../include/mpi_sieve.h"

#define main bgj_serial_main
#include "bin_hd_sieve.cu"
#undef main
#include "mpi_bkz_driver.inc"

static int mkdir_one(const std::string &path) {
    return (!mkdir(path.c_str(), 0755) || errno == EEXIST) ? 0 : -1;
}

static void request_mpi_stop(int) { mpi_sieve_request_stop(); }

static uintptr_t crash_image_base = 0;
static void trace_mpi_crash(int signal, siginfo_t *, void *context) {
    const char label[] = "[MPI-CRASH] image-relative pc=0x";
    write(STDERR_FILENO, label, sizeof(label)-1);
    // Avoid malloc, stdio and stack unwinding: another crashing worker may
    // hold their internal locks. x86_64 is the supported GPU-node platform.
    const auto *uc = static_cast<ucontext_t *>(context);
    uintptr_t pc = (uintptr_t)uc->uc_mcontext.gregs[REG_RIP] - crash_image_base;
    char hex[17]; hex[16] = '\n';
    for (int i = 15; i >= 0; --i) { hex[i] = "0123456789abcdef"[pc & 15]; pc >>= 4; }
    write(STDERR_FILENO, hex, sizeof(hex));
    // A null indirect call has RIP=0; its caller's return address is still
    // at RSP. Print it without invoking an unwinder or touching heap state.
    const char caller[] = "[MPI-CRASH] stack-return image-relative=0x";
    write(STDERR_FILENO, caller, sizeof(caller)-1);
    const uintptr_t *stack = reinterpret_cast<const uintptr_t *>(uc->uc_mcontext.gregs[REG_RSP]);
    pc = stack[0] - crash_image_base;
    for (int i = 15; i >= 0; --i) { hex[i] = "0123456789abcdef"[pc & 15]; pc >>= 4; }
    write(STDERR_FILENO, hex, sizeof(hex));
    _exit(128 + signal);
}

int main(int argc, char **argv) {
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided) != MPI_SUCCESS ||
        provided < MPI_THREAD_SERIALIZED) {
        fprintf(stderr, "[MPI] MPI_THREAD_SERIALIZED is required\n");
        return 2;
    }

    int rank = 0;
    if (getenv("HD_MPI_CRASH_TRACE")) {
        Dl_info info = {};
        if (dladdr((void *)&trace_mpi_crash, &info)) crash_image_base = (uintptr_t)info.dli_fbase;
        struct sigaction action = {};
        action.sa_sigaction = trace_mpi_crash;
        sigemptyset(&action.sa_mask);
        action.sa_flags = SA_RESETHAND | SA_SIGINFO;
        sigaction(SIGSEGV, &action, nullptr);
        sigaction(SIGABRT, &action, nullptr);
    }
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
    bool bkz_mode = false;
    for (size_t i = 1; i+1 < kept.size(); ++i)
        if ((kept[i] == "--task" || kept[i] == "-t") &&
            (kept[i+1] == "bkz" || kept[i+1] == "b" || kept[i+1] == "pump" || kept[i+1] == "p"))
            bkz_mode = true;
    if (!resume && !bkz_mode) {
        const char *job = getenv("SLURM_JOB_ID");
        checkpoint_root += "/run-" + std::string(job ? job : "manual");
    }
    checkpoint_root = std::filesystem::absolute(checkpoint_root).lexically_normal().string();
    mpi_bkz::root = checkpoint_root;
    mpi_bkz::resume_enabled = resume;

    // Make input/output paths independent of the rank-local working directory.
    for (std::size_t i = 1; i < kept.size(); ++i) {
        const bool path_arg = kept[i] == "--input" || kept[i] == "-i" ||
                              kept[i] == "--output" || kept[i] == "-o";
        if (path_arg && i + 1 < kept.size()) {
            char resolved[PATH_MAX];
            if (realpath(kept[i + 1].c_str(), resolved)) kept[i + 1] = resolved;
            else kept[i + 1] = std::filesystem::absolute(kept[i+1]).lexically_normal().string();
            if (kept[i+1].size() >= 240) {
                fprintf(stderr, "[MPI] input/output path exceeds task parser capacity\n");
                MPI_Abort(MPI_COMM_WORLD, 2);
            }
            ++i;
        }
    }

    if (mpi_sieve_runtime_init(checkpoint_root.c_str(), frame_mib, resume, bkz_mode))
        MPI_Abort(MPI_COMM_WORLD, 2);
    int run_lock = -1;
    if (bkz_mode && rank == 0) {
        run_lock = open((checkpoint_root + "/RUN.lock").c_str(), O_CREAT | O_RDWR, 0644);
        if (run_lock < 0 || flock(run_lock, LOCK_EX | LOCK_NB)) {
            fprintf(stderr, "[MPI] checkpoint root is already in use\n");
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
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
    int rc = 0;
    try {
        if (bkz_mode) {
            task_config_t task;
            if (task.parse_args((int)serial_argv.size(), serial_argv.data()))
                throw std::runtime_error("invalid MPI BKZ arguments");
            if (const char *spill = getenv("HD_MPI_SPILL_TEST")) {
                Lattice_QP basis(task.basis_file);
                Pool_hd_t pool(&basis);
                if (pool.set_sieving_context(16, 80)) MPI_Abort(MPI_COMM_WORLD, 2);
                if (!strcmp(spill, "uid")) pool.uid_table->dump_table(0);
                else {
                    pwc_manager_t manager(1, 1, 256, !strcmp(spill, "bucket"));
                    manager.set_pool(&pool);
                    // set_dirname adds the leading dot itself (MULTI_SSD).
                    manager.set_dirname(spill);
                    long id = manager.create_chunk();
                    chunk_t *chunk = manager.fetch(id); chunk->size = 1;
                    chunk->norm[0] = 1; chunk->vec[0] = 1;
                    manager.sync_release(id); // Must fail before opening a disk file.
                }
                throw std::runtime_error("spill guard failed to abort");
            } else if (const char *control = getenv("HD_MPI_CONTROL_TEST")) {
                Lattice_QP basis(task.basis_file);
                rc = mpi_sieve_control_test(&basis, !strcmp(control, "mismatch"));
            } else rc = task.run();
            _destroy_pool_hd_device_buffers();
            _destory_ck_allocator();
        } else rc = bgj_serial_main((int)serial_argv.size(), serial_argv.data());
        if (rc && rc != 75) MPI_Abort(MPI_COMM_WORLD, rc);
    } catch (const std::exception &error) {
        fprintf(stderr, "[MPI rank %d] %s\n", rank, error.what()); fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    mpi_sieve_runtime_finalize();
    MPI_Finalize();
    if (run_lock >= 0) close(run_lock);
    return rc;
}
