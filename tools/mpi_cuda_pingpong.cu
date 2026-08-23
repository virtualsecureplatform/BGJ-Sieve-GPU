#include <cuda_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

void fail(const char *what, int rank, const char *detail) {
    std::fprintf(stderr, "rank %d: %s: %s\n", rank, what, detail);
    std::fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 2);
}

void check_cuda(cudaError_t status, const char *what, int rank) {
    if (status != cudaSuccess) fail(what, rank, cudaGetErrorString(status));
}

void check_mpi(int status, const char *what, int rank) {
    if (status == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(status, message, &length);
    message[length] = '\0';
    fail(what, rank, message);
}

int iterations_for(size_t bytes) {
    if (bytes >= (256ULL << 20)) return 3;
    if (bytes >= (64ULL << 20)) return 5;
    if (bytes >= (16ULL << 20)) return 10;
    if (bytes >= (1ULL << 20)) return 30;
    if (bytes >= (64ULL << 10)) return 100;
    return 1000;
}

void run_pingpong(const char *kind, void *buffer, int rank, int world_size,
                  const std::vector<size_t> &sizes) {
    const int pairs = world_size / 2;
    const int peer = rank < pairs ? rank + pairs : rank - pairs;

    if (rank == 0) {
        std::printf("kind,ranks,pairs,bytes,iters,one_way_latency_us,per_pair_GBps,aggregate_GBps\n");
    }

    for (size_t bytes : sizes) {
        const int count = static_cast<int>(bytes);
        const int iterations = iterations_for(bytes);
        const int warmups = std::min(10, std::max(2, iterations / 10));

        check_mpi(MPI_Barrier(MPI_COMM_WORLD), "warmup barrier", rank);
        for (int i = 0; i < warmups; ++i) {
            if (rank < pairs) {
                check_mpi(MPI_Send(buffer, count, MPI_BYTE, peer, 10, MPI_COMM_WORLD), "warmup send", rank);
                check_mpi(MPI_Recv(buffer, count, MPI_BYTE, peer, 11, MPI_COMM_WORLD,
                                   MPI_STATUS_IGNORE), "warmup receive", rank);
            } else {
                check_mpi(MPI_Recv(buffer, count, MPI_BYTE, peer, 10, MPI_COMM_WORLD,
                                   MPI_STATUS_IGNORE), "warmup receive", rank);
                check_mpi(MPI_Send(buffer, count, MPI_BYTE, peer, 11, MPI_COMM_WORLD), "warmup send", rank);
            }
        }

        check_mpi(MPI_Barrier(MPI_COMM_WORLD), "timed barrier", rank);
        const double started = MPI_Wtime();
        for (int i = 0; i < iterations; ++i) {
            if (rank < pairs) {
                check_mpi(MPI_Send(buffer, count, MPI_BYTE, peer, 20, MPI_COMM_WORLD), "timed send", rank);
                check_mpi(MPI_Recv(buffer, count, MPI_BYTE, peer, 21, MPI_COMM_WORLD,
                                   MPI_STATUS_IGNORE), "timed receive", rank);
            } else {
                check_mpi(MPI_Recv(buffer, count, MPI_BYTE, peer, 20, MPI_COMM_WORLD,
                                   MPI_STATUS_IGNORE), "timed receive", rank);
                check_mpi(MPI_Send(buffer, count, MPI_BYTE, peer, 21, MPI_COMM_WORLD), "timed send", rank);
            }
        }
        const double local_elapsed = MPI_Wtime() - started;
        double elapsed = 0.0;
        check_mpi(MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
                             MPI_COMM_WORLD), "timing reduction", rank);

        if (rank == 0) {
            const double round_seconds = elapsed / iterations;
            const double latency_us = round_seconds * 0.5e6;
            const double per_pair_gbps = (2.0 * static_cast<double>(bytes)) /
                                         round_seconds / 1.0e9;
            std::printf("%s,%d,%d,%zu,%d,%.3f,%.3f,%.3f\n", kind, world_size,
                        pairs, bytes, iterations, latency_us, per_pair_gbps,
                        per_pair_gbps * pairs);
            std::fflush(stdout);
        }
    }
}

}  // namespace

int main(int argc, char **argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size != 2 && world_size != 8) {
        if (rank == 0) std::fprintf(stderr, "expected 2 or 8 MPI ranks, got %d\n", world_size);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    MPI_Comm local_comm = MPI_COMM_NULL;
    check_mpi(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                  MPI_INFO_NULL, &local_comm), "local communicator", rank);
    int local_rank = 0;
    int local_size = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_size);
    const int expected_local_size = world_size / 2;
    if (local_size != expected_local_size) {
        char detail[128];
        std::snprintf(detail, sizeof(detail), "expected %d local ranks, got %d",
                      expected_local_size, local_size);
        fail("rank placement", rank, detail);
    }

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount", rank);
    if (device_count < local_size) fail("GPU placement", rank, "not enough visible GPUs");
    check_cuda(cudaSetDevice(local_rank), "cudaSetDevice", rank);

    char hostname[MPI_MAX_PROCESSOR_NAME];
    int hostname_length = 0;
    MPI_Get_processor_name(hostname, &hostname_length);
    hostname[hostname_length] = '\0';
    char gpu_name[256];
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, local_rank), "cudaGetDeviceProperties", rank);
    std::snprintf(gpu_name, sizeof(gpu_name), "%s", properties.name);
    std::printf("rank=%d local_rank=%d host=%s gpu=%d name=%s mpi_thread=%d\n",
                rank, local_rank, hostname, local_rank, gpu_name, provided);
    std::fflush(stdout);

    const std::vector<size_t> sizes = {
        1, 8, 64, 512, 4096, 32768, 262144, 1ULL << 20, 4ULL << 20,
        16ULL << 20, 64ULL << 20, 256ULL << 20, 512ULL << 20,
    };
    const size_t max_bytes = sizes.back();

    void *host_buffer = nullptr;
    check_cuda(cudaMallocHost(&host_buffer, max_bytes), "cudaMallocHost", rank);
    std::memset(host_buffer, rank, max_bytes);
    run_pingpong("host_pinned", host_buffer, rank, world_size, sizes);
    check_cuda(cudaFreeHost(host_buffer), "cudaFreeHost", rank);

    void *device_buffer = nullptr;
    check_cuda(cudaMalloc(&device_buffer, max_bytes), "cudaMalloc", rank);
    check_cuda(cudaMemset(device_buffer, rank, max_bytes), "cudaMemset", rank);
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", rank);
    run_pingpong("cuda_direct", device_buffer, rank, world_size, sizes);
    check_cuda(cudaFree(device_buffer), "cudaFree", rank);

    MPI_Comm_free(&local_comm);
    MPI_Finalize();
    return 0;
}
