// Small, bounded network probe. Compile/run only in a compute allocation.
#include <mpi.h>
#include <algorithm>
#include <cstdio>
#include <vector>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world != 2) MPI_Abort(MPI_COMM_WORLD, 2);
    char host[MPI_MAX_PROCESSOR_NAME]; int length;
    MPI_Get_processor_name(host, &length);
    printf("[TRANSPORT-PROBE] rank=%d host=%s\n", rank, host); fflush(stdout);
    const int bytes = 64*1024*1024, repeats = 16;
    std::vector<unsigned char> sent(bytes, 17+rank), received(bytes);
    MPI_Barrier(MPI_COMM_WORLD);
    const double begin = MPI_Wtime();
    for (int i = 0; i < repeats; ++i)
        if (MPI_Sendrecv(sent.data(), bytes, MPI_BYTE, 1-rank, 91,
                         received.data(), bytes, MPI_BYTE, 1-rank, 91,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE) != MPI_SUCCESS)
            MPI_Abort(MPI_COMM_WORLD, 2);
    const double seconds = MPI_Wtime()-begin;
    if (!std::all_of(received.begin(), received.end(),
                     [rank](unsigned char b) { return b == 17+1-rank; }))
        MPI_Abort(MPI_COMM_WORLD, 2);
    printf("[TRANSPORT-PROBE] rank=%d bytes_each_direction=%lld seconds=%.6f GiB_per_second=%.6f\n",
           rank, (long long)bytes*repeats, seconds, 1.0/seconds);
    fflush(stdout);
    MPI_Finalize(); return 0;
}
