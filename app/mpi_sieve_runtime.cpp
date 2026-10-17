#include "../include/mpi_sieve.h"
#include "../include/bgj_hd.h"

#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

namespace {
int g_rank = 0;
int g_world = 1;
std::size_t g_frame_bytes = 256ULL << 20;
std::string g_checkpoint_root;
std::mutex g_candidate_mutex;
std::size_t g_candidate_limit = 1ULL << 30;
std::vector<uint8_t> g_remote_candidates;
std::atomic<uint64_t> g_bucket_sequence{0};
std::atomic<uint64_t> g_candidate_sequence{0};
std::atomic<uint64_t> g_processed_chunks{0};
std::atomic<bool> g_stop_requested{false};
int g_resumed_csd = -1;
std::atomic<uint64_t> g_sent_bytes{0};
std::atomic<uint64_t> g_recv_bytes{0};
double g_mpi_seconds = 0.0;
uint8_t *g_send_frame = nullptr;
uint8_t *g_recv_frame = nullptr;

constexpr int tag_bucket_count = 4100;
constexpr int tag_bucket_data = 4101;
constexpr int tag_candidate_count = 4200;
constexpr int tag_candidate_data = 4201;
constexpr int tag_pool_count = 4300;
constexpr int tag_pool_data = 4301;

struct wire_header_t {
    uint32_t magic;
    uint16_t version;
    uint16_t kind;
    uint32_t csd;
    uint32_t reserved;
    uint64_t sequence;
    uint64_t bytes;
    uint64_t checksum;
};
constexpr uint32_t wire_magic = 0x42474a4d; // BGJM

uint64_t checksum64(const uint8_t *data, uint64_t size) {
    uint64_t h = 1469598103934665603ULL;
    for (uint64_t i = 0; i < size; ++i) {
        h ^= data[i];
        h *= 1099511628211ULL;
    }
    return h;
}

double now_seconds() { return MPI_Wtime(); }

int exchange_bytes(const uint8_t *send_data, uint64_t send_bytes,
                   std::vector<uint8_t> &recv, int count_tag, int data_tag,
                   int csd, uint64_t sequence) {
    wire_header_t send_header = {wire_magic, 1, (uint16_t)count_tag,
                                 (uint32_t)csd, 0, sequence, send_bytes,
                                 checksum64(send_data, send_bytes)};
    wire_header_t recv_header = {};
    const double start = now_seconds();
    if (MPI_Sendrecv(&send_header, sizeof(send_header), MPI_BYTE, 1 - g_rank, count_tag,
                     &recv_header, sizeof(recv_header), MPI_BYTE, 1 - g_rank, count_tag,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE) != MPI_SUCCESS)
        return -1;
    if (recv_header.magic != wire_magic || recv_header.version != 1 ||
        recv_header.kind != count_tag || recv_header.csd != (uint32_t)csd ||
        recv_header.sequence != sequence) {
        fprintf(stderr, "[MPI rank %d] invalid wire header kind=%u csd=%u sequence=%llu\n",
                g_rank, recv_header.kind, recv_header.csd,
                (unsigned long long)recv_header.sequence);
        return -1;
    }
    const uint64_t recv_bytes = recv_header.bytes;
    // All callers stream bounded windows. Reject a corrupt/oversized header
    // before allocating peer-controlled memory.
    if (recv_bytes > (1ULL << 30) || send_bytes > (1ULL << 30)) {
        fprintf(stderr, "[MPI] oversized exchange payload\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    recv.resize(recv_bytes);
    uint64_t so = 0, ro = 0;
    while (so < send_bytes || ro < recv_bytes) {
        const int sn = (int)std::min<uint64_t>(g_frame_bytes, send_bytes - so);
        const int rn = (int)std::min<uint64_t>(g_frame_bytes, recv_bytes - ro);
        if (sn) memcpy(g_send_frame, send_data + so, sn);
        const void *sp = sn ? g_send_frame : nullptr;
        void *rp = rn ? g_recv_frame : nullptr;
        if (MPI_Sendrecv(sp, sn, MPI_BYTE, 1 - g_rank, data_tag,
                         rp, rn, MPI_BYTE, 1 - g_rank, data_tag,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE) != MPI_SUCCESS)
            return -1;
        if (rn) memcpy(recv.data() + ro, g_recv_frame, rn);
        so += sn;
        ro += rn;
    }
    g_sent_bytes += send_bytes;
    g_recv_bytes += recv_bytes;
    g_mpi_seconds += now_seconds() - start;
    if (checksum64(recv.data(), recv_bytes) != recv_header.checksum) {
        fprintf(stderr, "[MPI rank %d] payload checksum mismatch kind=%d sequence=%llu\n",
                g_rank, count_tag, (unsigned long long)sequence);
        return -1;
    }
    return 0;
}

int commit_candidates(const std::vector<uint8_t> &records, int csd,
                      swc_manager_t *swc, ut_checker_t *ut) {
    const int stride = csd + 14;
    if (records.size() % stride) return -1;
    const long count = records.size() / stride;
    chunk_t *dst = nullptr;
    for (long i = 0; i < count; ++i) {
        if (!dst) {
            dst = swc->fetch_for_write();
            if (!dst) return -1;
            // Returned partial SWC chunks can contain holes after UID dedup.
            // Match the local reducer's packing contract before appending.
            _normalize_chunk(dst, csd);
            if (dst->size < 0 || dst->size >= Pool_hd_t::chunk_max_nvecs)
                return -1;
        }
        const uint8_t *src = records.data() + i * (long)stride;
        const int p = dst->size++;
        memcpy(dst->u + p, src, 8);
        memcpy(dst->norm + p, src + 8, 4);
        memcpy(dst->score + p, src + 12, 2);
        memcpy(dst->vec + (long)csd * p, src + 14, csd);
        if (dst->size == Pool_hd_t::chunk_max_nvecs) {
            ut->task_commit(dst);
            dst = nullptr;
        }
    }
    if (dst) {
        memset(dst->score + dst->size, 0,
               (Pool_hd_t::chunk_max_nvecs - dst->size) * sizeof(uint16_t));
        memset(dst->norm + dst->size, 0,
               (Pool_hd_t::chunk_max_nvecs - dst->size) * sizeof(int32_t));
        ut->task_commit(dst);
    }
    return 0;
}

int make_dir(const std::string &path) {
    if (path.empty()) return -1;
    std::string current;
    for (char c : path) {
        current.push_back(c);
        if (c == '/' && current.size() > 1) mkdir(current.c_str(), 0755);
    }
    if (mkdir(path.c_str(), 0755) && errno != EEXIST) return -1;
    return 0;
}
} // namespace

int mpi_sieve_runtime_init(const char *checkpoint_root, int frame_mib,
                           bool resume, bool bkz) {
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_world);
    if (g_world != 2) {
        if (!g_rank) fprintf(stderr, "[MPI] exactly two ranks are required (got %d)\n", g_world);
        return -1;
    }
    char host[MPI_MAX_PROCESSOR_NAME] = {};
    int host_len = 0;
    MPI_Get_processor_name(host, &host_len);
    char hosts[2][MPI_MAX_PROCESSOR_NAME] = {};
    MPI_Allgather(host, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                  hosts, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, MPI_COMM_WORLD);
    if (!strcmp(hosts[0], hosts[1])) {
        if (!g_rank) fprintf(stderr, "[MPI] ranks must be placed on distinct nodes\n");
        return -1;
    }
    if (frame_mib < 1 || frame_mib > 1024) return -1;
    if (const char *limit = getenv("HD_MPI_CANDIDATE_QUEUE_KIB")) {
        char *end = nullptr;
        const long kib = strtol(limit, &end, 10);
        if (!*limit || *end || kib < 1 || kib > 1048576) return -1;
        g_candidate_limit = (std::size_t)kib << 10;
    }
    g_frame_bytes = (std::size_t)frame_mib << 20;
    if (cudaHostAlloc((void **)&g_send_frame, g_frame_bytes, cudaHostAllocPortable) != cudaSuccess ||
        cudaHostAlloc((void **)&g_recv_frame, g_frame_bytes, cudaHostAllocPortable) != cudaSuccess) {
        if (g_send_frame) cudaFreeHost(g_send_frame);
        if (g_recv_frame) cudaFreeHost(g_recv_frame);
        g_send_frame = g_recv_frame = nullptr;
        if (!g_rank) fprintf(stderr, "[MPI] unable to allocate pinned frame buffers\n");
        return -1;
    }
    g_checkpoint_root = checkpoint_root ? checkpoint_root : "";
    if (make_dir(g_checkpoint_root)) return -1;
    namespace fs = std::filesystem;
    const fs::path rank_root = fs::path(g_checkpoint_root) /
                               ("rank" + std::to_string(g_rank));
    fs::create_directories(rank_root);
    if (resume && !bkz) {
        std::ifstream latest(fs::path(g_checkpoint_root) / "LATEST");
        std::string line;
        long csd = -1;
        long world = -1;
        while (std::getline(latest, line))
            if (line.rfind("csd=", 0) == 0) csd = atol(line.c_str() + 4);
            else if (line.rfind("world=", 0) == 0) world = atol(line.c_str() + 6);
        if (csd >= 0) {
            if (world != g_world) {
                if (!g_rank) fprintf(stderr, "[MPI] checkpoint world %ld != runtime world %d\n",
                                     world, g_world);
                return -1;
            }
            const std::string generation = ".checkpoint-" + std::to_string(csd);
            const fs::path pool = rank_root / generation / "pool";
            if (!fs::is_directory(pool)) {
                fprintf(stderr, "[MPI rank %d] checkpoint CSD %ld is incomplete\n", g_rank, csd);
                return -1;
            }
            const fs::path link = rank_root / ".pool";
            const fs::path next = rank_root / ".pool.resume";
            std::error_code ec;
            fs::remove(next, ec);
            fs::create_directory_symlink(fs::path(generation) / "pool", next, ec);
            if (ec) return -1;
            fs::remove_all(link, ec);
            fs::rename(next, link, ec);
            if (ec) return -1;
            g_resumed_csd = (int)csd;
        }
    }
    if (!g_rank) {
        printf("[MPI] world=2 nodes=%s,%s transport=pinned-host frame=%d MiB resume=%s\n",
               hosts[0], hosts[1], frame_mib, resume ? "auto" : "none");
        fflush(stdout);
    }
    return 0;
}

int mpi_sieve_runtime_finalize() {
    if (!g_rank) {
        printf("[MPI] traffic sent=%.3f GiB received=%.3f GiB mpi_time=%.3f s\n",
               g_sent_bytes / (double)(1ULL << 30),
               g_recv_bytes / (double)(1ULL << 30), g_mpi_seconds);
        fflush(stdout);
    }
    if (g_send_frame) cudaFreeHost(g_send_frame);
    if (g_recv_frame) cudaFreeHost(g_recv_frame);
    g_send_frame = g_recv_frame = nullptr;
    return 0;
}

int mpi_sieve_select_insert(mpi_insert_candidate_t *candidate) {
    mpi_insert_candidate_t candidates[2];
    if (MPI_Allgather(candidate, sizeof(*candidate), MPI_BYTE, candidates,
                      sizeof(*candidate), MPI_BYTE, MPI_COMM_WORLD) != MPI_SUCCESS) return -1;
    int best = -1;
    for (int r = 0; r < 2; ++r) {
        const auto &c = candidates[r];
        if (!c.valid || !std::isfinite(c.objective)) continue;
        if (best < 0 || c.objective < candidates[best].objective ||
            (c.objective == candidates[best].objective && c.insert_pos < candidates[best].insert_pos))
            best = r;
    }
    *candidate = best < 0 ? mpi_insert_candidate_t{} : candidates[best];
    return 0;
}

mpi_sieve_metrics_t mpi_sieve_metrics() {
    std::lock_guard<std::mutex> lock(g_candidate_mutex);
    return {g_bucket_sequence.load(), g_candidate_sequence.load(), g_remote_candidates.size(),
            g_sent_bytes.load(), g_recv_bytes.load(), g_processed_chunks.load()};
}
void mpi_sieve_note_progress(uint64_t chunks) { g_processed_chunks.fetch_add(chunks, std::memory_order_relaxed); }

int mpi_sieve_control_test(Lattice_QP *basis, bool mismatch) {
    for (int test = 0; test < 5; ++test) {
        mpi_insert_candidate_t c;
        c.valid = test != 3 && !(test == 4 && g_rank == 0);
        c.objective = test == 0 ? (2-g_rank) : 1;
        c.insert_pos = test == 1 ? (2-g_rank) : 1;
        c.remove_pos = 5; c.coeff[0] = 7+g_rank;
        mpi_sieve_select_insert(&c);
        const int winner = test == 0 || test == 1 || test == 4 ? 1 : 0;
        if ((test == 3 && c.valid) || (test != 3 && (!c.valid || c.coeff[0] != 7+winner)))
            MPI_Abort(MPI_COMM_WORLD, 2);
    }
    if (mismatch) basis->get_b().hi[0][0] += g_rank;
    mpi_sieve_agree_basis(basis, "control-test");
    if (!g_rank) { printf("PASS: global insertion remote winner, ties, invalid candidates, basis agreement\n"); fflush(stdout); }
    return 0;
}

int mpi_sieve_agree_basis(Lattice_QP *basis, const char *phase) {
    // Compare both quad-float components, not a rounded display of the basis.
    uint64_t h = 1469598103934665603ULL;
    auto b = basis->get_b();
    for (long i = 0; i < basis->NumRows(); ++i) {
        h ^= checksum64((const uint8_t *)b.hi[i], basis->NumCols() * sizeof(double));
        h *= 1099511628211ULL;
        h ^= checksum64((const uint8_t *)b.lo[i], basis->NumCols() * sizeof(double));
        h *= 1099511628211ULL;
    }
    uint64_t hashes[2];
    MPI_Allgather(&h, 1, MPI_UINT64_T, hashes, 1, MPI_UINT64_T, MPI_COMM_WORLD);
    if (hashes[0] != hashes[1]) {
        fprintf(stderr, "[MPI] basis disagreement phase=%s\n", phase);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    return 0;
}

bool mpi_sieve_active() { return true; }
void mpi_sieve_request_stop() { g_stop_requested.store(true); }
bool mpi_sieve_should_stop() {
    int local = g_stop_requested.load() ? 1 : 0, global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    return global != 0;
}
int mpi_sieve_rank() { return g_rank; }
int mpi_sieve_world() { return g_world; }
int mpi_sieve_resumed_csd() { return g_resumed_csd; }

int mpi_sieve_sync_centers(int8_t *centers, std::size_t nbytes) {
    if (nbytes > INT32_MAX) return -1;
    return MPI_Bcast(centers, (int)nbytes, MPI_BYTE, 0, MPI_COMM_WORLD) == MPI_SUCCESS ? 0 : -1;
}

int mpi_sieve_sync_uid_coeffs(uint64_t *coeffs, std::size_t count) {
    if (count > INT32_MAX) return -1;
    return MPI_Bcast(coeffs, (int)count, MPI_UINT64_T, 0, MPI_COMM_WORLD) == MPI_SUCCESS ? 0 : -1;
}

uint64_t mpi_sieve_global_u64(uint64_t value) {
    uint64_t result = 0;
    MPI_Allreduce(&value, &result, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    return result;
}

int mpi_sieve_min_int(int value) {
    int result = 0;
    MPI_Allreduce(&value, &result, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    return result;
}

int mpi_sieve_global_score_stat(const uint32_t *local, uint64_t *global,
                                std::size_t count) {
    if (count > INT32_MAX) return -1;
    std::vector<uint64_t> tmp(count);
    for (std::size_t i = 0; i < count; ++i) tmp[i] = local[i];
    return MPI_Allreduce(tmp.data(), global, (int)count, MPI_UINT64_T, MPI_SUM,
                         MPI_COMM_WORLD) == MPI_SUCCESS ? 0 : -1;
}

int mpi_sieve_filter_candidates(int csd, int8_t *vec, int32_t *norm,
                                uint16_t *score, uint64_t *uid, int size) {
    if (size <= 0) return size;
    const int stride = csd + 14;
    std::unique_lock<std::mutex> lock(g_candidate_mutex);
    int local = 0;
    for (int i = 0; i < size; ++i) {
        const uint64_t normalized = uid[i] > UINT64_MAX / 2 + 1 ? -uid[i] : uid[i];
        const int owner = normalized % g_world;
        if (owner == g_rank) {
            if (local != i) {
                uid[local] = uid[i]; norm[local] = norm[i]; score[local] = score[i];
                memmove(vec + (long)csd * local, vec + (long)csd * i, csd);
            }
            ++local;
        } else {
            // Workers never call MPI. The coordinator drains this queue even
            // while waiting for buckets, so bounded producer waits cannot
            // form a cycle with bucket completion. No candidates are dropped.
            if (g_remote_candidates.size() + stride > g_candidate_limit) {
                static std::atomic<bool> reported{false};
                if (!reported.exchange(true)) {
                    fprintf(stderr, "[MPI-CANDIDATE-BACKPRESSURE] rank=%d limit_bytes=%zu; waiting for coordinator drain\n",
                            g_rank, g_candidate_limit);
                    fflush(stderr);
                }
            }
            // Use the cache subsystem's polling idiom rather than the new
            // libstdc++ condition-variable wait entrypoint. GPU-node testing
            // reached a null indirect call immediately after entering that
            // path. Never hold the queue mutex while waiting for the drain.
            while (g_remote_candidates.size() + stride > g_candidate_limit) {
                lock.unlock();
                usleep(1000);
                lock.lock();
            }
            const std::size_t old = g_remote_candidates.size();
            g_remote_candidates.resize(old + stride);
            uint8_t *dst = g_remote_candidates.data() + old;
            memcpy(dst, uid + i, 8); memcpy(dst + 8, norm + i, 4);
            memcpy(dst + 12, score + i, 2); memcpy(dst + 14, vec + (long)csd * i, csd);
        }
    }
    return local;
}

int mpi_sieve_flush_candidates(int csd, swc_manager_t *swc,
                               ut_checker_t *ut_checker) {
    std::vector<uint8_t> outgoing;
    {
        std::lock_guard<std::mutex> lock(g_candidate_mutex);
        outgoing.swap(g_remote_candidates);
    }
    const uint64_t sequence = g_candidate_sequence.fetch_add(1);
    const size_t stride = csd + 14;
    const size_t window = std::max(stride, g_frame_bytes / stride * stride);
    const uint64_t local_rounds = (outgoing.size() + window - 1) / window;
    uint64_t rounds = 0;
    MPI_Allreduce(&local_rounds, &rounds, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
    for (uint64_t i = 0; i < rounds; ++i) {
        const size_t offset = std::min<size_t>(i * window, outgoing.size());
        const size_t bytes = std::min(window, outgoing.size() - offset);
        std::vector<uint8_t> incoming;
        if (exchange_bytes(bytes ? outgoing.data() + offset : nullptr, bytes, incoming,
                           tag_candidate_count, tag_candidate_data, csd,
                           (sequence << 32) | i) ||
                commit_candidates(incoming, csd, swc, ut_checker)) MPI_Abort(MPI_COMM_WORLD, 2);
    }
    return 0;
}

int mpi_sieve_drain_buckets(bwc_manager_t *bwc, swc_manager_t *swc,
                             ut_checker_t *ut, int csd, long limit) {
    const auto started = std::chrono::steady_clock::now();
    for (;;) {
        if (mpi_sieve_flush_candidates(csd, swc, ut)) return -1;
        // Both ranks execute the same collectives, including when just one
        // rank is waiting for a producer blocked by the candidate high-water.
        if (mpi_sieve_min_int(bwc->num_active() <= limit)) return 0;
        if (std::chrono::steady_clock::now() - started > std::chrono::hours(1)) {
            fprintf(stderr, "[MPI] bucket/candidate drain timed out\n");
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
        usleep(1000);
    }
}

int mpi_sieve_exchange_buckets(bwc_manager_t *bwc, swc_manager_t *swc,
                               ut_checker_t *ut_checker, const int32_t *ids,
                               int num_buckets, int csd) {
    if (mpi_sieve_flush_candidates(csd, swc, ut_checker)) return -1;
    const int stride = csd + (int)sizeof(int32_t);
    for (int i = 0; i < num_buckets; ++i) {
        const uint64_t sequence = g_bucket_sequence.fetch_add(1);
        const int owner = sequence % g_world;
        for (uint64_t part = 0;; ++part) {
            if (mpi_sieve_flush_candidates(csd, swc, ut_checker)) return -1;
            std::vector<uint8_t> outgoing, incoming;
            bool done = true;
            if (g_rank != owner && bwc->mpi_export_bucket_chunk(
                    ids[i], outgoing, stride, part == 0, &done)) MPI_Abort(MPI_COMM_WORLD, 2);
            if (mpi_sieve_min_int(done ? 1 : 0)) break;
            if (exchange_bytes(outgoing.data(), outgoing.size(), incoming,
                               tag_bucket_count, tag_bucket_data, csd,
                               (sequence << 32) | part)) MPI_Abort(MPI_COMM_WORLD, 2);
            if (g_rank == owner && (incoming.size() % stride ||
                bwc->mpi_append_bucket(ids[i], incoming.data(), incoming.size() / stride, stride)))
                MPI_Abort(MPI_COMM_WORLD, 2);
        }
        if (g_rank == owner) bwc->bucket_finalize(ids[i]);
    }
    return 0;
}

int mpi_sieve_global_stuck(uint64_t checked, uint64_t not_inserted, int csd) {
    uint64_t values[2] = {checked, not_inserted}, global[2] = {};
    MPI_Allreduce(values, global, 2, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    (void)csd;
    return global[1] * 36 < global[0] && global[0] > 100;
}

int mpi_sieve_partition_pool(Pool_hd_t *pool, long target_global_size) {
    const long target = target_global_size / g_world +
                        (g_rank < target_global_size % g_world);
    for (int attempt = 0; attempt < 6; ++attempt) {
        if (pool->mpi_retain_owner(g_rank, g_world)) return -1;
        long have = pool->pwc_manager->num_vec();
        if (have >= target) {
            if (have > target && pool->shrink(target)) return -1;
            break;
        }
        const long shortfall = target - have;
        if (pool->sampling(have + 2 * shortfall + 4096)) return -1;
    }
    const uint64_t total = mpi_sieve_global_u64(pool->pwc_manager->num_vec());
    if (total != (uint64_t)target_global_size) {
        if (!g_rank) fprintf(stderr, "[MPI] pool partition count mismatch: %llu != %ld\n",
                             (unsigned long long)total, target_global_size);
        return -1;
    }
    return 0;
}

int mpi_sieve_prune_pool(Pool_hd_t *pool, long target_global_size) {
    if (target_global_size < 0) return -1;
    const uint64_t local_count = pool->pwc_manager->num_vec();
    const uint64_t global_count = mpi_sieve_global_u64(local_count);
    if (global_count <= (uint64_t)target_global_size) return 0;

    // UID ownership is hash-balanced.  Give each rank a quota proportional to
    // the vectors it currently owns, then let the existing shrink operation
    // keep that rank's best-scoring vectors.  Do not derive quotas from
    // score_stat: redistribution can leave that diagnostic histogram stale.
    std::vector<uint64_t> counts(g_world), quotas(g_world), remainders(g_world);
    if (MPI_Allgather(&local_count, 1, MPI_UINT64_T, counts.data(), 1,
                      MPI_UINT64_T, MPI_COMM_WORLD) != MPI_SUCCESS)
        return -1;
    uint64_t assigned = 0;
    for (int rank = 0; rank < g_world; ++rank) {
        const unsigned __int128 product =
            (unsigned __int128)(uint64_t)target_global_size * counts[rank];
        quotas[rank] = (uint64_t)(product / global_count);
        remainders[rank] = (uint64_t)(product % global_count);
        assigned += quotas[rank];
    }
    while (assigned < (uint64_t)target_global_size) {
        int best = -1;
        for (int rank = 0; rank < g_world; ++rank) {
            if (quotas[rank] >= counts[rank]) continue;
            if (best < 0 || remainders[rank] > remainders[best]) best = rank;
        }
        if (best < 0) return -1;
        ++quotas[best];
        remainders[best] = 0;
        ++assigned;
    }
    const uint64_t local_target = quotas[g_rank];
    if (local_target > local_count || local_target > (uint64_t)LONG_MAX) return -1;
    if (local_target < local_count && pool->mpi_trim_exact((long)local_target))
        return -1;

    uint64_t retained = mpi_sieve_global_u64(pool->pwc_manager->num_vec());
    // shrink() chooses among equal-score vectors in parallel and can retain a
    // handful more than requested.  Remove that residual exactly from the
    // authoritative chunk contents before publishing the boundary.
    if (retained > (uint64_t)target_global_size) {
        uint64_t current = pool->pwc_manager->num_vec();
        std::vector<uint64_t> current_counts(g_world);
        if (MPI_Allgather(&current, 1, MPI_UINT64_T, current_counts.data(), 1,
                          MPI_UINT64_T, MPI_COMM_WORLD) != MPI_SUCCESS)
            return -1;
        uint64_t excess = retained - (uint64_t)target_global_size;
        std::vector<uint64_t> remove(g_world, 0);
        for (int rank = 0; rank < g_world && excess; ++rank) {
            remove[rank] = std::min(excess, current_counts[rank]);
            excess -= remove[rank];
        }
        if (remove[g_rank] &&
            pool->mpi_trim_exact((long)(current - remove[g_rank]))) return -1;
        retained = mpi_sieve_global_u64(pool->pwc_manager->num_vec());
    }
    if (retained != (uint64_t)target_global_size) {
        if (!g_rank)
            fprintf(stderr, "[MPI] global prune count mismatch: %llu != %ld\n",
                    (unsigned long long)retained, target_global_size);
        return -1;
    }
    if (!g_rank) {
        printf("[MPI] pruned CSD=%ld global_vectors=%llu quotas=%llu,%llu\n",
               pool->CSD, (unsigned long long)retained,
               (unsigned long long)quotas[0], (unsigned long long)quotas[1]);
        fflush(stdout);
    }
    return 0;
}

int mpi_sieve_redistribute_pool(Pool_hd_t *pool) {
    const int stride = pool->CSD + 14;
    const long original_chunks = pool->pwc_manager->num_chunks();
    pool->mpi_reset_append_hint();
    uint64_t max_chunks = 0, local_chunks = original_chunks;
    MPI_Allreduce(&local_chunks, &max_chunks, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
    for (uint64_t cid = 0; cid < max_chunks; ++cid) {
        std::vector<uint8_t> outgoing;
        if (cid < (uint64_t)original_chunks) {
            chunk_t *chunk = pool->pwc_manager->fetch(cid);
            if (chunk) {
                // Basis transforms and UID dedup leave sparse live slots.
                // size is a count, not the last occupied index: pack before
                // prefix iteration or empty slots replace live tail vectors.
                int live = 0;
                for (int i = 0; i < Pool_hd_t::chunk_max_nvecs; ++i)
                    live += chunk->score[i] != 0;
                if (live != chunk->size) {
                    fprintf(stderr, "[MPI] sparse chunk count mismatch id=%llu live=%d size=%d\n",
                            (unsigned long long)cid, live, chunk->size);
                    MPI_Abort(MPI_COMM_WORLD, 2);
                }
                _normalize_chunk(chunk, pool->CSD);
                const int old_size = chunk->size;
                int out = 0;
                for (int i = 0; i < old_size; ++i) {
                    const int owner = pool->uid_table->normalize(chunk->u[i]) % g_world;
                    if (owner != g_rank) {
                        const std::size_t old = outgoing.size();
                        outgoing.resize(old + stride);
                        uint8_t *dst = outgoing.data() + old;
                        memcpy(dst, chunk->u + i, 8); memcpy(dst + 8, chunk->norm + i, 4);
                        memcpy(dst + 12, chunk->score + i, 2);
                        memcpy(dst + 14, chunk->vec + (long)pool->CSD * i, pool->CSD);
                        pool->uid_table->erase(chunk->u[i]);
                        pool->score_stat[chunk->score[i]]--;
                    } else {
                        if (out != i) {
                            chunk->u[out] = chunk->u[i]; chunk->norm[out] = chunk->norm[i];
                            chunk->score[out] = chunk->score[i];
                            memmove(chunk->vec + (long)pool->CSD * out,
                                    chunk->vec + (long)pool->CSD * i, pool->CSD);
                        }
                        ++out;
                    }
                }
                chunk->size = out;
                if (out < Pool_hd_t::chunk_max_nvecs) {
                    memset(chunk->u + out, 0,
                           sizeof(uint64_t) * (Pool_hd_t::chunk_max_nvecs - out));
                    memset(chunk->norm + out, 0,
                           sizeof(int32_t) * (Pool_hd_t::chunk_max_nvecs - out));
                    memset(chunk->score + out, 0,
                           sizeof(uint16_t) * (Pool_hd_t::chunk_max_nvecs - out));
                }
                pool->pwc_manager->release_sync(cid);
            }
        }
        std::vector<uint8_t> incoming;
        if (exchange_bytes(outgoing.data(), outgoing.size(), incoming,
                           tag_pool_count, tag_pool_data, pool->CSD, cid)) return -1;
        if (incoming.size() % stride ||
            (!incoming.empty() &&
             pool->mpi_append_records(incoming.data(), incoming.size() / stride, stride)))
            return -1;
    }
    const uint64_t count = mpi_sieve_global_u64(pool->pwc_manager->num_vec());
    if (!g_rank) {
        printf("[MPI] redistributed CSD=%ld global_vectors=%llu\n", pool->CSD,
               (unsigned long long)count);
        fflush(stdout);
    }
    return 0;
}

int mpi_sieve_prepare_working_pool(Pool_hd_t *pool) {
    namespace fs = std::filesystem;
    const fs::path work = ".mpi-active-pool";
    std::error_code ec;
    fs::remove_all(work, ec);
    if (ec) return -1;
    fs::create_directories(work / "0", ec);
    if (ec) return -1;
    fs::create_directories(work / "1", ec);
    if (ec || pool->pwc_manager->set_dirname("mpi-active-pool")) return -1;
    // set_dirname changes the backing path but clean cache entries are not
    // automatically rewritten.  Materialize a complete mutable working copy.
    for (long cid = 0; cid < pool->pwc_manager->num_chunks(); ++cid) {
        chunk_t *chunk = pool->pwc_manager->fetch(cid);
        if (!chunk) return -1;
        pool->pwc_manager->release_sync(cid);
    }
    return pool->store(true);
}

int mpi_sieve_checkpoint(Pool_hd_t *pool) {
    namespace fs = std::filesystem;
    const std::string generation = ".checkpoint-" + std::to_string(pool->CSD);
    if (pool->store(true)) return -1;
    std::error_code ec;
    fs::remove_all(generation, ec);
    if (ec) return -1;
    fs::create_directories(generation, ec);
    if (ec) return -1;
    // Publish by renaming the fully flushed working tree.  The generation is
    // never used as the live cache, so later extend/sieve writes cannot mutate
    // an allegedly completed checkpoint.
    fs::rename(".mpi-active-pool", fs::path(generation) / "pool", ec);
    if (ec) return -1;
    const uint64_t local_count = pool->pwc_manager->num_vec();
    const uint64_t global_count = mpi_sieve_global_u64(local_count);
    std::string tmp = g_checkpoint_root + "/rank" + std::to_string(g_rank) + ".json.tmp";
    std::string dst = g_checkpoint_root + "/rank" + std::to_string(g_rank) + ".json";
    {
        std::ofstream out(tmp);
        out << "{\"version\":1,\"rank\":" << g_rank << ",\"world\":2,\"csd\":"
            << pool->CSD << ",\"index_l\":" << pool->index_l << ",\"index_r\":"
            << pool->index_r << ",\"basis_hash\":" << pool->basis_hash
            << ",\"vectors\":" << local_count << "}\n";
        if (!out) return -1;
    }
    if (rename(tmp.c_str(), dst.c_str())) return -1;
    MPI_Barrier(MPI_COMM_WORLD);

    // Publish the rank-local pool only after its data and manifest are durable.
    // LATEST remains unchanged until both ranks have reached this point.
    {
        ec.clear();
        fs::remove(".pool.next", ec);
        fs::create_directory_symlink(fs::path(generation) / "pool", ".pool.next", ec);
        if (ec) return -1;
        fs::remove_all(".pool", ec);
        fs::rename(".pool.next", ".pool", ec);
        if (ec) return -1;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (!g_rank) {
        tmp = g_checkpoint_root + "/LATEST.tmp";
        dst = g_checkpoint_root + "/LATEST";
        std::ofstream out(tmp);
        out << "version=1\nworld=2\ncsd=" << pool->CSD << "\nbasis_hash="
            << pool->basis_hash << "\nvectors=" << global_count << "\n";
        out.close();
        if (!out || rename(tmp.c_str(), dst.c_str())) return -1;
        printf("[MPI] checkpoint CSD=%ld global_vectors=%llu\n", pool->CSD,
               (unsigned long long)global_count);
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (const auto &entry : fs::directory_iterator(".")) {
        const std::string name = entry.path().filename().string();
        if (entry.is_directory() && name.rfind(".checkpoint-", 0) == 0 &&
            name != generation) {
            ec.clear();
            fs::remove_all(entry.path(), ec);
            if (ec) fprintf(stderr, "[MPI rank %d] warning: cannot prune %s: %s\n",
                            g_rank, name.c_str(), ec.message().c_str());
        }
    }
    // Recreate the live tree from the immutable generation at the filesystem
    // level.  This also covers chunks evicted from RAM: after the rename their
    // old manager path no longer exists, so fetching them to rewrite would be
    // unsafe.
    fs::remove_all(".mpi-active-pool", ec);
    if (ec) return -1;
    fs::copy(fs::path(generation) / "pool", ".mpi-active-pool",
             fs::copy_options::recursive, ec);
    if (ec) return -1;
    return pool->pwc_manager->set_dirname("mpi-active-pool");
}

void mpi_sieve_report_dimension(Pool_hd_t *pool, const char *phase) {
    const uint64_t total = mpi_sieve_global_u64(pool->pwc_manager->num_vec());
    if (!g_rank) {
        printf("[MPI] phase=%s CSD=%ld global_vectors=%llu\n", phase, pool->CSD,
               (unsigned long long)total);
        fflush(stdout);
    }
}
