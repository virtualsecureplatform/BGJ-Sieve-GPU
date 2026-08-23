#include "../include/mpi_sieve.h"
#include "../include/bgj_hd.h"

#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
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
std::vector<uint8_t> g_remote_candidates;
std::atomic<uint64_t> g_bucket_sequence{0};
std::atomic<uint64_t> g_candidate_sequence{0};
std::atomic<bool> g_stop_requested{false};
int g_resumed_csd = -1;
uint64_t g_sent_bytes = 0;
uint64_t g_recv_bytes = 0;
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
    if (dst) ut->task_commit(dst);
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
                           bool resume) {
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
    if (resume) {
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
    std::lock_guard<std::mutex> lock(g_candidate_mutex);
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
    std::vector<uint8_t> incoming;
    const uint64_t sequence = g_candidate_sequence.fetch_add(1);
    if (exchange_bytes(outgoing.data(), outgoing.size(), incoming,
                       tag_candidate_count, tag_candidate_data, csd, sequence)) return -1;
    return commit_candidates(incoming, csd, swc, ut_checker);
}

int mpi_sieve_exchange_buckets(bwc_manager_t *bwc, swc_manager_t *swc,
                               ut_checker_t *ut_checker, const int32_t *ids,
                               int num_buckets, int csd) {
    if (mpi_sieve_flush_candidates(csd, swc, ut_checker)) return -1;
    const int stride = csd + (int)sizeof(int32_t);
    for (int i = 0; i < num_buckets; ++i) {
        const uint64_t sequence = g_bucket_sequence.fetch_add(1);
        const int owner = sequence % g_world;
        std::vector<uint8_t> outgoing;
        if (g_rank != owner && bwc->mpi_export_bucket(ids[i], outgoing, stride)) return -1;
        std::vector<uint8_t> incoming;
        if (exchange_bytes(outgoing.data(), outgoing.size(), incoming,
                           tag_bucket_count, tag_bucket_data, csd, sequence)) return -1;
        if (g_rank == owner) {
            if (incoming.size() % stride) return -1;
            if (bwc->mpi_append_bucket(ids[i], incoming.data(), incoming.size() / stride,
                                       stride)) return -1;
            bwc->bucket_finalize(ids[i]);
        }
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
