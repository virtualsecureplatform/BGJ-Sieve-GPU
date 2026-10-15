// Included by bin_hd_sieve_mpi.cu after the serial task definitions.
#include <atomic>
#include <condition_variable>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <thread>
#include <fcntl.h>
#include <NTL/mat_ZZ.h>

namespace mpi_bkz {
namespace fs = std::filesystem;
fs::path root;
bool resume_enabled = true;
std::atomic<long> live_pump{-1}, live_csd{-1}, completed_phases{0};
std::atomic<int> live_phase{0};

void require(bool ok, const std::string &message) {
    if (!ok) throw std::runtime_error("[MPI-BKZ] " + message);
}
uint64_t hash_bytes(uint64_t h, const void *data, size_t n) {
    const auto *p = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}
uint64_t file_hash(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);
    require(bool(in), "cannot read " + path.string());
    char buffer[1 << 20]; uint64_t h = 1469598103934665603ULL;
    while (in.read(buffer, sizeof(buffer)) || in.gcount())
        h = hash_bytes(h, buffer, in.gcount());
    require(in.eof(), "read failure " + path.string());
    return h;
}
void durable(const fs::path &path, bool directory = false) {
    int fd = open(path.c_str(), O_RDONLY | (directory ? O_DIRECTORY : 0));
    require(fd >= 0, "open for fsync " + path.string());
    int rc = fsync(fd); close(fd);
    require(rc == 0, "fsync " + path.string());
}
void atomic_text(const fs::path &path, const std::string &text) {
    const fs::path tmp = path.string() + ".tmp";
    { std::ofstream out(tmp); out << text; out.close(); require(bool(out), "write manifest"); }
    durable(tmp); fs::rename(tmp, path); durable(path.parent_path(), true);
}
// Binary quad components preserve projected, non-integral bases exactly.
void save_basis(Lattice_QP &basis, const fs::path &path) {
    std::ofstream out(path, std::ios::binary);
    uint64_t shape[2] = {(uint64_t)basis.NumRows(), (uint64_t)basis.NumCols()};
    out.write((char *)shape, sizeof(shape));
    auto b = basis.get_b();
    for (long i = 0; i < basis.NumRows(); ++i) {
        out.write((char *)b.hi[i], shape[1] * sizeof(double));
        out.write((char *)b.lo[i], shape[1] * sizeof(double));
    }
    for (auto matrix : {basis.get_b_star(), basis.get_miu()}) {
        for (long i = 0; i < basis.NumRows(); ++i) {
            out.write((char *)matrix.hi[i], shape[1] * sizeof(double));
            out.write((char *)matrix.lo[i], shape[1] * sizeof(double));
        }
    }
    auto norms = basis.get_B();
    out.write((char *)norms.hi, shape[0] * sizeof(double));
    out.write((char *)norms.lo, shape[0] * sizeof(double));
    out.close(); require(bool(out), "basis snapshot write"); durable(path);
}
std::unique_ptr<Lattice_QP> load_basis(const fs::path &path) {
    std::ifstream in(path, std::ios::binary); uint64_t shape[2] = {};
    in.read((char *)shape, sizeof(shape));
    require(in && shape[0] > 0 && shape[0] <= 176 && shape[1] == shape[0], "invalid basis shape");
    auto basis = std::make_unique<Lattice_QP>(shape[0], shape[1]);
    auto b = basis->get_b();
    for (long i = 0; i < basis->NumRows(); ++i) {
        in.read((char *)b.hi[i], shape[1] * sizeof(double));
        in.read((char *)b.lo[i], shape[1] * sizeof(double));
    }
    for (auto matrix : {basis->get_b_star(), basis->get_miu()}) {
        for (long i = 0; i < basis->NumRows(); ++i) {
            in.read((char *)matrix.hi[i], shape[1] * sizeof(double));
            in.read((char *)matrix.lo[i], shape[1] * sizeof(double));
        }
    }
    auto norms = basis->get_B();
    in.read((char *)norms.hi, shape[0] * sizeof(double));
    in.read((char *)norms.lo, shape[0] * sizeof(double));
    require(bool(in) && in.peek() == EOF, "truncated or oversized basis snapshot");
    basis->set_gso_status(GSO_COMPUTED_QP);
    return basis;
}
struct state_t {
    uint64_t sequence = 0, identity = 0, basis_hash = 0, local_vectors = 0;
    long pump = 0, left = 0, right = 0, csd = 0, cursor = 0;
    // Next operation: 0 first sieve, 1 extend+sieve, 2 insert+sieve, 3 finish pump, 4 pump done.
    int next = 0;
    uint64_t coeff[176] = {};
};
std::string encode(const state_t &s) {
    std::ostringstream out;
    out << "BGJ_MPI_BKZ_V2 2 " << s.sequence << ' ' << s.identity << ' ' << s.basis_hash
        << ' ' << s.local_vectors << ' ' << s.pump << ' ' << s.left << ' ' << s.right
        << ' ' << s.csd << ' ' << s.cursor << ' ' << s.next << '\n';
    for (int i = 0; i < 176; ++i) out << s.coeff[i] << ' ';
    out << '\n'; return out.str();
}
state_t decode(const fs::path &path) {
    state_t s; std::string magic; int world = 0;
    std::ifstream in(path);
    in >> magic >> world >> s.sequence >> s.identity >> s.basis_hash >> s.local_vectors
       >> s.pump >> s.left >> s.right >> s.csd >> s.cursor >> s.next;
    for (auto &c : s.coeff) in >> c;
    require(in && magic == "BGJ_MPI_BKZ_V2" && world == 2 && s.next >= 0 && s.next <= 4,
            "invalid checkpoint state");
    require(s.next == 4 || (s.left >= 0 && s.right <= 176 && s.right-s.left == s.csd &&
            s.csd >= 1 && s.csd <= Pool_hd_t::host_vec_nbytes), "invalid checkpoint context");
    return s;
}
fs::path generation(uint64_t seq, int rank) {
    return root / ("generation-" + std::to_string(seq)) / ("rank" + std::to_string(rank));
}
// Each snapshot is immutable; verify every file before loading a checkpoint.
void inventory(const fs::path &dir, bool verify) {
    if (verify) {
        std::ifstream in(dir / "FILES"); std::string name; uint64_t size, hash; size_t count = 0;
        while (in >> std::quoted(name) >> size >> hash) {
            fs::path relative(name);
            require(!relative.is_absolute() && name.find("..") == std::string::npos,
                    "unsafe snapshot path");
            require(fs::is_regular_file(dir/relative) && fs::file_size(dir/relative) == size &&
                    file_hash(dir/relative) == hash, "snapshot checksum mismatch: " + name);
            ++count;
        }
        require(in.eof() && count >= 2, "incomplete snapshot inventory");
    } else {
        std::ostringstream out;
        for (const auto &e : fs::recursive_directory_iterator(dir)) {
            if (!e.is_regular_file() || e.path().filename() == "FILES") continue;
            durable(e.path());
            out << std::quoted(fs::relative(e.path(), dir).string()) << ' '
                << e.file_size() << ' ' << file_hash(e.path()) << '\n';
        }
        atomic_text(dir/"FILES", out.str());
        for (const auto &e : fs::recursive_directory_iterator(dir))
            if (e.is_directory()) durable(e.path(), true);
        durable(dir, true);
    }
}
void commit(state_t &s, Lattice_QP &full, Lattice_QP *local, Pool_hd_t *pool) {
    const int rank = mpi_sieve_rank();
    live_phase.store(100); // Explicit checkpoint I/O, not a stalled GPU sieve.
    mpi_sieve_agree_basis(&full, "checkpoint-full");
    if (local) mpi_sieve_agree_basis(local, "checkpoint-local");
    if (pool) {
        pool->pwc_manager->wait_work();
        require(pool->store(true) == 0, "pool flush");
        s.left = pool->index_l; s.right = pool->index_r; s.csd = pool->CSD;
        s.basis_hash = pool->basis_hash; s.local_vectors = pool->pwc_manager->num_vec();
        for (long i = 0; i < pool->CSD; ++i) s.coeff[i] = pool->uid_table->coeff(i);
        uint64_t bytes = 0;
        for (const auto &file : fs::recursive_directory_iterator(".pool"))
            if (file.is_regular_file()) bytes += file.file_size();
        const uint64_t needed = mpi_sieve_global_u64(bytes);
        uint64_t available = fs::space(root).available, minimum = 0;
        MPI_Allreduce(&available, &minimum, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
        require(minimum > needed + needed/10 + (1ULL<<30), "insufficient free filesystem space for both snapshots");
    }
    const uint64_t previous = s.sequence;
    if (!rank) {
        do { ++s.sequence; } while (fs::exists(generation(s.sequence, 0).parent_path()));
    }
    MPI_Bcast(&s.sequence, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    const fs::path dst = generation(s.sequence, rank);
    require(!fs::exists(dst), "refusing to overwrite a checkpoint generation");
    fs::create_directories(dst);
    save_basis(full, dst/"full.qp");
    if (local) save_basis(*local, dst/"local.qp");
    if (pool) fs::copy(".pool", dst/"pool", fs::copy_options::recursive);
    atomic_text(dst/"STATE", encode(s));
    inventory(dst, false);
    const char *fault = getenv("HD_MPI_BKZ_FAULT");
    if (fault && s.sequence > 1 && rank == 1 && !strcmp(fault, "after-rank-snapshot"))
        MPI_Abort(MPI_COMM_WORLD, 86);
    MPI_Barrier(MPI_COMM_WORLD);
    if (!rank) {
        if (fault && s.sequence > 1 && !strcmp(fault, "before-publish")) MPI_Abort(MPI_COMM_WORLD, 86);
        // Both ranks must agree on the shared execution cursor before publication.
        state_t peer = decode(generation(s.sequence, 1)/"STATE");
        require(peer.identity == s.identity && peer.pump == s.pump && peer.next == s.next &&
                peer.csd == s.csd && peer.cursor == s.cursor && peer.basis_hash == s.basis_hash,
                "rank checkpoint state disagreement");
        durable(dst.parent_path(), true);
        atomic_text(root/"LATEST-BKZ", std::to_string(s.sequence) + "\n");
        printf("[MPI-BKZ] committed generation=%llu pump=%ld csd=%ld next=%d\n",
               (unsigned long long)s.sequence, s.pump, s.csd, s.next); fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    // Keep current plus previous generation. Only remove older, run-local trees.
    if (!rank) for (const auto &entry : fs::directory_iterator(root)) {
        const std::string name = entry.path().filename().string();
        if (entry.is_directory() && name.rfind("generation-", 0) == 0) {
            const std::string suffix = name.substr(11);
            if (!suffix.empty() && suffix.find_first_not_of("0123456789") == std::string::npos) {
                const uint64_t seq = std::stoull(suffix);
                if (seq < s.sequence && seq != previous && fs::exists(entry.path()/"rank0/STATE")) {
                    try {
                        if (decode(entry.path()/"rank0/STATE").identity == s.identity) fs::remove_all(entry.path());
                    } catch (const std::exception &) {
                        fprintf(stderr, "[MPI-BKZ] keeping unrecognized generation %s\n", name.c_str());
                    }
                }
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}
uint64_t identity(task_config_t &t) {
    std::ostringstream out;
    out << file_hash(t.basis_file) << ' ' << t.task << ' ' << t.bkz_sieving_dim << ' '
        << t.bkz_dim_for_free << ' ' << t.bkz_jump_step << ' ' << t.bkz_start_index << ' '
        << t.bkz_stop_index << ' ' << t.ind_l << ' ' << t.ind_r << ' '
        << t.start_sieving_dim << ' ' << t.max_sieving_dim << ' ' << t.enable_down_sieve
        << ' ' << Pool_hd_t::host_vec_nbytes << ' ' << PWC_DRAM_SLIMIT << ' '
        << BWC_DRAM_SLIMIT << ' ' << SWC_DRAM_SLIMIT;
    for (const char *key : {"HD_SOURCE_COMMIT", "HD_SIZE_RATIO", "HD_BGJ2_MIN_CSD",
            "HD_BGJ3_MIN_CSD", "HD_USE_BGJ3L", "HD_FAIL_ON_SPILL", "HD_POOL_THREADS",
            "HD_BGJ2_REDUCER_THREADS", "HD_BUCKET_THREADS"})
        out << ' ' << key << '=' << (getenv(key) ? getenv(key) : "");
    auto text = out.str(); return hash_bytes(1469598103934665603ULL, text.data(), text.size());
}
NTL::mat_ZZ integers(Lattice_QP &basis) {
    NTL::mat_ZZ out; out.SetDims(basis.NumRows(), basis.NumCols());
    auto b = basis.get_b();
    for (long i = 0; i < basis.NumRows(); ++i) for (long j = 0; j < basis.NumCols(); ++j) {
        NTL::quad_float q(b.hi[i][j], b.lo[i][j]);
        NTL::conv(out[i][j], q);
        require(std::fabs((q - NTL::to_quad_float(out[i][j])).hi) < 1e-8, "non-integral full basis");
    }
    return out;
}
std::unique_ptr<Lattice_QP> clone_basis(Lattice_QP &basis) {
    auto copy = std::make_unique<Lattice_QP>(basis.NumRows(), basis.NumCols());
    for (long i = 0; i < basis.NumRows(); ++i) {
        memcpy(copy->get_b().hi[i], basis.get_b().hi[i], basis.NumCols()*sizeof(double));
        memcpy(copy->get_b().lo[i], basis.get_b().lo[i], basis.NumCols()*sizeof(double));
    }
    copy->compute_gso_QP(); return copy;
}
void exact_equivalent(const NTL::mat_ZZ &before, Lattice_QP &after) {
    if (mpi_sieve_rank()) return;
    NTL::mat_ZZ adj, product; NTL::ZZ det;
    NTL::inv(det, adj, before);
    require(!NTL::IsZero(det), "singular input lattice");
    NTL::mul(product, integers(after), adj);
    for (long i = 0; i < product.NumRows(); ++i) for (long j = 0; j < product.NumCols(); ++j) {
        require(NTL::IsZero(product[i][j] % det), "output not in input lattice");
        product[i][j] /= det;
    }
    NTL::ZZ transform_det; NTL::determinant(transform_det, product);
    require(abs(transform_det) == 1, "non-unimodular pump transform");
    printf("[MPI-BKZ] exact lattice-equivalence check passed\n"); fflush(stdout);
}
struct heartbeat_t {
    std::mutex mutex; std::condition_variable cv; bool stop = false; std::thread worker;
    heartbeat_t() : worker([this] {
        std::unique_lock<std::mutex> lock(mutex);
        while (!cv.wait_for(lock, std::chrono::seconds(60), [this] { return stop; })) {
            const auto metrics = mpi_sieve_metrics();
            printf("[MPI-BKZ heartbeat] rank=%d epoch=%ld pump=%ld csd=%ld next=%d completed_phases=%ld "
                   "buckets=%llu candidate_exchanges=%llu pending_candidate_bytes=%llu sent_bytes=%llu received_bytes=%llu\n",
                   mpi_sieve_rank(), (long)time(nullptr), live_pump.load(), live_csd.load(),
                   live_phase.load(), completed_phases.load(),
                   (unsigned long long)metrics.buckets, (unsigned long long)metrics.candidate_exchanges,
                   (unsigned long long)metrics.pending_candidate_bytes,
                   (unsigned long long)metrics.sent_bytes, (unsigned long long)metrics.received_bytes);
            report_host_memory("mpi-bkz-heartbeat", live_csd.load()); fflush(stdout);
            std::ifstream io("/proc/self/io"); std::string key; uint64_t value;
            while (io >> key >> value)
                if (key == "read_bytes:" || key == "write_bytes:")
                    printf("[MPI-BKZ io] rank=%d %s %llu\n", mpi_sieve_rank(),
                           key.c_str(), (unsigned long long)value);
            fflush(stdout);
        }
    }) {}
    ~heartbeat_t() { { std::lock_guard<std::mutex> lock(mutex); stop = true; } cv.notify_all(); worker.join(); }
};
bool pause(const state_t &s) {
    const char *limit = getenv("HD_MPI_BKZ_MAX_PHASES");
    const char *after = getenv("HD_MPI_BKZ_STOP_AFTER");
    bool requested = (limit && completed_phases.load() >= atol(limit)) ||
        (after && !strcmp(after, "top") && s.next == 2) ||
        (after && !strcmp(after, "first-down") && s.next == 2 && s.cursor > 0);
    return mpi_sieve_global_u64(requested ? 1 : 0) || mpi_sieve_should_stop();
}
void compact(Pool_hd_t &pool) {
    const long before = pool.pwc_manager->num_vec();
    if (before) require(pool.shrink(before) == 0, "lossless pool compaction");
    require(pool.pwc_manager->num_vec() == before, "compaction changed the vector count");
    require(pool.pwc_manager->num_chunks() <= (before + Pool_hd_t::chunk_max_nvecs-1)/Pool_hd_t::chunk_max_nvecs,
            "pool compaction left chunk holes");
}
void redistribute(Pool_hd_t &pool) {
    pool.pwc_manager->wait_work();
    require(mpi_sieve_agree_basis(pool.basis, "basis-transition") == 0, "basis agreement");
    require(mpi_sieve_redistribute_pool(&pool) == 0, "pool redistribution");
    require(pool.mpi_retain_owner(mpi_sieve_rank(), 2) == 0, "score/ownership rebuild");
    compact(pool);
}
} // namespace mpi_bkz

int mpi_bkz_run(void *opaque) {
    using namespace mpi_bkz;
    auto &t = *static_cast<task_config_t *>(opaque);
    require(!fs::exists(root/"LATEST"), "cannot mix version1 final-sieve and BKZ checkpoints");
    const bool bkz = t.task == task_config_t::task_bkz;
    require((bkz ? t.bkz_dual_hash_ratio == 0 : !t.enable_dual_hash),
            "distributed dual-hash insertion is unsupported; use --BDH 0");
    const long max_dim = bkz ? t.bkz_sieving_dim : t.max_sieving_dim;
    require(max_dim >= 40 && max_dim <= 144 && max_dim <= Pool_hd_t::host_vec_nbytes,
            "MPI BKZ v1 supports sieve dimensions 40..144 within the build limit");
    if (bkz) {
        require(t.bkz_jump_step > 0 && t.bkz_dim_for_free >= 0 && t.bkz_start_index >= 0,
                "invalid BKZ jump/D4F/start index");
    }
    if (getenv("HD_SIZE_RATIO")) require(std::fabs(atof(getenv("HD_SIZE_RATIO"))-3.2) < 1e-12,
            "MPI BKZ v1 requires the full size ratio 3.2");
    state_t s; s.identity = identity(t);
    uint64_t ids[2]; MPI_Allgather(&s.identity, 1, MPI_UINT64_T, ids, 1, MPI_UINT64_T, MPI_COMM_WORLD);
    require(ids[0] == ids[1], "rank configuration disagreement");
    if (!mpi_sieve_rank()) {
        if (fs::exists(root/"IDENTITY-BKZ")) {
            uint64_t id = 0; std::ifstream in(root/"IDENTITY-BKZ"); in >> id;
            require(in && id == s.identity, "checkpoint root belongs to a different run/configuration");
        } else {
            require(!fs::exists(root/"LATEST-BKZ"), "checkpoint root lacks its identity manifest");
            atomic_text(root/"IDENTITY-BKZ", std::to_string(s.identity)+"\n");
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto full = std::make_unique<Lattice_QP>(t.basis_file);
    require(full->NumRows() > max_dim && full->NumRows() <= 176, "invalid lattice dimension");
    if (bkz) require(t.bkz_start_index < full->NumRows() &&
        (t.bkz_stop_index == task_config_t::val_not_assigned ||
         (t.bkz_stop_index > t.bkz_start_index && t.bkz_stop_index <= full->NumRows())),
        "invalid BKZ stop index");
    bool restored = resume_enabled && fs::exists(root/"LATEST-BKZ");
    fs::path restored_dir;
    if (restored) {
        uint64_t seq = 0; std::ifstream latest(root/"LATEST-BKZ"); latest >> seq;
        require(latest && seq > 0, "invalid latest generation");
        restored_dir = generation(seq, mpi_sieve_rank()); inventory(restored_dir, true);
        state_t saved = decode(restored_dir/"STATE");
        require(saved.identity == s.identity && saved.sequence == seq, "incompatible checkpoint");
        s = saved; full = load_basis(restored_dir/"full.qp");
    } else {
        require(!fs::exists(root/"LATEST-BKZ"), "--resume none requires a fresh checkpoint root");
        s.pump = bkz ? t.bkz_start_index : (t.ind_l < 0 ? 0 : t.ind_l);
    }
    heartbeat_t heartbeat;
    for (;;) {
        const long begin = s.pump;
        const long end = bkz ? std::min(full->NumRows(), begin + max_dim + t.bkz_dim_for_free) :
            (t.ind_r < 0 ? full->NumRows() : t.ind_r);
        require(begin >= 0 && begin < end && end <= full->NumRows(), "invalid pump window");
        bool last = !bkz || begin + max_dim + 2 >= full->NumRows() ||
            (t.bkz_stop_index != task_config_t::val_not_assigned &&
             begin + t.bkz_jump_step >= t.bkz_stop_index);
        if (restored && s.next == 4) {
            if (last) break;
            s.pump += t.bkz_jump_step; restored = false; s.next = 0; s.cursor = 0;
            continue;
        }
        const auto original = integers(*full);
        auto local = restored ? load_basis(restored_dir/"local.qp") :
            std::unique_ptr<Lattice_QP>(full->b_loc_QP(begin, end));
        const long top = std::min(max_dim, local->NumRows()-1);
        const long start = std::min(top, bkz ? task_config_t::default_bkz_sieve_ssd : t.start_sieving_dim);
        require(top >= 40 && start >= 40, "invalid local sieve dimension");
        // Scratch is mutable and rank-local. Snapshots are never cache backing files.
        fs::remove_all(".pool"); fs::create_directories(".pool/0"); fs::create_directories(".pool/1");
        if (restored) fs::copy(restored_dir/"pool", ".pool", fs::copy_options::recursive | fs::copy_options::overwrite_existing);
        {
            Pool_hd_t pool(local.get());
            require(pool.set_sieving_context(restored ? s.left : local->NumRows()-start,
                                            local->NumRows()) == 0, "set context");
            pool.set_num_threads(configured_pool_threads()); pool.set_boost_depth(0);
            if (restored) {
                pool.basis_hash = s.basis_hash; pool.pwc_manager->set_pool(&pool);
                require(pool.uid_table->reset_hash_function(pool.CSD, s.coeff) == 0, "restore UID coefficients");
                require(pool.load(0) == 0 && pool.pwc_manager->num_vec() == (long)s.local_vectors,
                        "pool recovery count mismatch");
                require(pool.check(0) == 0, "recovered pool check");
            } else {
                s.cursor = 0; s.next = 0;
                const long target = BGJ1_SIZE_RATIO * pow(4./3., pool.CSD*.5) - 5;
                require(pool.sampling(target/2 + 4096) == 0 &&
                        mpi_sieve_partition_pool(&pool, target) == 0, "initial sampling");
                // Skip orphaned generations left by an interrupted publication.
                commit(s, *full, local.get(), &pool);
            }
            restored = false;
            for (;;) {
                live_pump.store(begin); live_csd.store(pool.CSD); live_phase.store(s.next);
                if (s.next == 3) break;
                if (mpi_sieve_should_stop()) return 75;
                if (s.next == 1) { require(pool.extend_left() == 0, "extend left"); redistribute(pool); }
                else if (s.next == 2) {
                    if (pool.CSD < 60) { s.next = 3; break; }
                    pool.down_sieve_flag = 1;
                    long pos = -1; int ret = pool.insert(s.cursor++, 1.2, &pos);
                    require(ret == 0 || ret == -1, "insertion failed");
                    redistribute(pool);
                    // Validate an insertion before publishing its phase checkpoint.
                    // trans_to consumes the projected basis, so use private copies.
                    if (!mpi_sieve_rank()) {
                        auto probe = clone_basis(*full), projection = clone_basis(*local);
                        require(probe->trans_to(begin, end, projection.get()) == 1, "insertion reconstruction");
                        exact_equivalent(original, *probe);
                    }
                    MPI_Barrier(MPI_COMM_WORLD);
                    if (s.cursor <= pos) s.cursor = pos+1;
                    if (s.cursor >= pool.index_l) s.cursor = pool.index_l-1;
                    const long target = BGJ4_SIZE_RATIO * pow(4./3., pool.CSD*.5) - 5;
                    require(mpi_sieve_prune_pool(&pool, target) == 0, "global prune");
                    compact(pool);
                    if (mpi_sieve_global_u64(pool.pwc_manager->num_vec()) < 0.7*target) { s.next = 3; break; }
                    if (!bkz && !t.enable_down_sieve) {
                        if (s.cursor > 10) { s.next = 3; break; }
                        continue;
                    }
                }
                live_csd.store(pool.CSD);
                srand((unsigned)(s.identity ^ (s.sequence+1) ^ mpi_sieve_rank()));
                int ret = _sieve(&pool);
                if (mpi_sieve_should_stop()) return 75; // Never publish a stopped, partial sieve.
                if ((ret == -1 && pool.CSD < 80) || pool.CSD == 80)
                    require(pool.check_dim_lose() == 0, "dimension loss");
                s.next = pool.down_sieve_flag ? (pool.CSD < 60 ? 3 : 2) : (pool.CSD < top ? 1 : 2);
                commit(s, *full, local.get(), &pool); ++completed_phases;
                if (pause(s)) return 75;
            }
            if (local->NumRows() < 90) local->LLL_DEEP_QP(0.99);
            require(full->trans_to(begin, end, local.get()) == 1, "full basis update");
            full->compute_gso_QP(); full->size_reduce(); full->compute_gso_QP();
            mpi_sieve_agree_basis(full.get(), "pump-complete");
            exact_equivalent(original, *full);
            MPI_Barrier(MPI_COMM_WORLD);
        }
        s.next = 4; commit(s, *full, nullptr, nullptr);
        if (last) break;
        s.pump += t.bkz_jump_step; s.next = 0; s.cursor = 0;
    }
    if (!mpi_sieve_rank()) {
        fs::path output(t.output_file); fs::path tmp = output.string()+".tmp";
        full->store(tmp.c_str()); durable(tmp); fs::rename(tmp, output);
        durable(output.parent_path().empty() ? fs::path(".") : output.parent_path(), true);
        full->show_dist_vec();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return 0;
}
