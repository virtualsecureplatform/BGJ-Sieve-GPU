// Included in mpi_sieve_runtime.cpp's anonymous namespace. Only run() calls
// MPI. The worker pools own no MPI requests, and never touch a live TX buffer.
struct pipeline_callbacks_t {
    void *context = nullptr;
    void (*ready)(void *) = nullptr;
    bool (*service)(void *, bool) = nullptr;
} g_pipeline_callbacks;

struct candidate_queue_t {
    std::mutex mutex;
    std::deque<std::vector<uint8_t>> batches;
};
std::mutex g_producer_mutex;
std::map<int, std::unique_ptr<candidate_queue_t>> g_producers;
std::atomic<uint64_t> g_pipeline_pending{0}, g_pipeline_highwater{0};
std::atomic<uint64_t> g_ready_batches{0};
std::atomic<uint64_t> g_candidate_wait_ns{0}, g_swc_wait_ns{0}, g_uid_wait_ns{0}, g_crc_ns{0};
std::atomic<uint64_t> g_bucket_ready_ns[bwc_manager_t::bwc_max_buckets];
std::atomic<uint64_t> g_bucket_wait_ns{0}, g_bucket_started{0};
uint64_t pipeline_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

uint64_t normalized_uid(uint64_t uid) {
    return uid > UINT64_MAX / 2 + 1 ? -uid : uid;
}

candidate_queue_t *producer_queue(int producer) {
    std::lock_guard<std::mutex> lock(g_producer_mutex);
    auto &queue = g_producers[producer];
    if (!queue) queue.reset(new candidate_queue_t);
    return queue.get();
}

// Charge records until the peer has applied them, not just until they leave
// the producer queue. This also bounds work held by packing/TX workers.
void reserve_candidate_bytes(size_t bytes) {
    static std::atomic<bool> reported{false};
    const auto start = pipeline_now_ns();
    for (;;) {
        uint64_t pending = g_pipeline_pending.load(std::memory_order_relaxed);
        if (pending + bytes <= g_candidate_limit &&
            g_pipeline_pending.compare_exchange_weak(pending, pending + bytes)) {
            uint64_t high = g_pipeline_highwater.load();
            while (high < pending + bytes &&
                   !g_pipeline_highwater.compare_exchange_weak(high, pending + bytes)) {}
            g_candidate_wait_ns += pipeline_now_ns()-start;
            return;
        }
        if (!reported.exchange(true)) {
            fprintf(stderr, "[MPI-CANDIDATE-BACKPRESSURE] rank=%d limit_bytes=%zu; async credit wait\n",
                    g_rank, g_candidate_limit);
        }
        usleep(1000);
    }
}

int pipeline_filter(int csd, int8_t *vec, int32_t *norm, uint16_t *score,
                    uint64_t *uid, int size, int producer) {
    candidate_queue_t *queue = producer_queue(producer);
    const size_t stride = csd + 14;
    const size_t capacity = std::min<size_t>(std::min<size_t>(64ULL << 10, g_candidate_limit),
                                            std::min<size_t>(16ULL<<20, g_frame_bytes/16)-128);
    const size_t max_count = (capacity - 4) / stride;
    if (!max_count) return -1;
    // Publish remote records before compacting local records: compaction can
    // otherwise overwrite an earlier remote record still waiting to be packed.
    for (int begin = 0; begin < size;) {
        std::vector<int> indices;
        indices.reserve(max_count);
        while (begin < size && indices.size() < max_count) {
            if (normalized_uid(uid[begin]) % g_world != uint64_t(g_rank))
                indices.push_back(begin);
            ++begin;
        }
        if (indices.empty()) continue;
        const uint32_t count = indices.size();
        const size_t bytes = 4 + count * stride;
        reserve_candidate_bytes(bytes);
        std::vector<uint8_t> batch(bytes);
        memcpy(batch.data(), &count, 4);
        uint8_t *uids = batch.data() + 4;
        uint8_t *norms = uids + count * 8;
        uint8_t *scores = norms + count * 4;
        uint8_t *vectors = scores + count * 2;
        for (uint32_t j = 0; j < count; ++j) {
            const int i = indices[j];
            memcpy(uids + j * 8, uid + i, 8);
            memcpy(norms + j * 4, norm + i, 4);
            memcpy(scores + j * 2, score + i, 2);
            memcpy(vectors + j * csd, vec + (long)i * csd, csd);
        }
        std::lock_guard<std::mutex> lock(queue->mutex);
        queue->batches.emplace_back(std::move(batch));
        g_ready_batches.fetch_add(1, std::memory_order_release);
    }
    int local = 0;
    for (int i = 0; i < size; ++i) {
        if (normalized_uid(uid[i]) % g_world != uint64_t(g_rank)) continue;
        if (local != i) {
            uid[local] = uid[i]; norm[local] = norm[i]; score[local] = score[i];
            memmove(vec + (long)local * csd, vec + (long)i * csd, csd);
        }
        ++local;
    }
    return local;
}

struct pipeline_header_t {
    uint32_t magic = wire_magic;
    uint16_t version = 3, kind = 0;
    uint32_t csd = 0, final = 0;
    uint64_t wave = 0, bucket = 0, part = 0, bytes = 0, checksum = 0;
};
struct pipeline_done_t {
    uint64_t wave = 0, bucket_frames = 0, candidate_frames = 0;
};

struct pipeline_buffer_t {
    uint8_t *data = nullptr;
    size_t capacity = 0;
    ~pipeline_buffer_t() { if (data) cudaFreeHost(data); }
} g_pipeline_buffers[2][2][2];

void pipeline_release_buffers() {
    for (auto &channel : g_pipeline_buffers) for (auto &lane : channel)
        for (auto &buffer : lane) {
            if (buffer.data) cudaFreeHost(buffer.data);
            buffer.data = nullptr; buffer.capacity = 0;
        }
}

struct pipeline_slot_t {
    uint8_t *data = nullptr;
    size_t capacity = 0, charged = 0;
    // 0: idle, 1: worker owns buffer, 2: MPI owns buffer, 3: waiting for ACK.
    int state = 0;
    std::atomic<bool> worked{false};
    MPI_Request data_request = MPI_REQUEST_NULL, ack_request = MPI_REQUEST_NULL;
    uint64_t ack = 0;
    pipeline_header_t header;
    int bucket_index = -1;
    uint64_t part = 0;
    bwc_manager_t::mpi_export_cursor_t cursor;
};

struct pipeline_t {
    bwc_manager_t *bwc;
    swc_manager_t *swc;
    ut_checker_t *ut;
    std::vector<int32_t> ids;
    uint64_t base;
    int csd, next_export = 0, owned = 0;
    pipeline_slot_t tx[2][2], rx[2][2];
    thread_pool::thread_pool transfer_workers{2}, apply_worker{1};
    std::atomic<int> error{0}, finalized{0};
    std::atomic<uint64_t> pack_ns{0}, apply_ns{0}, receive_ns{0};
    std::vector<uint64_t> expected_part;
    std::vector<uint8_t> bucket_done;
    std::vector<int> bucket_lane;
    uint64_t sent[2] = {}, applied[2] = {};
    pipeline_done_t local_done, peer_done;
    MPI_Request done_send = MPI_REQUEST_NULL, done_recv = MPI_REQUEST_NULL;
    bool published_done = false, received_done = false;
    unsigned test_delay_us = 0;
    std::atomic<bool> test_corrupt{false};
    uint64_t swc0 = g_swc_wait_ns.load(), uid0 = g_uid_wait_ns.load(), crc0 = g_crc_ns.load();
    uint64_t bucket_wait0 = g_bucket_wait_ns.load(), bucket_start0 = g_bucket_started.load();

    static constexpr int data_tag(int channel, int slot) { return 5100 + channel*8 + slot; }
    static constexpr int ack_tag(int channel, int slot) { return 5104 + channel*8 + slot; }

    pipeline_t(bwc_manager_t *b, swc_manager_t *s, ut_checker_t *u,
               const int32_t *bids, int n, int dim)
        : bwc(b), swc(s), ut(u), ids(bids, bids+n),
          base(g_bucket_sequence.fetch_add(n)), csd(dim), expected_part(n, 0),
          bucket_done(n, false), bucket_lane(n, -1) {
        // Frame limits scale with --mpi-frame-mib. Default allocation: 320 MiB.
        const size_t limits[2] = {std::min<size_t>(64ULL<<20, g_frame_bytes/4),
                                  std::min<size_t>(16ULL<<20, g_frame_bytes/16)};
        if (const char *delay = getenv("HD_MPI_PIPELINE_TEST_DELAY_US")) {
            char *end = nullptr; const long value = strtol(delay, &end, 10);
            if (!*delay || *end || value < 0 || value > 100000)
                throw std::runtime_error("invalid pipeline test delay");
            test_delay_us = value;
        }
        test_corrupt.store(getenv("HD_MPI_PIPELINE_TEST_CORRUPT") != nullptr);
        for (int i = 0; i < n; ++i) if ((base+i)%g_world == uint64_t(g_rank)) ++owned;
        for (int c = 0; c < 2; ++c) for (int k = 0; k < 2; ++k) for (int direction = 0; direction < 2; ++direction) {
                auto &buffer = g_pipeline_buffers[c][k][direction];
                auto *slot = direction ? &rx[c][k] : &tx[c][k];
                slot->capacity = limits[c];
                if (!buffer.data) {
                    if (cudaHostAlloc((void **)&buffer.data, slot->capacity,
                                      cudaHostAllocPortable) != cudaSuccess) throw std::bad_alloc();
                    buffer.capacity = slot->capacity;
                }
                if (buffer.capacity != slot->capacity) throw std::runtime_error("pipeline buffer size changed");
                slot->data = buffer.data;
            }
    }
    ~pipeline_t() {
        transfer_workers.wait_sleep(); apply_worker.wait_sleep();
    }
    template<class F> void work(pipeline_slot_t &slot, bool candidate, F function) {
        slot.worked.store(false, std::memory_order_relaxed);
        slot.state = 1;
        (candidate ? apply_worker : transfer_workers).push([this, &slot, function] {
            try { function(); }
            catch (const std::exception &e) {
                fprintf(stderr, "[MPI-PIPELINE] rank=%d worker: %s\n", g_rank, e.what());
                error.store(1);
            } catch (...) { error.store(1); }
            slot.worked.store(true, std::memory_order_release);
        });
    }
    static bool complete(MPI_Request &request) {
        if (request == MPI_REQUEST_NULL) return true;
        int flag = 0;
        if (MPI_Test(&request, &flag, MPI_STATUS_IGNORE) != MPI_SUCCESS)
            throw std::runtime_error("MPI_Test failed");
        return flag;
    }
    static void check(int rc) {
        if (rc != MPI_SUCCESS) throw std::runtime_error("MPI pipeline operation failed");
    }
    void post_receive(int c, int k) {
        auto &slot = rx[c][k];
        check(MPI_Irecv(slot.data, slot.capacity, MPI_BYTE, 1-g_rank,
                        data_tag(c,k), MPI_COMM_WORLD, &slot.data_request));
        slot.state = 2;
    }
    bool assign_bucket(pipeline_slot_t &slot) {
        if (slot.bucket_index >= 0 && !slot.cursor.done) return true;
        while (next_export < int(ids.size()) &&
               (base+next_export)%g_world == uint64_t(g_rank)) ++next_export;
        if (next_export == int(ids.size())) return false;
        slot.bucket_index = next_export++;
        slot.cursor = {};
        slot.part = 0;
        return true;
    }
    bool candidates_ready() {
        return g_ready_batches.load(std::memory_order_acquire) != 0;
    }
    void pack(pipeline_slot_t &slot, int c) {
        const auto start = std::chrono::steady_clock::now();
        pipeline_header_t h;
        h.kind = c; h.csd = csd; h.wave = base;
        uint8_t *payload = slot.data + sizeof(h);
        const size_t capacity = slot.capacity - sizeof(h);
        if (!c) {
            size_t used = 0;
            if (bwc->mpi_export_bucket_frame(ids[slot.bucket_index], payload, capacity,
                                             &used, &slot.cursor)) throw std::runtime_error("bucket export");
            h.bucket = base + slot.bucket_index;
            h.part = slot.part++;
            h.final = slot.cursor.done;
            h.bytes = used;
        } else {
            // One worker takes a batch as an indivisible columnar block. Its
            // charge moves to this slot and is returned only on peer ACK.
            h.part = slot.part;
            std::vector<candidate_queue_t *> queues;
            {
                std::lock_guard<std::mutex> producers(g_producer_mutex);
                for (auto &entry : g_producers) queues.push_back(entry.second.get());
            }
            for (auto *queue : queues) {
                std::lock_guard<std::mutex> lock(queue->mutex);
                auto &batches = queue->batches;
                while (!batches.empty() && h.bytes + batches.front().size() <= capacity) {
                    const auto &batch = batches.front();
                    memcpy(payload + h.bytes, batch.data(), batch.size());
                    h.bytes += batch.size();
                    batches.pop_front();
                    g_ready_batches.fetch_sub(1);
                }
            }
            slot.charged = h.bytes;
            if (h.bytes) ++slot.part;
        }
        const auto crc_start = pipeline_now_ns();
        h.checksum = checksum64(payload, h.bytes);
        g_crc_ns += pipeline_now_ns()-crc_start;
        if (h.bytes && test_corrupt.exchange(false)) payload[0] ^= 1;
        slot.header = h;
        memcpy(slot.data, &h, sizeof(h));
        pack_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::steady_clock::now()-start).count();
    }
    void apply(pipeline_slot_t &slot, int c) {
        const auto start = std::chrono::steady_clock::now();
        if (test_delay_us) usleep(test_delay_us);
        const auto &h = slot.header;
        const uint8_t *payload = slot.data + sizeof(h);
        const auto crc_start = pipeline_now_ns();
        const auto checksum = checksum64(payload, h.bytes);
        g_crc_ns += pipeline_now_ns()-crc_start;
        if (checksum != h.checksum)
            throw std::runtime_error("payload checksum mismatch");
        // Validate the entire frame before touching cache state.
        const size_t stride = csd + (c ? 14 : 4);
        for (size_t offset = 0; offset < h.bytes;) {
            uint32_t count;
            if (h.bytes-offset < 4) throw std::runtime_error("short block");
            memcpy(&count, payload+offset, 4); offset += 4;
            if (!count || count > Pool_hd_t::chunk_max_nvecs ||
                count > (h.bytes-offset)/stride) throw std::runtime_error("invalid block count");
            if (c) for (uint32_t j = 0; j < count; ++j) {
                uint64_t uid; memcpy(&uid, payload+offset+j*8, 8);
                if (normalized_uid(uid)%g_world != uint64_t(g_rank))
                    throw std::runtime_error("candidate owner mismatch");
            }
            offset += count * stride;
        }
        const size_t index = h.bucket-base;
        chunk_t *dst = nullptr;
        for (size_t offset = 0; offset < h.bytes;) {
            uint32_t count; memcpy(&count, payload+offset, 4); offset += 4;
            const uint8_t *block = payload+offset;
            if (!c) {
                if (bwc->mpi_append_bucket_soa(ids[index], block, block+count*4, count))
                    throw std::runtime_error("bucket import");
            } else {
                const uint8_t *uids = block, *norms = block+count*8;
                const uint8_t *scores = norms+count*4, *vectors = scores+count*2;
                if (g_replay_seen) {
                    for (uint32_t j = 0; j < count; ++j) {
                        uint64_t uid; int32_t norm; uint16_t score;
                        memcpy(&uid,uids+j*8,8); memcpy(&norm,norms+j*4,4); memcpy(&score,scores+j*2,2);
                        replay_candidate(uid,norm,score,vectors+(long)j*csd);
                    }
                    offset += count*stride;
                    continue;
                }
                if (!swc || !ut) throw std::runtime_error("unexpected candidates");
                for (uint32_t pos = 0; pos < count;) {
                    if (!dst) {
                        const auto wait = pipeline_now_ns();
                        dst = swc->fetch_for_write();
                        g_swc_wait_ns += pipeline_now_ns()-wait;
                        if (!dst) throw std::runtime_error("SWC import");
                        _normalize_chunk(dst, csd);
                    }
                    const long n = std::min<long>(count-pos, Pool_hd_t::chunk_max_nvecs-dst->size);
                    if (n <= 0) throw std::runtime_error("invalid SWC size");
                    memcpy(dst->u+dst->size, uids+pos*8, n*8);
                    memcpy(dst->norm+dst->size, norms+pos*4, n*4);
                    memcpy(dst->score+dst->size, scores+pos*2, n*2);
                    memcpy(dst->vec+(long)dst->size*csd, vectors+(long)pos*csd, n*csd);
                    dst->size += n; pos += n;
                    if (dst->size == Pool_hd_t::chunk_max_nvecs) {
                        const auto wait = pipeline_now_ns();
                        ut->task_commit(dst);
                        g_uid_wait_ns += pipeline_now_ns()-wait;
                        dst = nullptr;
                    }
                }
            }
            offset += count*stride;
        }
        if (dst) {
            memset(dst->score+dst->size, 0, (Pool_hd_t::chunk_max_nvecs-dst->size)*2);
            memset(dst->norm+dst->size, 0, (Pool_hd_t::chunk_max_nvecs-dst->size)*4);
            const auto wait = pipeline_now_ns();
            ut->task_commit(dst);
            g_uid_wait_ns += pipeline_now_ns()-wait;
        }
        if (!c && h.final) {
            bwc->bucket_finalize(ids[index]);
            if (g_pipeline_callbacks.ready)
                g_pipeline_callbacks.ready(g_pipeline_callbacks.context);
            finalized.fetch_add(1, std::memory_order_release);
        }
        (c ? apply_ns : receive_ns) += std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::steady_clock::now()-start).count();
    }
    void progress_receive(int c, int k) {
        auto &slot = rx[c][k];
        if (slot.state == 3 && complete(slot.ack_request)) slot.state = 2;
        if (slot.state == 2) {
            int flag = 0; MPI_Status status;
            check(MPI_Test(&slot.data_request, &flag, &status));
            if (!flag) return;
            int bytes = 0; check(MPI_Get_count(&status, MPI_BYTE, &bytes));
            if (bytes < int(sizeof(pipeline_header_t))) throw std::runtime_error("short frame");
            memcpy(&slot.header, slot.data, sizeof(slot.header));
            const auto &h = slot.header;
            if (h.magic != wire_magic || h.version != 3 || h.kind != c || h.csd != uint32_t(csd) ||
                h.wave != base || h.final > 1 || h.bytes != bytes-sizeof(h) ||
                h.bytes > slot.capacity-sizeof(h) || (c && (!h.bytes || h.final || h.bucket)) ||
                (!c && !h.bytes && !h.final))
                throw std::runtime_error("invalid pipeline header");
            if (!c) {
                const uint64_t index = h.bucket-base;
                if (index >= ids.size() || h.bucket%g_world != uint64_t(g_rank) ||
                    bucket_done[index] || h.part != expected_part[index]++ ||
                    (bucket_lane[index] != -1 && bucket_lane[index] != k))
                    throw std::runtime_error("bucket sequence/lane mismatch");
                bucket_lane[index] = k;
                bucket_done[index] = h.final;
            }
            if (c && h.part != slot.part++) throw std::runtime_error("candidate frame sequence mismatch");
            work(slot, c, [this, &slot, c] { apply(slot,c); });
        } else if (slot.state == 1 && slot.worked.load(std::memory_order_acquire)) {
            g_recv_bytes += slot.header.bytes; ++applied[c];
            slot.ack = slot.header.checksum ^ slot.header.part ^ slot.header.bucket ^
                       slot.header.wave ^ (uint64_t(c)<<63);
            post_receive(c,k);
            check(MPI_Isend(&slot.ack, 1, MPI_UINT64_T, 1-g_rank, ack_tag(c,k),
                            MPI_COMM_WORLD, &slot.ack_request));
            slot.state = 3;
        }
    }
    void progress_send(int c, int k) {
        auto &slot = tx[c][k];
        if (slot.state == 2) {
            const bool data_done = complete(slot.data_request);
            const bool ack_done = complete(slot.ack_request);
            if (!data_done || !ack_done) return;
            if (slot.ack != (slot.header.checksum ^ slot.header.part ^ slot.header.bucket ^
                             slot.header.wave ^ (uint64_t(c)<<63)))
                throw std::runtime_error("frame acknowledgement mismatch");
            if (c) g_pipeline_pending.fetch_sub(slot.charged);
            slot.charged = 0; slot.state = 0;
        }
        if (slot.state == 0 && !published_done) {
            if (!c ? assign_bucket(slot) : candidates_ready())
                work(slot, false, [this, &slot, c] { pack(slot,c); });
        } else if (slot.state == 1 && slot.worked.load(std::memory_order_acquire)) {
            if (c && !slot.header.bytes) { slot.state = 0; return; }
            check(MPI_Irecv(&slot.ack, 1, MPI_UINT64_T, 1-g_rank, ack_tag(c,k),
                            MPI_COMM_WORLD, &slot.ack_request));
            check(MPI_Isend(slot.data, sizeof(pipeline_header_t)+slot.header.bytes, MPI_BYTE,
                            1-g_rank, data_tag(c,k), MPI_COMM_WORLD, &slot.data_request));
            g_sent_bytes += slot.header.bytes; ++sent[c];
            g_wire_exchanges.fetch_add(1);
            if (c) g_candidate_sequence.fetch_add(1);
            slot.state = 2;
        }
    }
    void run() {
        const auto start = std::chrono::steady_clock::now();
        for (int c = 0; c < 2; ++c) for (int k = 0; k < 2; ++k) post_receive(c,k);
        check(MPI_Irecv(&peer_done, sizeof(peer_done), MPI_BYTE, 1-g_rank, 5130,
                        MPI_COMM_WORLD, &done_recv));
        uint64_t iterations = 0;
        for (;;) {
            if (error.load()) throw std::runtime_error("pipeline worker failed (CRC/cache/protocol)");
            for (int k = 0; k < 2; ++k) for (int channel = 0; channel < 2; ++channel) {
                const int c = (channel + iterations) % 2;
                progress_receive(c,k); progress_send(c,k);
            }
            const bool incoming_done = received_done &&
                applied[0] == peer_done.bucket_frames && applied[1] == peer_done.candidate_frames;
            const bool insertion_active = g_pipeline_callbacks.service &&
                g_pipeline_callbacks.service(g_pipeline_callbacks.context, !(published_done && incoming_done));
            bool sending = false;
            for (int c = 0; c < 2; ++c) for (int k = 0; k < 2; ++k)
                sending |= tx[c][k].state != 0;
            const bool buckets_done = next_export == int(ids.size()) && finalized.load() == owned;
            // bucket_finalize happens AFTER the reducer publishes its last
            // output. Never infer producer completion from empty queues alone.
            const bool producers_done = buckets_done && (!swc || bwc->num_active() == 0);
            if (!published_done && producers_done && !sending && !g_pipeline_pending.load()) {
                local_done = {base, sent[0], sent[1]};
                check(MPI_Isend(&local_done, sizeof(local_done), MPI_BYTE, 1-g_rank, 5130,
                                MPI_COMM_WORLD, &done_send));
                published_done = true;
            }
            if (!received_done && complete(done_recv)) {
                received_done = true;
                if (peer_done.wave != base) throw std::runtime_error("done epoch mismatch");
            }
            if (received_done && (applied[0] > peer_done.bucket_frames || applied[1] > peer_done.candidate_frames))
                throw std::runtime_error("done frame count mismatch");
            if (published_done && received_done && complete(done_send) && !insertion_active &&
                applied[0] == peer_done.bucket_frames && applied[1] == peer_done.candidate_frames) break;
            if (std::chrono::steady_clock::now()-start > std::chrono::hours(1))
                throw std::runtime_error("pipeline wave timed out");
            // Keep MPI progress frequent; yield without a millisecond sleep.
            if (!(++iterations % 256)) std::this_thread::yield();
        }
        for (int c = 0; c < 2; ++c) for (int k = 0; k < 2; ++k) {
            auto &slot = rx[c][k];
            check(MPI_Wait(&slot.ack_request, MPI_STATUS_IGNORE));
            check(MPI_Cancel(&slot.data_request));
            check(MPI_Wait(&slot.data_request, MPI_STATUS_IGNORE));
        }
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        printf("[MPI-PIPELINE] rank=%d wave=%llu buckets=%zu seconds=%.6f pack_worker_s=%.6f "
               "bucket_apply_worker_s=%.6f candidate_apply_worker_s=%.6f tx_frames=%llu/%llu "
               "candidate_highwater=%llu\n", g_rank, (unsigned long long)base, ids.size(), elapsed,
               pack_ns.load()*1e-9, receive_ns.load()*1e-9, apply_ns.load()*1e-9,
               (unsigned long long)sent[0], (unsigned long long)sent[1],
               (unsigned long long)g_pipeline_highwater.load());
        fflush(stdout);
        const auto starts = g_bucket_started.load()-bucket_start0;
        printf("[MPI-PRESSURE] rank=%d wave=%llu checksum_worker_s=%.6f swc_wait_s=%.6f "
               "uid_commit_wait_s=%.6f ready_to_dequeue_mean_ms=%.6f dequeued=%llu\n",
               g_rank,(unsigned long long)base,(g_crc_ns.load()-crc0)*1e-9,
               (g_swc_wait_ns.load()-swc0)*1e-9,(g_uid_wait_ns.load()-uid0)*1e-9,
               starts ? (g_bucket_wait_ns.load()-bucket_wait0)*1e-6/starts : 0.0,
               (unsigned long long)starts);
        fflush(stdout);
    }
};

// Representative deterministic ~70--75 MiB peer fragments, bidirectional
// candidate traffic, and partial tails. Preparation/verification are untimed.
int pipeline_replay() {
    const char *dim_env = getenv("HD_MPI_REPLAY_CSD");
    const int dim = dim_env ? atoi(dim_env) : 128;
    if ((dim != 128 && dim != 144) || dim > Pool_hd_t::host_vec_nbytes)
        throw std::runtime_error("replay requires supported CSD128 or CSD144");
    NTL::Mat<NTL::ZZ> identity; identity.SetDims(167,167);
    for (int i = 0; i < 167; ++i) identity[i][i] = 1;
    Lattice_QP basis(identity);
    Pool_hd_t pool(&basis);
    if (pool.set_sieving_context(167-dim,167)) throw std::runtime_error("replay context");
    bwc_manager_t buckets(&pool);
    constexpr int nb = 32;
    const long n = 64*Pool_hd_t::chunk_max_nvecs+17;
    std::vector<int32_t> ids(nb), norms(Pool_hd_t::chunk_max_nvecs);
    std::vector<int8_t> vectors(Pool_hd_t::chunk_max_nvecs*dim);
    std::vector<uint64_t> uids(Pool_hd_t::chunk_max_nvecs);
    std::vector<uint16_t> scores(Pool_hd_t::chunk_max_nvecs);
    for (int b = 0; b < nb; ++b) {
        ids[b] = buckets.push_bucket();
        for (long begin = 0; begin < n; begin += Pool_hd_t::chunk_max_nvecs) {
            const long count = std::min(Pool_hd_t::chunk_max_nvecs,n-begin);
            for (long j = 0; j < count; ++j) {
                const long value = (g_rank*nb+b)*n+begin+j;
                norms[j] = value+1;
                memset(vectors.data()+j*dim,int8_t(value%251-125),dim);
            }
            if (buckets.mpi_append_bucket_soa(ids[b],(uint8_t *)norms.data(),
                                             (uint8_t *)vectors.data(),count))
                throw std::runtime_error("replay fixture append");
        }
    }
    const long candidates = 32*Pool_hd_t::chunk_max_nvecs+31;
    std::vector<uint8_t> seen(candidates,0);
    g_replay_csd = dim; g_replay_seen = &seen;
    for (long begin = 0; begin < candidates; begin += Pool_hd_t::chunk_max_nvecs) {
        const long count = std::min(Pool_hd_t::chunk_max_nvecs,candidates-begin);
        for (long j = 0; j < count; ++j) {
            const long value = begin+j;
            uids[j] = (value+1)*2+(1-g_rank);
            norms[j] = value+1; scores[j] = 1+value%65534;
            memset(vectors.data()+j*dim,int8_t(value%251-125),dim);
        }
        if (mpi_sieve_filter_candidates(dim,vectors.data(),norms.data(),scores.data(),uids.data(),count,0))
            throw std::runtime_error("replay candidate routing");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    if (mpi_sieve_exchange_buckets(&buckets,nullptr,nullptr,ids.data(),nb,dim))
        throw std::runtime_error("replay exchange");
    const double local_seconds = MPI_Wtime()-start;
    double seconds = 0;
    MPI_Allreduce(&local_seconds,&seconds,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    if (std::count(seen.begin(),seen.end(),uint8_t(1)) != candidates)
        throw std::runtime_error("replay missing candidates");
    uint64_t found = 0;
    for (int id; (id = buckets.pop_bucket()) >= 0;) {
        const auto it = std::find(ids.begin(),ids.end(),id);
        if (it == ids.end()) throw std::runtime_error("replay unexpected bucket");
        const int b = it-ids.begin();
        std::vector<uint8_t> bucket_seen(n*2,0);
        while (chunk_t *chunk = buckets.fetch_for_read(id)) {
            for (int j = 0; j < chunk->size; ++j) {
                const long value = chunk->norm[j]-1;
                const long source = value/(nb*n), offset = value%n;
                if (value < 0 || source > 1 || (value/n)%nb != b || bucket_seen[source*n+offset])
                    throw std::runtime_error("replay bucket identity mismatch");
                bucket_seen[source*n+offset] = 1; ++found;
                for (int k = 0; k < dim; ++k)
                    if (chunk->vec[(long)j*dim+k] != int8_t(value%251-125))
                        throw std::runtime_error("replay bucket coordinate mismatch");
            }
            buckets.read_done(chunk,id);
        }
        buckets.bucket_finalize(id);
        if (std::count(bucket_seen.begin(),bucket_seen.end(),uint8_t(1)) != 2*n)
            throw std::runtime_error("replay bucket missing records");
    }
    found = mpi_sieve_global_u64(found);
    if (found != uint64_t(nb*n*2)) throw std::runtime_error("replay missing buckets");
    g_replay_seen = nullptr;
    if (!g_rank) {
        printf("[MPI-REPLAY] csd=%d mode=%s elapsed_seconds=%.9f bucket_records=%llu candidate_records=%ld\n",
               dim,g_async_mode ? "async" : "legacy",seconds,(unsigned long long)found,candidates*2);
        fflush(stdout);
    }
    return 0;
}

int pipeline_cache_pressure(Lattice_QP *basis) {
    if (!g_async_mode) throw std::runtime_error("cache pressure test requires async pipeline");
    Pool_hd_t pool(basis);
    if (pool.set_sieving_context(basis->NumRows()-60,basis->NumRows()))
        throw std::runtime_error("cache test context");
    bwc_manager_t buckets(&pool);
    swc_manager_t solutions(&pool);
    solutions.set_max_cached_chunks(256);
    pthread_spinlock_t stats_lock;
    pthread_spin_init(&stats_lock,PTHREAD_PROCESS_PRIVATE);
    uint64_t checked = 0, fresh = 0;
    ut_checker_t checker(ut_checker_t::type_sieve,&pool,pool.uid_table,(pwc_manager_t *)&solutions);
    checker.set_num_threads(8); checker.set_max_holding(8); checker.set_exp_batch(4);
    checker.set_stuck_stat(&checked,&fresh,stats_lock);
    const int32_t ids[2] = {int32_t(buckets.push_bucket()),int32_t(buckets.push_bucket())};
    constexpr long total = 262144;
    const uint64_t initial_uids = pool.uid_table->size();
    std::vector<uint8_t> seen(total,0);
    std::atomic<int> produced{0};
    std::atomic<bool> stop{false};
    long highwater = 0;
    std::thread consumer([&] {
        bool draining = false;
        for (;;) {
            const long live = solutions.num_using();
            highwater = std::max(highwater,live);
            if (live >= 240 || stop.load()) draining = true;
            checker.trigger_batch();
            if (draining) {
                if (chunk_t *chunk = solutions.fetch_for_read()) {
                    for (int j = 0; j < chunk->size; ++j) {
                        const uint64_t index = (chunk->u[j]-g_rank)/2-1;
                        if (index >= total || seen[index] || chunk->u[j]%2 != uint64_t(g_rank) ||
                            chunk->norm[j] != int32_t(index+1) || chunk->score[j] != 1+index%65534)
                            std::abort();
                        for (int k = 0; k < 60; ++k)
                            if (chunk->vec[j*60+k] != int8_t(index%251-125)) std::abort();
                        seen[index] = 1;
                    }
                    solutions.read_done(chunk);
                    usleep(2000); // Slow independent consumer: force SWC pressure.
                }
            }
            if (produced.load() == 2)
                for (int id; (id = buckets.pop_bucket()) >= 0;) buckets.bucket_finalize(id);
            if (stop.load() && !solutions.num_ready()) break;
            usleep(1000);
        }
    });
    std::thread producers[2];
    for (int producer = 0; producer < 2; ++producer) producers[producer] = std::thread([&,producer] {
        std::vector<uint64_t> uid(8192);
        std::vector<int32_t> norm(8192);
        std::vector<uint16_t> score(8192);
        std::vector<int8_t> vec(8192*60);
        for (long begin = producer*(total/2); begin < (producer+1)*(total/2); begin += 8192) {
            for (long j = 0; j < 8192; ++j) {
                const long index = begin+j;
                uid[j] = (index+1)*2+(1-g_rank); norm[j] = index+1; score[j] = 1+index%65534;
                memset(vec.data()+j*60,int8_t(index%251-125),60);
            }
            if (mpi_sieve_filter_candidates(60,vec.data(),norm.data(),score.data(),uid.data(),8192,producer))
                std::abort();
        }
        produced.fetch_add(1);
    });
    if (mpi_sieve_exchange_buckets(&buckets,&solutions,&checker,ids,2,60)) std::abort();
    for (auto &producer : producers) producer.join();
    checker.input_done(); checker.wait_work();
    stop.store(true); consumer.join();
    if (highwater < 240 || std::count(seen.begin(),seen.end(),uint8_t(1)) != total ||
        pool.uid_table->size() != initial_uids+total || g_pipeline_pending.load())
        throw std::runtime_error("cache/UID pressure test lost records or did not reach pressure");
    printf("PASS: async SWC/UID pressure rank=%d records=%ld SWC_highwater=%ld/256 UID_holding=8\n",
           g_rank,total,highwater); fflush(stdout);
    // checker owns no further work; its destructor joins worker task tails.
    pthread_spin_destroy(&stats_lock);
    return 0;
}
