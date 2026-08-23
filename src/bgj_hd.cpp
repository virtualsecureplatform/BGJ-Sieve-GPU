#include "../include/bgj_hd.h"
#include "../include/mpi_sieve.h"

#include <sys/time.h>
#include <unistd.h>

#include <omp.h>

static uint64_t __host_mem_available() {
    FILE *fp = fopen("/proc/meminfo", "r");
    if (!fp) return 0;
    char key[64], unit[16];
    unsigned long long value = 0;
    uint64_t available = 0;
    while (fscanf(fp, "%63s %llu %15s", key, &value, unit) == 3) {
        if (strcmp(key, "MemAvailable:") == 0) {
            available = value * 1024ULL;
            break;
        }
    }
    fclose(fp);
    return available;
}

static long __swc_runtime_max_cached_chunks(Pool_hd_t *p) {
    const long base = SWC_DEFAULT_MAX_CACHED_CHUNKS;
    const char *grow_env = getenv("HD_SWC_GROW_GB");
    // Keep the profiled cache size inside the physical-memory envelope by
    // default.  Runtime growth remains available as an explicit experiment.
    double grow_gb = grow_env ? atof(grow_env) : 0.0;
    if (grow_gb <= 0.0) return base;

    const char *min_csd_env = getenv("HD_SWC_GROW_MIN_CSD");
    const long min_csd = min_csd_env ? atol(min_csd_env) : 124;
    if (!p || p->CSD < min_csd) return base;

    const char *reserve_env = getenv("HD_SWC_RESERVE_GB");
    // The reducer allocates up to roughly 8 GiB of additional pinned staging
    // memory after this decision at the largest dimensions.  Keeping 20 GiB
    // here therefore preserves about a 12 GiB live-run cushion.
    double reserve_gb = reserve_env ? atof(reserve_env) : 20.0;
    if (reserve_gb < 8.0) reserve_gb = 8.0;
    const uint64_t gib = 1ULL << 30;
    const uint64_t available = __host_mem_available();
    const uint64_t reserve = (uint64_t)(reserve_gb * gib);
    if (available <= reserve) return base;

    uint64_t allowed_extra = available - reserve;
    const uint64_t requested_extra = (uint64_t)(grow_gb * gib);
    if (allowed_extra > requested_extra) allowed_extra = requested_extra;
    const uint64_t chunk_nbytes = Pool_hd_t::chunk_max_nvecs *
                                  (uint64_t)POOL_HOST_VEC_SLOT_NBYTES +
                                  (ONE_TIME_IO ? 4096ULL : 0ULL);
    const long extra_chunks = (long)(allowed_extra / chunk_nbytes);
    return extra_chunks > 0 ? base + extra_chunks : base;
}

template <class logger_t> bwc_manager_tmpl<logger_t>::bwc_manager_tmpl(Pool_hd_t *p) : 
                          pwc_manager_tmpl<logger_t>(bwc_default_loading_threads,
                                                     bwc_default_syncing_threads,
                                                     bwc_default_max_cached_chunks,
                                                     true) {
    _num_buckets = 0;
    _num_deleted_buckets = 0;
    _num_wl = 0;
    _num_wp = 0;
    _num_ready_buckets = 0;
    _num_prefetched_bucket = 0;
    
    for (long i = 0; i < bwc_bucket_locks; i++) {
        pthread_spin_init(&_bucket_lock[i], PTHREAD_PROCESS_SHARED);
    }

    pthread_spin_init(&_bwc_wp_lock, PTHREAD_PROCESS_SHARED);
    pthread_spin_init(&_bwc_lock, PTHREAD_PROCESS_SHARED);
    
    this->_pool = p;
    this->set_dirname("bucket");

    lg_info("manager initialized, (%d, %d) threads for I/O, #caching = %d", 
            _loading_threads, _syncing_threads, _max_cached_chunks);
}

template <class logger_t> void bwc_manager_tmpl<logger_t>::configure_hbm_cache(bool enable) {
    if (!enable) {
        __destroy_hbm_cache();
        return;
    }
    if (_hbm_enabled) return;

    const char *env_gb = getenv("HD_HBM_BWC_GB");
    const double requested_gb = env_gb ? atof(env_gb) : 12.0;
    if (requested_gb <= 0.0 || hw::gpu_num <= 0) return;

    const size_t gib = 1ULL << 30;
    const size_t reserve_nbytes = 24ULL * gib;
    const size_t raw_slot_nbytes = Pool_hd_t::chunk_max_nvecs *
                                   (sizeof(int32_t) + this->_pool->CSD);
    const size_t slot_nbytes = (raw_slot_nbytes + 255) & ~(size_t)255;
    int enabled_devices = 0;

    for (int device_ptr = 0; device_ptr < hw::gpu_num; device_ptr++) {
        size_t free_nbytes = 0, total_nbytes = 0;
        if (_cuda_device_mem_info(device_ptr, &free_nbytes, &total_nbytes)) continue;
        size_t target_nbytes = (size_t)(requested_gb * gib);
        if (free_nbytes <= reserve_nbytes) target_nbytes = 0;
        else if (target_nbytes > free_nbytes - reserve_nbytes)
            target_nbytes = free_nbytes - reserve_nbytes;

        int32_t num_slots = (int32_t)(target_nbytes / slot_nbytes);
        int alloc_status = -1;
        while (num_slots > 0) {
            target_nbytes = (size_t)num_slots * slot_nbytes;
            alloc_status = _cuda_device_malloc(device_ptr,
                                                (void **)&_hbm[device_ptr].base,
                                                target_nbytes);
            if (alloc_status == 0) break;
            num_slots /= 2;
        }
        if (alloc_status != 0 || num_slots == 0) {
            _hbm[device_ptr].base = NULL;
            continue;
        }

        _hbm[device_ptr].free_slots = (int32_t *)malloc(num_slots * sizeof(int32_t));
        if (!_hbm[device_ptr].free_slots) {
            _cuda_device_free(device_ptr, _hbm[device_ptr].base);
            _hbm[device_ptr].base = NULL;
            continue;
        }
        pthread_spin_init(&_hbm[device_ptr].lock, PTHREAD_PROCESS_PRIVATE);
        _hbm[device_ptr].lock_initialized = true;
        _hbm[device_ptr].slot_nbytes = slot_nbytes;
        _hbm[device_ptr].num_slots = num_slots;
        _hbm[device_ptr].num_free = num_slots;
        for (int32_t i = 0; i < num_slots; i++)
            _hbm[device_ptr].free_slots[i] = i;
        enabled_devices++;
        lg_info("GPU %d HBM bucket cache: %.2f GiB, %d exact chunks",
                hw::gpu_id_list[device_ptr], target_nbytes / (double)gib, num_slots);
    }
    _hbm_enabled = enabled_devices > 0;
}

template <class logger_t> void bwc_manager_tmpl<logger_t>::configure_gpu_native_buckets(
        bool enable, long max_bucket_nvecs) {
    _gpu_native_enabled = false;
    _gpu_native_max_nvecs = 0;
    _gpu_native_max_chunks = 0;

    const char *env = getenv("HD_GPU_NATIVE_BWC");
    const bool requested = enable && env && atol(env) != 0 && !mpi_sieve_active();
    const char *int4_env = getenv("HD_INT4_BUCKETS");
    if (!requested || !_hbm_enabled || max_bucket_nvecs <= 0) return;
    if (int4_env && atol(int4_env) != 0) {
        fprintf(stderr, "[GPU-BWC] disabled because HD_INT4_BUCKETS is active\n");
        return;
    }
    if (max_bucket_nvecs > INT32_MAX) {
        fprintf(stderr, "[GPU-BWC] disabled: bucket capacity %ld exceeds INT32_MAX\n",
                max_bucket_nvecs);
        return;
    }

    _gpu_native_max_nvecs = (int32_t)max_bucket_nvecs;
    _gpu_native_max_chunks =
        (_gpu_native_max_nvecs + Pool_hd_t::chunk_max_nvecs - 1) /
        Pool_hd_t::chunk_max_nvecs;
    _gpu_native_buckets.store(0, std::memory_order_relaxed);
    _gpu_native_host_fallbacks.store(0, std::memory_order_relaxed);
    _gpu_native_entries.store(0, std::memory_order_relaxed);
    _gpu_native_overflows.store(0, std::memory_order_relaxed);
    _gpu_native_enabled = _gpu_native_max_chunks > 0;
    if (_gpu_native_enabled) {
        printf("[GPU-BWC] enabled: exact GPU materialization, "
               "capacity %d vectors (%d HBM slots) per "
               "selected bucket\n",
               _gpu_native_max_nvecs, _gpu_native_max_chunks);
        fflush(stdout);
    }
}

template <class logger_t> void bwc_manager_tmpl<logger_t>::__destroy_hbm_cache() {
    _gpu_native_enabled = false;
    for (int device_ptr = 0; device_ptr < hw::gpu_num; device_ptr++) {
        hbm_arena_t &arena = _hbm[device_ptr];
        if (arena.base) {
            if (_cuda_device_free(device_ptr, arena.base))
                fprintf(stderr, "[Warning] failed to free GPU %d HBM bucket cache\n",
                        hw::gpu_id_list[device_ptr]);
            arena.base = NULL;
        }
        if (arena.lock_initialized) {
            pthread_spin_destroy(&arena.lock);
            arena.lock_initialized = false;
        }
        free(arena.free_slots);
        arena.free_slots = NULL;
        arena.num_slots = arena.num_free = 0;
        arena.slot_nbytes = 0;
    }
    _hbm_enabled = false;
}

template <class logger_t> int bwc_manager_tmpl<logger_t>::__hbm_alloc(int device_ptr) {
    hbm_arena_t &arena = _hbm[device_ptr];
    if (!arena.base) return -1;
    pthread_spin_lock(&arena.lock);
    int ret = arena.num_free ? arena.free_slots[--arena.num_free] : -1;
    pthread_spin_unlock(&arena.lock);
    return ret;
}

template <class logger_t> bool bwc_manager_tmpl<logger_t>::__hbm_reserve(
        int device_ptr, int32_t num_slots, int32_t *slots) {
    if (device_ptr < 0 || device_ptr >= hw::gpu_num || num_slots <= 0 || !slots)
        return false;
    hbm_arena_t &arena = _hbm[device_ptr];
    if (!arena.base) return false;
    pthread_spin_lock(&arena.lock);
    if (arena.num_free < num_slots) {
        pthread_spin_unlock(&arena.lock);
        return false;
    }
    for (int32_t i = 0; i < num_slots; i++)
        slots[i] = arena.free_slots[--arena.num_free];
    pthread_spin_unlock(&arena.lock);
    return true;
}

template <class logger_t> void bwc_manager_tmpl<logger_t>::__hbm_release(int device_ptr,
                                                                         int slot_id) {
    if (device_ptr < 0 || device_ptr >= hw::gpu_num || slot_id < 0) return;
    hbm_arena_t &arena = _hbm[device_ptr];
    if (!arena.base) return;
    pthread_spin_lock(&arena.lock);
    arena.free_slots[arena.num_free++] = slot_id;
    pthread_spin_unlock(&arena.lock);
}

template <class logger_t> int8_t *bwc_manager_tmpl<logger_t>::__hbm_slot(int device_ptr,
                                                                         int slot_id) {
    return _hbm[device_ptr].base + (size_t)slot_id * _hbm[device_ptr].slot_nbytes;
}

template <class logger_t> bool bwc_manager_tmpl<logger_t>::__start_gpu_native_bucket(
        int32_t bucket_id) {
    if (!_gpu_native_enabled || _gpu_native_max_chunks <= 0) return false;

    l0_bucket_t &bucket = _bucket[bucket_id];
    bucket.reserve_chunks(_gpu_native_max_chunks);
    int device_ptr = -1;
    for (int step = 0; step < hw::gpu_num; step++) {
        int candidate = (bucket_id + step) % hw::gpu_num;
        if (__hbm_reserve(candidate, _gpu_native_max_chunks, bucket.chunk_ids)) {
            device_ptr = candidate;
            break;
        }
    }
    if (device_ptr < 0) {
        _gpu_native_host_fallbacks.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    bucket.status |= l0_bucket_t::_bk_gpu_write;
    bucket.hbm_device = device_ptr;
    bucket.hbm_reserved_chunks = _gpu_native_max_chunks;
    bucket.hbm_capacity_nvecs = _gpu_native_max_nvecs;
    bucket.hbm_nvecs = 0;
    _gpu_native_buckets.fetch_add(1, std::memory_order_relaxed);
    return true;
}

template <class logger_t> bool bwc_manager_tmpl<logger_t>::reserve_gpu_native_write(
        long bucket_id, int entry_size, bwc_gpu_write_desc_t *desc) {
    if (!desc) return false;
    memset(desc, 0, sizeof(*desc));
    desc->device_ptr = -1;
    if (bucket_id < 0 || bucket_id >= _num_buckets || entry_size < 0) return false;

    pthread_spin_lock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
    l0_bucket_t &bucket = _bucket[bucket_id];
    if ((bucket.status & (l0_bucket_t::_bk_writing | l0_bucket_t::_bk_gpu_write)) !=
        (l0_bucket_t::_bk_writing | l0_bucket_t::_bk_gpu_write)) {
        pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
        return false;
    }

    desc->enabled = 1;
    desc->device_ptr = bucket.hbm_device;
    desc->entry_size = entry_size;
    const int32_t available = bucket.hbm_capacity_nvecs - bucket.hbm_nvecs;
    const int32_t write_size = entry_size < available ? entry_size : available;
    const int32_t start = bucket.hbm_nvecs;
    bucket.hbm_nvecs += write_size;
    desc->write_size = write_size;

    int32_t logical_pos = start;
    int32_t described = 0;
    while (described < write_size && desc->num_spans < bwc_gpu_write_max_spans) {
        const int32_t slot_ptr = logical_pos / Pool_hd_t::chunk_max_nvecs;
        const int32_t slot_offset = logical_pos % Pool_hd_t::chunk_max_nvecs;
        const int32_t room = Pool_hd_t::chunk_max_nvecs - slot_offset;
        const int32_t span_size =
            write_size - described < room ? write_size - described : room;
        int8_t *slot = __hbm_slot(bucket.hbm_device, bucket.chunk_ids[slot_ptr]);
        const int span = desc->num_spans++;
        described += span_size;
        desc->span_end[span] = described;
        desc->d_norm[span] = (int32_t *)slot + slot_offset;
        desc->d_vec[span] = slot +
            Pool_hd_t::chunk_max_nvecs * sizeof(int32_t) +
            (size_t)slot_offset * this->_pool->CSD;
        logical_pos += span_size;
    }
    pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);

    if (described != write_size || write_size != entry_size) {
        _gpu_native_overflows.fetch_add((uint64_t)(entry_size - described),
                                        std::memory_order_relaxed);
        fprintf(stderr,
                "[GPU-BWC] fatal capacity error: bucket %ld requested %d entries, "
                "only %d described; refusing a lossy bucket\n",
                bucket_id, entry_size, described);
        abort();
    }
    _gpu_native_entries.fetch_add(write_size, std::memory_order_relaxed);
    return true;
}

template <class logger_t> void bwc_manager_tmpl<logger_t>::report_gpu_native() {
    if (!_gpu_native_max_chunks && !_gpu_native_buckets.load(std::memory_order_relaxed))
        return;
    printf("[GPU-BWC] buckets direct %lu, host fallback %lu, exact entries %lu, "
           "overflow %lu\n",
           (unsigned long)_gpu_native_buckets.load(std::memory_order_relaxed),
           (unsigned long)_gpu_native_host_fallbacks.load(std::memory_order_relaxed),
           (unsigned long)_gpu_native_entries.load(std::memory_order_relaxed),
           (unsigned long)_gpu_native_overflows.load(std::memory_order_relaxed));
    fflush(stdout);
}

template <class logger_t> bool bwc_manager_tmpl<logger_t>::__stage_bucket_to_hbm(
        int32_t bucket_id) {
    if (!_hbm_enabled || !this->_lazy_sync || _bucket[bucket_id].num_chunks <= 0)
        return false;

    const int32_t num_chunks = _bucket[bucket_id].num_chunks;
    int device_ptr = -1;
    for (int step = 0; step < hw::gpu_num; step++) {
        int candidate = (bucket_id + step) % hw::gpu_num;
        hbm_arena_t &arena = _hbm[candidate];
        if (!arena.base) continue;
        pthread_spin_lock(&arena.lock);
        bool fits = arena.num_free >= num_chunks;
        pthread_spin_unlock(&arena.lock);
        if (fits) { device_ptr = candidate; break; }
    }
    if (device_ptr < 0) return false;

    int32_t *slots = (int32_t *)malloc(num_chunks * sizeof(int32_t));
    chunk_t **chunks = (chunk_t **)malloc(num_chunks * sizeof(chunk_t *));
    if (!slots || !chunks) {
        free(slots);
        free(chunks);
        return false;
    }

    int32_t reserved = 0;
    for (; reserved < num_chunks; reserved++) {
        slots[reserved] = __hbm_alloc(device_ptr);
        if (slots[reserved] < 0) break;
    }
    if (reserved != num_chunks) {
        for (int32_t i = 0; i < reserved; i++) __hbm_release(device_ptr, slots[i]);
        free(slots);
        free(chunks);
        return false;
    }

    int32_t fetched = 0;
    for (; fetched < num_chunks; fetched++) {
        chunks[fetched] = pwc_manager_tmpl<logger_t>::fetch(_bucket[bucket_id].chunk_ids[fetched]);
        if (!chunks[fetched]) break;
    }
    if (fetched != num_chunks) {
        for (int32_t i = 0; i < fetched; i++)
            pwc_manager_tmpl<logger_t>::release(chunks[i]->id);
        for (int32_t i = 0; i < num_chunks; i++) __hbm_release(device_ptr, slots[i]);
        free(slots);
        free(chunks);
        return false;
    }

    const size_t norm_capacity = Pool_hd_t::chunk_max_nvecs * sizeof(int32_t);
    for (int32_t i = 0; i < num_chunks; i++) {
        int8_t *slot = __hbm_slot(device_ptr, slots[i]);
        if (_cuda_device_h2d_pair_enqueue(
                device_ptr,
                slot, chunks[i]->norm, chunks[i]->size * sizeof(int32_t),
                slot + norm_capacity, chunks[i]->vec,
                (size_t)chunks[i]->size * this->_pool->CSD)) {
            fprintf(stderr, "[Error] failed to stage bucket %d in GPU %d HBM\n",
                    bucket_id, hw::gpu_id_list[device_ptr]);
            abort();
        }
        _bucket[bucket_id].hbm_sizes[i] = chunks[i]->size;
    }
    // Keep the bucket private until every chunk copy is complete, but avoid a
    // host/device round trip after each pair of copies.
    if (_cuda_device_h2d_wait(device_ptr)) {
        fprintf(stderr, "[Error] failed to finish staging bucket %d in GPU %d HBM\n",
                bucket_id, hw::gpu_id_list[device_ptr]);
        abort();
    }
    for (int32_t i = 0; i < num_chunks; i++) {
        release_del(chunks[i]->id);
        _bucket[bucket_id].chunk_ids[i] = slots[i];
    }
    _bucket[bucket_id].hbm_device = device_ptr;
    free(slots);
    free(chunks);
    return true;
}

template <class logger_t> bool bwc_manager_tmpl<logger_t>::bucket_in_hbm(long bucket_id) {
    pthread_spin_lock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
    bool ret = (_bucket[bucket_id].status & l0_bucket_t::_bk_hbm) != 0;
    pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
    return ret;
}

template <class logger_t> bool bwc_manager_tmpl<logger_t>::fetch_hbm_for_read(
        long bucket_id, int device_ptr, int *size, const int32_t **d_norm,
        const int8_t **d_vec, int *slot_id) {
    pthread_spin_lock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
    if ((_bucket[bucket_id].status & (l0_bucket_t::_bk_reading | l0_bucket_t::_bk_hbm)) !=
            (l0_bucket_t::_bk_reading | l0_bucket_t::_bk_hbm) ||
        _bucket[bucket_id].hbm_device != device_ptr ||
        _bucket[bucket_id].num_chunks == 0) {
        pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
        return false;
    }
    const int32_t ptr = --_bucket[bucket_id].num_chunks;
    *slot_id = _bucket[bucket_id].chunk_ids[ptr];
    *size = _bucket[bucket_id].hbm_sizes[ptr];
    int8_t *slot = __hbm_slot(device_ptr, *slot_id);
    *d_norm = (const int32_t *)slot;
    *d_vec = slot + Pool_hd_t::chunk_max_nvecs * sizeof(int32_t);
    pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
    return true;
}

template <class logger_t> void bwc_manager_tmpl<logger_t>::hbm_read_done(int device_ptr,
                                                                         int slot_id) {
    __hbm_release(device_ptr, slot_id);
}

template <class logger_t> bwc_manager_tmpl<logger_t>::~bwc_manager_tmpl() {
    this->_syncing_pool.wait_work();
    this->_loading_pool.wait_work();

    if (_num_wl) lg_err("still %d(4w) chunks loading? ignored.", _num_wl);
    
    for (int32_t i = 0; i < _num_wp; i++) release_del(_writing_prefetch_chunks[i]->id);
    pthread_spin_destroy(&_bwc_wp_lock);

    for (int32_t i = 0; i < _num_buckets; i++) {
        if (_bucket[i].status) {
            if (_bucket[i].status & l0_bucket_t::_bk_gpu_write) {
                for (int32_t j = 0; j < _bucket[i].hbm_reserved_chunks; j++)
                    __hbm_release(_bucket[i].hbm_device, _bucket[i].chunk_ids[j]);
            } else if (_bucket[i].status & l0_bucket_t::_bk_hbm) {
                for (int32_t j = 0; j < _bucket[i].num_chunks; j++)
                    __hbm_release(_bucket[i].hbm_device, _bucket[i].chunk_ids[j]);
            } else {
                for (int32_t j = 0; j < _bucket[i].num_chunks; j++)
                    _chunk_status[_bucket[i].chunk_ids[j]] &=
                        ~(_ck_writing | _ck_reading | _ck_to_sync);
            }
        }
        if (_bucket[i].chunk_ids) free(_bucket[i].chunk_ids);
        if (_bucket[i].hbm_sizes) free(_bucket[i].hbm_sizes);
        if (_bucket[i].writing_chunk && _bucket[i].writing_chunk != (chunk_t *) -1) release_del(_bucket[i].writing_chunk->id);
    }

    for (int32_t i = 0; i < bwc_bucket_locks; i++) {
        pthread_spin_destroy(&_bucket_lock[i]);
    }
    pthread_spin_destroy(&_bwc_lock);

    this->_syncing_pool.wait_work();
    this->_loading_pool.wait_work();

    /// delete all files
    char chunk_filename[256];
    for (long i = 0; i < _num_chunks; i++) {
        #if MULTI_SSD
        snprintf(chunk_filename, sizeof(chunk_filename), "%s/%s/%s%06lx", this->_dir, hw::ssd_name(i), this->_pfx, i);
        #else
        snprintf(chunk_filename, sizeof(chunk_filename), "%s%06lx", this->_prefix, i);
        #endif
        remove(chunk_filename);
    }
    __destroy_hbm_cache();
}

template <class logger_t> void bwc_manager_tmpl<logger_t>::__prefetch_for_writing() {
    lg_init();
    volatile int32_t *_num_wp_ptr_vol = reinterpret_cast<volatile int32_t*>(&_num_wp);
    volatile int32_t *_num_wl_ptr_vol = reinterpret_cast<volatile int32_t*>(&_num_wl);

    // almost the same as swc_manager_t::__prefetch_for_writing
    if (_num_wl + _num_wp >= bwc_auto_prefetch_for_write) {
        lg_exit();
        return;
    }

    pthread_spin_lock(&_bwc_wp_lock);
    int to_prefetch = bwc_auto_prefetch_for_write - *_num_wl_ptr_vol - *_num_wp_ptr_vol;
    if (to_prefetch <= 0) {
        pthread_spin_unlock(&_bwc_wp_lock);
        lg_exit();
        return;
    }
    _num_wl += to_prefetch;
    pthread_spin_unlock(&_bwc_wp_lock);

    int32_t num_fail = 0;
    for (int32_t i = 0; i < to_prefetch; i++) {
        int32_t id = create_chunk();
        pthread_spin_lock(&_locks[id % pwc_locks]);
        if (_chunk_status[id]) {
            num_fail++;
        } else {
            _chunk_status[id] |= _ck_loading | _ck_writing;
            this->_loading_pool.push([=]() {
                __load_chunk(id);
                pthread_spin_lock(&_bwc_wp_lock);
                _writing_prefetch_chunks[_num_wp++] = &_cached_chunks[_chunk_status[id] & _ck_cache_id_mask];
                _num_wl--;
                pthread_spin_unlock(&_bwc_wp_lock);
            });
            _num_loading_chunks++;
        }
        pthread_spin_unlock(&_locks[id % pwc_locks]);
    }

    if (num_fail) {
        lg_err("%d of %d new chunk already in use?", num_fail, to_prefetch);
        pthread_spin_lock(&_bwc_wp_lock);
        _num_wl -= num_fail;
        pthread_spin_unlock(&_bwc_wp_lock);
    }
    lg_exit();
}

template <class logger_t> void bwc_manager_tmpl<logger_t>::__prefetch_for_reading(int32_t bucket_id) {
    lg_init();
    if (_bucket[bucket_id].status & l0_bucket_t::_bk_hbm) {
        lg_exit();
        return;
    }
    volatile uint32_t *status_ptr_vol = reinterpret_cast<volatile uint32_t*>(&_bucket[bucket_id].status);
    volatile  int32_t *num_pb_ptr_vol = reinterpret_cast<volatile  int32_t*>(&_num_prefetched_bucket);

    /// try to convert it to a hard prefetch bucket
    if (_bucket[bucket_id].status == l0_bucket_t::_bk_reading) {
        if (_num_prefetched_bucket < bwc_auto_prefetch_for_read) {
            pthread_spin_lock(&_bwc_lock);
            if (*num_pb_ptr_vol < bwc_auto_prefetch_for_read) {
                _prefetched_bucket_id[num_pb_ptr_vol[0]++] = bucket_id;
                pthread_spin_unlock(&_bwc_lock);

                pthread_spin_lock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
                if (status_ptr_vol[0] == l0_bucket_t::_bk_reading) {
                    status_ptr_vol[0] |= l0_bucket_t::_bk_caching;
                    pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
                } else {
                    pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
                    pthread_spin_lock(&_bwc_lock);
                    for (int32_t i = 0; i < _num_prefetched_bucket; i++) {
                        if (_prefetched_bucket_id[i] == bucket_id) {
                            _prefetched_bucket_id[i] = _prefetched_bucket_id[--_num_prefetched_bucket];
                            break;
                        }
                    }
                    pthread_spin_unlock(&_bwc_lock);
                }
            } else pthread_spin_unlock(&_bwc_lock);
        }
    }
    

    if (_bucket[bucket_id].status & l0_bucket_t::_bk_caching) {
        pthread_spin_lock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
        for (int32_t i = _bucket[bucket_id].num_chunks - 1; i >= 0 && 
                    i >= _bucket[bucket_id].num_chunks - bwc_auto_prefetch_for_read_depth; i--) {
            if (_chunk_status[_bucket[bucket_id].chunk_ids[i]] & _ck_writing) continue;
            pthread_spin_lock(&_locks[_bucket[bucket_id].chunk_ids[i] % pwc_locks]);
            _chunk_status[_bucket[bucket_id].chunk_ids[i]] |= _ck_writing;
            pthread_spin_unlock(&_locks[_bucket[bucket_id].chunk_ids[i] % pwc_locks]);
        }
        pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
    }

    pthread_spin_lock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
    for (int32_t i = _bucket[bucket_id].num_chunks - 1; i >= 0 && 
                i >= _bucket[bucket_id].num_chunks - bwc_auto_prefetch_for_read_depth; i--) {
        int32_t chunk_id = _bucket[bucket_id].chunk_ids[i];
        if (_chunk_status[chunk_id] & (_ck_caching | _ck_loading)) continue;
        if (chunk_id >= _num_chunks || chunk_id < 0 || _chunk_status[chunk_id] == _ck_size_mask) {
            lg_err("invalid chunk id %d(%x)", chunk_id, _chunk_status[chunk_id]);
            continue;
        }
        
        volatile chunk_status_t *_chunk_status_vol = reinterpret_cast<volatile chunk_status_t*>(_chunk_status);

        pthread_spin_lock(&_locks[chunk_id % pwc_locks]);
        chunk_status_t status = _chunk_status_vol[chunk_id];
        if (!(status & (_ck_caching | _ck_loading)) && (status != _ck_size_mask)) {
            _chunk_status[chunk_id] |= _ck_loading;
            _loading_pool.push([=]() { __load_chunk(chunk_id); });
            _num_loading_chunks++;
        }
        pthread_spin_unlock(&_locks[chunk_id % pwc_locks]);
    }
    pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);

    lg_exit();
}

template <class logger_t> long bwc_manager_tmpl<logger_t>::push_bucket() {
    lg_init();
    volatile int32_t *_num_wp_ptr_vol = reinterpret_cast<volatile int32_t*>(&_num_wp);

    int32_t ret = -1;
    pthread_spin_lock(&_bwc_lock);
    if (_num_deleted_buckets) {
        ret = _deleted_bucket_ids[--_num_deleted_buckets];
    } else if (_num_buckets < bwc_max_buckets) {
        ret = _num_buckets++;
    } else lg_err("no place for new buckets(%ld)", bwc_max_buckets);
    pthread_spin_unlock(&_bwc_lock);

    if (ret >= 0) {
        _bucket[ret].init();

        const bool gpu_native = __start_gpu_native_bucket(ret);
        if (!gpu_native && _num_wp) {
            pthread_spin_lock(&_bwc_wp_lock);
            if (*_num_wp_ptr_vol) _bucket[ret].writing_chunk = _writing_prefetch_chunks[--_num_wp];
            pthread_spin_unlock(&_bwc_wp_lock);
            if (_bucket[ret].writing_chunk) {
                _bucket[ret].add_chunk(_bucket[ret].writing_chunk->id);
            }
        }

        if (!gpu_native) __prefetch_for_writing();
    }

    lg_exit();
    return ret;
}

template <class logger_t> long bwc_manager_tmpl<logger_t>::pop_bucket(int device_ptr) {
    lg_init();
    int32_t ret = -1;
    bool was_prefetched = false;

    pthread_spin_lock(&_bwc_lock);
    for (int32_t i = 0; i < _num_prefetched_bucket; i++) {
        int32_t candidate = _prefetched_bucket_id[i];
        bool device_match = !(_bucket[candidate].status & l0_bucket_t::_bk_hbm) ||
                            device_ptr < 0 || _bucket[candidate].hbm_device == device_ptr;
        if ((_bucket[candidate].status & l0_bucket_t::_bk_ready) && device_match) {
            ret = _prefetched_bucket_id[i];
            was_prefetched = true;
            _bucket[_prefetched_bucket_id[i]].status = 
            (_bucket[_prefetched_bucket_id[i]].status & ~l0_bucket_t::_bk_ready) | l0_bucket_t::_bk_reading;
            break;
        }
    }
    if (ret == -1 && _num_ready_buckets) {
        int32_t ready_ptr = -1;
        for (int32_t i = _num_ready_buckets - 1; i >= 0; i--) {
            int32_t candidate = _ready_bucket_id[i];
            if (!(_bucket[candidate].status & l0_bucket_t::_bk_hbm) ||
                device_ptr < 0 || _bucket[candidate].hbm_device == device_ptr) {
                ready_ptr = i;
                break;
            }
        }
        if (ready_ptr >= 0) {
            ret = _ready_bucket_id[ready_ptr];
            _ready_bucket_id[ready_ptr] = _ready_bucket_id[--_num_ready_buckets];
        }
    }
    if (ret != -1) {
        _bucket[ret].status = (_bucket[ret].status & ~l0_bucket_t::_bk_ready) | l0_bucket_t::_bk_reading;        
        if (!was_prefetched && !(_bucket[ret].status & l0_bucket_t::_bk_hbm) &&
            _num_prefetched_bucket < bwc_auto_prefetch_for_read) {
            _prefetched_bucket_id[_num_prefetched_bucket++] = ret;
            _bucket[ret].status |= l0_bucket_t::_bk_caching;
        }
    }
    pthread_spin_unlock(&_bwc_lock);

    if (ret != -1 && !(_bucket[ret].status & l0_bucket_t::_bk_hbm))
        __prefetch_for_reading(ret);

    lg_exit();
    return ret;
}

template <class logger_t> void bwc_manager_tmpl<logger_t>::bucket_finalize(long bucket_id) {
    lg_init();

    volatile int32_t *_num_wp_ptr_vol = reinterpret_cast<volatile int32_t*>(&_num_wp);
    volatile uint32_t *_chunk_status_vol = reinterpret_cast<volatile uint32_t*>(_chunk_status);
    chunk_t *volatile *writing_chunk_ptr_vol = reinterpret_cast<chunk_t *volatile *>(&_bucket[bucket_id].writing_chunk);

    if ((_bucket[bucket_id].status &
         (l0_bucket_t::_bk_reading | l0_bucket_t::_bk_hbm)) ==
        (l0_bucket_t::_bk_reading | l0_bucket_t::_bk_hbm)) {
        pthread_spin_lock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
        int32_t num_chunks = _bucket[bucket_id].num_chunks;
        int32_t device_ptr = _bucket[bucket_id].hbm_device;
        _bucket[bucket_id].num_chunks = 0;
        _bucket[bucket_id].status = 0;
        pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
        for (int32_t i = 0; i < num_chunks; i++)
            __hbm_release(device_ptr, _bucket[bucket_id].chunk_ids[i]);
        pthread_spin_lock(&_bwc_lock);
        _deleted_bucket_ids[_num_deleted_buckets++] = bucket_id;
        pthread_spin_unlock(&_bwc_lock);
    } else if (_bucket[bucket_id].status & l0_bucket_t::_bk_reading) {
        pthread_spin_lock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
        int32_t num_chunks = _bucket[bucket_id].num_chunks;
        _bucket[bucket_id].num_chunks = 0;
        pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);

        pthread_spin_lock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
        _bucket[bucket_id].status = 0;
        pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
        
        pthread_spin_lock(&_bwc_lock);
        for (int32_t i = 0; i < _num_prefetched_bucket; i++) {
            if (_prefetched_bucket_id[i] == bucket_id) 
                _prefetched_bucket_id[i] = _prefetched_bucket_id[--_num_prefetched_bucket];
        }
        pthread_spin_unlock(&_bwc_lock);

        /// Clean all chunks in the chunk list. I/O completion is lossless:
        /// wait for the worker that owns a busy chunk instead of abandoning
        /// the remainder after an arbitrary ten-second deadline.
        bool warned = false;
        while (num_chunks) {
            int32_t busy_id = -1;
            for (int32_t i = num_chunks - 1; i >= 0; i--) {
                int32_t id = _bucket[bucket_id].chunk_ids[i];
                pthread_spin_lock(&_locks[id % pwc_locks]);
                if ((_chunk_status_vol[id] & (_ck_syncing | _ck_loading)) == 0) {
                    _chunk_status[id] |= _ck_writing | _ck_loading;
                } else {
                    busy_id = id;
                    pthread_spin_unlock(&_locks[id % pwc_locks]);
                    continue;
                }
                pthread_spin_unlock(&_locks[id % pwc_locks]);
                _bucket[bucket_id].chunk_ids[i] = _bucket[bucket_id].chunk_ids[--num_chunks];
                release_del(id);
            }

            if (num_chunks && busy_id >= 0 &&
                !__wait_chunk_io(busy_id, std::chrono::seconds(10))) {
                if (!warned) {
                    lg_warn("%d chunks still loading/syncing after 10 seconds; waiting losslessly",
                            num_chunks);
                    warned = true;
                }
            }
        }

        pthread_spin_lock(&_bwc_lock);
        _deleted_bucket_ids[_num_deleted_buckets++] = bucket_id;
        pthread_spin_unlock(&_bwc_lock);
    } else if ((_bucket[bucket_id].status &
                (l0_bucket_t::_bk_writing | l0_bucket_t::_bk_gpu_write)) ==
               (l0_bucket_t::_bk_writing | l0_bucket_t::_bk_gpu_write)) {
        pthread_spin_lock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
        l0_bucket_t &bucket = _bucket[bucket_id];
        const int32_t nvecs = bucket.hbm_nvecs;
        const int32_t num_chunks =
            (nvecs + Pool_hd_t::chunk_max_nvecs - 1) /
            Pool_hd_t::chunk_max_nvecs;
        const int32_t reserved_chunks = bucket.hbm_reserved_chunks;
        const int32_t device_ptr = bucket.hbm_device;
        for (int32_t i = 0; i < num_chunks; i++) {
            const int32_t used = nvecs - i * Pool_hd_t::chunk_max_nvecs;
            bucket.hbm_sizes[i] = used < Pool_hd_t::chunk_max_nvecs ?
                                  used : Pool_hd_t::chunk_max_nvecs;
        }
        bucket.num_chunks = num_chunks;
        bucket.hbm_reserved_chunks = num_chunks;
        bucket.status = l0_bucket_t::_bk_ready | l0_bucket_t::_bk_hbm;
        pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);

        for (int32_t i = num_chunks; i < reserved_chunks; i++)
            __hbm_release(device_ptr, bucket.chunk_ids[i]);

        pthread_spin_lock(&_bwc_lock);
        if (_num_ready_buckets < bwc_max_buckets) {
            _ready_bucket_id[_num_ready_buckets++] = bucket_id;
        } else {
            lg_err("ready list full(%d), GPU-native bucket %d discarded",
                   _num_ready_buckets, bucket_id);
            bucket.status = l0_bucket_t::_bk_reading | l0_bucket_t::_bk_hbm;
        }
        pthread_spin_unlock(&_bwc_lock);
        if (!(bucket.status & l0_bucket_t::_bk_ready)) bucket_finalize(bucket_id);
    } else if (_bucket[bucket_id].status & l0_bucket_t::_bk_writing) {
        for (;;) {
            if (_bucket[bucket_id].writing_chunk == (chunk_t *) -1) continue;

            pthread_spin_lock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
            chunk_t *tmp = *writing_chunk_ptr_vol;
            if (tmp != (chunk_t *) -1) {
                _bucket[bucket_id].writing_chunk = (chunk_t *) -1;
                pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
                if (tmp != NULL) {
                    if (tmp->size) {
                        release_sync(tmp->id);
                    } else if (_bucket[bucket_id].num_chunks > 0) {
                        if (_bucket[bucket_id].chunk_ids[_bucket[bucket_id].num_chunks - 1] == tmp->id) {
                            _bucket[bucket_id].num_chunks--;
                            release_del(tmp->id);
                        } else lg_err("last empty chunk %d not in the list of bucket %d", tmp->id, bucket_id);
                    } else lg_err("empty chunk %d not in the list of bucket %d", tmp->id, bucket_id);
                }
                break;
            }
            pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
        }

        bool staged_in_hbm = __stage_bucket_to_hbm(bucket_id);
        _bucket[bucket_id].status = l0_bucket_t::_bk_ready |
                                    (staged_in_hbm ? l0_bucket_t::_bk_hbm : 0);
        pthread_spin_lock(&_bwc_lock);
        if (_num_ready_buckets < bwc_max_buckets) {
            _ready_bucket_id[_num_ready_buckets++] = bucket_id;
        } else {
            lg_err("ready list full(%d), bucket %d discarded", _num_ready_buckets, bucket_id);
            _bucket[bucket_id].status = l0_bucket_t::_bk_reading |
                                        (staged_in_hbm ? l0_bucket_t::_bk_hbm : 0);
        }
        pthread_spin_unlock(&_bwc_lock);

        if (_bucket[bucket_id].status & l0_bucket_t::_bk_ready) {
            if (!(_bucket[bucket_id].status & l0_bucket_t::_bk_hbm))
                __prefetch_for_reading(bucket_id);
        } else {
            bucket_finalize(bucket_id);
        }
    } else lg_err("input bucket %d status(%x) not in reading or writing", bucket_id, _bucket[bucket_id].status);
    
    lg_exit();
    return;
}

template <class logger_t> long bwc_manager_tmpl<logger_t>::num_ready() {
    pthread_spin_lock(&_bwc_lock);
    long ret = _num_ready_buckets;
    for (int32_t i = 0; i < _num_prefetched_bucket; i++) {
        if (_bucket[_prefetched_bucket_id[i]].status & l0_bucket_t::_bk_ready) ret++;
    }
    pthread_spin_unlock(&_bwc_lock);

    return ret;
}

template <class logger_t> chunk_t *bwc_manager_tmpl<logger_t>::fetch_for_write(long bucket_id) {
    lg_init();
    constexpr uint32_t _bk_all_status = l0_bucket_t::_bk_writing | l0_bucket_t::_bk_ready |
                                        l0_bucket_t::_bk_reading | l0_bucket_t::_bk_caching;

    volatile int32_t *_num_wp_ptr_vol = reinterpret_cast<volatile int32_t*>(&_num_wp);
    chunk_t *volatile *writing_chunk_ptr_vol = reinterpret_cast<chunk_t * volatile *>(&_bucket[bucket_id].writing_chunk);
    
    #if ENABLE_PROFILING
    ev_f4w.fetch_add(1);
    int first_try = 0x2;
    #endif

    for (;;) {
        #if ENABLE_PROFILING
        first_try >>= 1;
        #endif
        if ((_bucket[bucket_id].status & _bk_all_status) != l0_bucket_t::_bk_writing) {
            lg_err("wrong input bucket(%ld) status(%x)", bucket_id, _bucket[bucket_id].status);
            lg_exit();
            return NULL;
        }

        asm volatile("" ::: "memory");

        if (_bucket[bucket_id].writing_chunk == (chunk_t *) -1) continue;

        if (_bucket[bucket_id].writing_chunk == NULL) {
            pthread_spin_lock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
            if (*writing_chunk_ptr_vol == NULL) {
                _bucket[bucket_id].writing_chunk = (chunk_t *) -1;
                pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
                chunk_t *ret = NULL;
                for (;;) {
                    if (_num_wp) {
                        pthread_spin_lock(&_bwc_wp_lock);
                        if (*_num_wp_ptr_vol) ret = _writing_prefetch_chunks[--_num_wp];
                        pthread_spin_unlock(&_bwc_wp_lock);
                    }
                    __prefetch_for_writing();
                    if (ret) break;
                    #if ENABLE_PROFILING
                    first_try >>= 1;
                    #endif
                }
                pthread_spin_lock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
                _bucket[bucket_id].add_chunk(ret->id);
                pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
                lg_exit();
                #if ENABLE_PROFILING
                if (first_try) ev_f4w_hit.fetch_add(1);
                #endif
                return ret;
            }
            pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);            
        }
        

        pthread_spin_lock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
        chunk_t *writing_chunk = *writing_chunk_ptr_vol;
        if (writing_chunk != NULL && writing_chunk != (chunk_t *) -1) {
            chunk_t *ret = writing_chunk;
            _bucket[bucket_id].writing_chunk = (chunk_t *) -1;
            pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
            lg_exit();
            #if ENABLE_PROFILING
            if (first_try) ev_f4w_hit.fetch_add(1);
            #endif
            return ret;
        }
        pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
    }
}

template <class logger_t> chunk_t *bwc_manager_tmpl<logger_t>::fetch_for_read(long bucket_id) {
    lg_init();
    if (_bucket[bucket_id].status & l0_bucket_t::_bk_hbm) {
        lg_exit();
        return NULL;
    }
    volatile uint32_t *_chunk_status_vol = reinterpret_cast<volatile uint32_t*>(_chunk_status);
    volatile int32_t *num_chunks_ptr_vol = reinterpret_cast<volatile int32_t*>(&_bucket[bucket_id].num_chunks);
    volatile uint32_t *status_ptr_vol = reinterpret_cast<volatile uint32_t*>(&_bucket[bucket_id].status);
    volatile int32_t *chunk_ids_vol = reinterpret_cast<volatile int32_t*>(_bucket[bucket_id].chunk_ids);

    constexpr uint32_t _bk_all_status = l0_bucket_t::_bk_writing | l0_bucket_t::_bk_ready |
                                        l0_bucket_t::_bk_reading;

    #if ENABLE_PROFILING
    ev_f4r.fetch_add(1);
    int first_try = 0x2;
    #endif

    for (;;) {
        #if ENABLE_PROFILING
        first_try >>= 1;
        #endif
        uint32_t status = _bucket[bucket_id].status;
        uint32_t num_chunks = _bucket[bucket_id].num_chunks;

        if ((status & _bk_all_status) != l0_bucket_t::_bk_reading) {
            lg_err("wrong input bucket(%ld) status(%x)", bucket_id, status);
            return NULL;
        }

        if (num_chunks == 0) {
            lg_exit();
            return NULL;
        }

        int32_t ret = -1;
        for (int32_t i = num_chunks - 1; i >= 0 && i >= num_chunks - bwc_auto_prefetch_for_read_depth; i--) {
            int32_t id = _bucket[bucket_id].chunk_ids[i];
            if ((_chunk_status[id] & (_ck_reading | _ck_caching)) == _ck_caching) {
                pthread_spin_lock(&_locks[id % pwc_locks]);
                if ((_chunk_status_vol[id] & (_ck_reading | _ck_caching)) == _ck_caching) _chunk_status[id] |= _ck_writing;
                else {
                    pthread_spin_unlock(&_locks[id % pwc_locks]);
                    continue;
                }
                pthread_spin_unlock(&_locks[id % pwc_locks]);

                pthread_spin_lock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
                if (i < *num_chunks_ptr_vol && status == *status_ptr_vol && id == chunk_ids_vol[i]) {
                    chunk_ids_vol[i] = chunk_ids_vol[--num_chunks_ptr_vol[0]];
                    ret = id;
                }
                pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
                if (ret != -1) break;
            }
        }

        __prefetch_for_reading(bucket_id);
        
        if (ret != -1) {
            lg_exit();
            #if ENABLE_PROFILING
            if (first_try) ev_f4r_hit.fetch_add(1);
            #endif
            return &_cached_chunks[_chunk_status[ret] & _ck_cache_id_mask];
        }
    }
}

template <class logger_t> void bwc_manager_tmpl<logger_t>::write_done(chunk_t *chunk, long bucket_id) {
    lg_init();
    volatile int32_t *_num_wp_ptr_vol = reinterpret_cast<volatile int32_t*>(&_num_wp);

    if (chunk->size == Pool_hd_t::chunk_max_nvecs) {
        release_sync(chunk->id);

        chunk_t *new_chunk = NULL;
        if (_num_wp) {
            pthread_spin_lock(&_bwc_wp_lock);
            if (*_num_wp_ptr_vol) {
                new_chunk = _writing_prefetch_chunks[--_num_wp_ptr_vol[0]];
            }
            pthread_spin_unlock(&_bwc_wp_lock);
        }
        __prefetch_for_writing();
        if (new_chunk) {
            pthread_spin_lock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
            _bucket[bucket_id].add_chunk(new_chunk->id);
            pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
            _bucket[bucket_id].writing_chunk = new_chunk;
        } else _bucket[bucket_id].writing_chunk = NULL;
    } else {
        _bucket[bucket_id].writing_chunk = chunk;
    }
    lg_exit();
}

template <class logger_t> void bwc_manager_tmpl<logger_t>::read_done(chunk_t *chunk, long bucket_id) {
    lg_init();
    release_del(chunk->id);
    lg_exit();
}

template <class logger_t> long bwc_manager_tmpl<logger_t>::bucket_num_chunks(long bucket_id) {
    pthread_spin_lock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
    long ret = (_bucket[bucket_id].status & l0_bucket_t::_bk_gpu_write) ?
        (_bucket[bucket_id].hbm_nvecs + Pool_hd_t::chunk_max_nvecs - 1) /
            Pool_hd_t::chunk_max_nvecs :
        _bucket[bucket_id].num_chunks;
    pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
    return ret;
}

template <class logger_t> int bwc_manager_tmpl<logger_t>::mpi_export_bucket(
        long bucket_id, std::vector<uint8_t> &records, int record_size) {
    if (bucket_id < 0 || bucket_id >= _num_buckets ||
        record_size != (int)(sizeof(int32_t) + this->_pool->CSD)) return -1;
    l0_bucket_t &bucket = _bucket[bucket_id];
    if (bucket.status & l0_bucket_t::_bk_gpu_write) {
        fprintf(stderr, "[MPI] GPU-native BWC must be disabled for remote buckets\n");
        return -1;
    }
    if (!(bucket.status & l0_bucket_t::_bk_writing)) return -1;

    // Close the last partial chunk exactly as bucket_finalize(), but mark the
    // bucket reading without putting it on the reducer-visible ready list.
    for (;;) {
        if (bucket.writing_chunk == (chunk_t *)-1) continue;
        pthread_spin_lock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
        chunk_t *last = bucket.writing_chunk;
        if (last == (chunk_t *)-1) {
            pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
            continue;
        }
        bucket.writing_chunk = (chunk_t *)-1;
        pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
        if (last) {
            if (last->size) release_sync(last->id);
            else {
                pthread_spin_lock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
                if (bucket.num_chunks &&
                    bucket.chunk_ids[bucket.num_chunks - 1] == last->id)
                    --bucket.num_chunks;
                pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
                release_del(last->id);
            }
        }
        break;
    }
    pthread_spin_lock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
    bucket.status = l0_bucket_t::_bk_reading;
    pthread_spin_unlock(&_bucket_lock[bucket_id % bwc_bucket_locks]);
    __prefetch_for_reading(bucket_id);

    for (;;) {
        chunk_t *chunk = fetch_for_read(bucket_id);
        if (!chunk) break;
        const std::size_t old = records.size();
        records.resize(old + (std::size_t)chunk->size * record_size);
        uint8_t *dst = records.data() + old;
        for (int i = 0; i < chunk->size; ++i) {
            memcpy(dst, chunk->norm + i, sizeof(int32_t));
            memcpy(dst + sizeof(int32_t),
                   chunk->vec + (long)this->_pool->CSD * i, this->_pool->CSD);
            dst += record_size;
        }
        read_done(chunk, bucket_id);
    }
    bucket_finalize(bucket_id);
    return 0;
}

template <class logger_t> int bwc_manager_tmpl<logger_t>::mpi_append_bucket(
        long bucket_id, const uint8_t *records, long count, int record_size) {
    if (count < 0 || record_size != (int)(sizeof(int32_t) + this->_pool->CSD))
        return -1;
    if (count == 0) return 0;
    if (!records) return -1;
    long pos = 0;
    while (pos < count) {
        chunk_t *dst = fetch_for_write(bucket_id);
        if (!dst) return -1;
        const int n = std::min<long>(count - pos, Pool_hd_t::chunk_max_nvecs - dst->size);
        for (int i = 0; i < n; ++i) {
            const uint8_t *src = records + (pos + i) * (long)record_size;
            memcpy(dst->norm + dst->size + i, src, sizeof(int32_t));
            memcpy(dst->vec + (long)this->_pool->CSD * (dst->size + i),
                   src + sizeof(int32_t), this->_pool->CSD);
        }
        dst->size += n;
        pos += n;
        write_done(dst, bucket_id);
    }
    return 0;
}

template <class logger_t> swc_manager_tmpl<logger_t>::swc_manager_tmpl(Pool_hd_t *p) : 
                          pwc_manager_tmpl<logger_t>(swc_default_loading_threads,
                                                     swc_default_syncing_threads,
                                                     __swc_runtime_max_cached_chunks(p)) {
    const long requested_chunks = _max_cached_chunks;
    const long arena_chunks = _ensure_regular_chunk_capacity(
        PWC_DEFAULT_MAX_CACHED_CHUNKS + requested_chunks + 256);
    const long supported_chunks = arena_chunks - PWC_DEFAULT_MAX_CACHED_CHUNKS - 256;
    if (supported_chunks < _max_cached_chunks)
        this->set_max_cached_chunks(supported_chunks > 0 ? supported_chunks
                                                         : swc_default_max_cached_chunks);
    _num_ready = 0;
    _num_writing = 0;
    _num_rp = 0; 
    _num_rl = 0;
    _num_wp = 0;
    _num_wl = 0;
    _ready_chunks = (int32_t *) malloc(swc_max_ready_chunks * sizeof(int32_t));
    _writing_chunks = (chunk_t **) malloc(_max_cached_chunks * sizeof(chunk_t *));
    if (!_ready_chunks || !_writing_chunks) {
        fprintf(stderr, "[Error] SWC metadata allocation failed\n");
        abort();
    }
    pthread_spin_init(&_swc_lock, PTHREAD_PROCESS_SHARED);

    this->_pool = p;
    this->set_dirname("sol");

    if (_max_cached_chunks > swc_default_max_cached_chunks) {
        const double extra_gib = (_max_cached_chunks - swc_default_max_cached_chunks) *
                                 Pool_hd_t::chunk_max_nvecs *
                                 (double)POOL_HOST_VEC_SLOT_NBYTES / (1ULL << 30);
        lg_warn("guarded SWC growth enabled at CSD %ld: +%.2f GiB (%ld chunks total)",
                p->CSD, extra_gib, _max_cached_chunks);
    } else if (requested_chunks > swc_default_max_cached_chunks) {
        lg_warn("guarded SWC growth unavailable at CSD %ld; using base capacity", p->CSD);
    }

    lg_info("manager initialized, (%d, %d) threads for I/O, #caching = %d", 
            _loading_threads, _syncing_threads, _max_cached_chunks);
}

template <class logger_t> swc_manager_tmpl<logger_t>::~swc_manager_tmpl() {
    constexpr chunk_status_t _ck_busy = _ck_writing | _ck_reading | _ck_loading;
    this->_syncing_pool.wait_work();
    this->_loading_pool.wait_work();

    pthread_spin_lock(&_swc_lock);
    if (_num_rl || _num_wl) lg_err("still %d(4r) + %d(4w) chunks loading?", _num_rl, _num_wl);
    int32_t num_ready = _num_ready;
    int32_t num_writing = _num_writing;
    int32_t num_rp = _num_rp;
    int32_t num_wp = _num_wp;
    _num_ready = 0;
    _num_writing = 0;
    _num_rp = 0;
    _num_wp = 0;
    _num_rl = swc_auto_prefetch_for_read;
    _num_wl = swc_auto_prefetch_for_write;
    pthread_spin_unlock(&_swc_lock);

    for (long i = 0; i < num_rp; i++) release_del(_reading_prefetch_chunks[i]->id);
    for (long i = 0; i < num_wp; i++) release_del(_writing_prefetch_chunks[i]->id);
    for (long i = 0; i < num_writing; i++) release_del(_writing_chunks[i]->id);
    for (long i = 0; i < num_ready; i++) {
        int32_t id = _ready_chunks[i];
        if (_chunk_status[id] & _ck_busy) lg_err("ready chunk %d status %x", id, _chunk_status[id]);
        pthread_spin_lock(&_locks[id % pwc_locks]);
        _chunk_status[_ready_chunks[i]] |= _ck_writing;
        pthread_spin_unlock(&_locks[id % pwc_locks]);
        release_del(id);
    }

    free(_ready_chunks);
    free(_writing_chunks);
    pthread_spin_destroy(&_swc_lock);

    this->_syncing_pool.wait_work();
    this->_loading_pool.wait_work();
    
    /// delete all files
    char chunk_filename[256];
    
    for (long i = 0; i < _num_chunks; i++) {
        #if MULTI_SSD
        snprintf(chunk_filename, sizeof(chunk_filename), "%s/%s/%s%06lx", this->_dir, hw::ssd_name(i), this->_pfx, i);
        #else
        snprintf(chunk_filename, sizeof(chunk_filename), "%s%06lx", this->_prefix, i);
        #endif
        remove(chunk_filename);
    }
}

template <class logger_t> void swc_manager_tmpl<logger_t>::__prefetch_for_writing() {
    lg_init();
    volatile int32_t *_num_wp_ptr_vol = reinterpret_cast<volatile int32_t*>(&_num_wp);
    volatile int32_t *_num_wl_ptr_vol = reinterpret_cast<volatile int32_t*>(&_num_wl);

    if (_num_writing + _num_wl + _num_wp >= _max_cached_chunks) {
        lg_exit(); 
        return; 
    }
    if (_num_wl + _num_wp >= swc_auto_prefetch_for_write) { 
        lg_exit(); 
        return; 
    }

    pthread_spin_lock(&_swc_lock);
    int to_prefetch = swc_auto_prefetch_for_write - _num_wl_ptr_vol[0] - _num_wp_ptr_vol[0];
    if (to_prefetch > _max_cached_chunks - _num_writing - _num_wl - _num_wp)
        to_prefetch = _max_cached_chunks - _num_writing - _num_wl - _num_wp;
    if (to_prefetch <= 0) {
        pthread_spin_unlock(&_swc_lock);
        lg_exit();
        return;
    }
    _num_wl += to_prefetch;
    pthread_spin_unlock(&_swc_lock);

    int32_t num_fail = 0;
    for (int32_t i = 0; i < to_prefetch; i++) {
        int32_t id = create_chunk();
        pthread_spin_lock(&_locks[id % pwc_locks]);
        if (_chunk_status[id]) {
            num_fail++;
        } else {
            _chunk_status[id] |= _ck_loading | _ck_writing;
            this->_loading_pool.push([=]() {
                __load_chunk(id);
                pthread_spin_lock(&_swc_lock);
                _writing_prefetch_chunks[_num_wp++] = &_cached_chunks[_chunk_status[id] & _ck_cache_id_mask];
                _num_wl--;
                pthread_spin_unlock(&_swc_lock);
            });
            _num_loading_chunks++;
        }
        pthread_spin_unlock(&_locks[id % pwc_locks]);
    }

    if (num_fail) {
        lg_err("%d of %d new chunk already in use?", num_fail, to_prefetch);
        pthread_spin_lock(&_swc_lock);
        _num_wl -= num_fail;
        pthread_spin_unlock(&_swc_lock);
    }
    lg_exit();
}

template <class logger_t> void swc_manager_tmpl<logger_t>::__prefetch_for_reading() {
    lg_init();
    volatile int32_t *_num_rp_ptr_vol = reinterpret_cast<volatile int32_t*>(&_num_rp);
    volatile int32_t *_num_rl_ptr_vol = reinterpret_cast<volatile int32_t*>(&_num_rl);

    if (_num_ready == 0 || _num_rp + _num_rl >= swc_auto_prefetch_for_read) {
        lg_exit();
        return;
    }

    constexpr chunk_status_t _ck_busy = _ck_reading | _ck_writing | _ck_loading;
    constexpr int32_t max_search = 64;
    int32_t num_possible_ids = 0;
    int32_t possible_ids[max_search];
    do {
        int exp_prefetch = swc_auto_prefetch_for_read - _num_rp - _num_rl;
        if (2 * exp_prefetch > _num_ready || exp_prefetch <= 0) break;
        for (int32_t i = _num_ready - 1; i >= 0 && i >= _num_ready - max_search; i--) {
            int id = _ready_chunks[i];
            if (id >= _num_chunks || id < 0) continue;
            if ((_chunk_status[id] & (_ck_caching | _ck_busy )) == _ck_caching) {
                possible_ids[num_possible_ids++] = i;
                if (num_possible_ids >= 2 * exp_prefetch || num_possible_ids >= max_search) break;
            }
        }
    } while (0);

    pthread_spin_lock(&_swc_lock);
    int to_prefetch = swc_auto_prefetch_for_read - _num_rp_ptr_vol[0] - _num_rl_ptr_vol[0];
    if (to_prefetch > _num_ready) to_prefetch = _num_ready;
    if (to_prefetch <= 0) {
        pthread_spin_unlock(&_swc_lock);
        lg_exit();
        return;
    }
    _num_rl += to_prefetch;

    int32_t to_prefetch_ids[swc_auto_prefetch_for_read];
    do {
        int32_t i = 0;
        while (num_possible_ids) {
            int32_t ptr = possible_ids[--num_possible_ids];
            if (ptr >= _num_ready) continue;
            if ((_chunk_status[_ready_chunks[ptr]] & (_ck_busy | _ck_caching)) == _ck_caching) {
                to_prefetch_ids[i++] = _ready_chunks[ptr];
                _ready_chunks[ptr] = _ready_chunks[--_num_ready];
                if (i >= to_prefetch) break;
            }
        }

        _num_ready -= to_prefetch - i;
        for (int32_t j = 0; j < to_prefetch - i; j++) {
            to_prefetch_ids[i + j] = _ready_chunks[_num_ready + j];
        }
    } while (0);
    pthread_spin_unlock(&_swc_lock);

    int32_t num_fail = 0;
    for (int32_t i = 0; i < to_prefetch; i++) {
        int32_t id = to_prefetch_ids[i];
        if (id < 0 || id >= _num_chunks) {
            lg_err("invalid chunk id %d, %d/%d", id, i, to_prefetch);
            num_fail++;
            continue;
        }
        pthread_spin_lock(&_locks[id % pwc_locks]);
        if (_chunk_status[id] & _ck_busy) {
            lg_err("chunk(%d) in ready list is busy(%x)", id, _chunk_status[id]);
            num_fail++;
        } else if (_chunk_status[id] == _ck_size_mask) {
            lg_err("chunk(%d) in ready list is deleted", id);
            num_fail++;
        } else if (_chunk_status[id] & _ck_caching) {
            _chunk_status[id] |= _ck_writing;
            pthread_spin_unlock(&_locks[id % pwc_locks]);

            pthread_spin_lock(&_swc_lock);
            _reading_prefetch_chunks[_num_rp++] = &_cached_chunks[_chunk_status[id] & _ck_cache_id_mask];
            _num_rl--;
            pthread_spin_unlock(&_swc_lock);
            continue;
        } else {
            _chunk_status[id] |= _ck_loading | _ck_writing;
            this->_loading_pool.push([=]() {
                __load_chunk(id);
                pthread_spin_lock(&_swc_lock);
                _reading_prefetch_chunks[_num_rp++] = &_cached_chunks[_chunk_status[id] & _ck_cache_id_mask];
                _num_rl--;
                pthread_spin_unlock(&_swc_lock);
            });
            _num_loading_chunks++;
        }
        pthread_spin_unlock(&_locks[id % pwc_locks]);
    }

    if (num_fail) {
        pthread_spin_lock(&_swc_lock);
        _num_rl -= num_fail;
        pthread_spin_unlock(&_swc_lock);
    }

    lg_exit();
}

template <class logger_t> void swc_manager_tmpl<logger_t>::chunk_finalize(chunk_t *chunk) {
    lg_init();
    volatile int32_t *_num_rp_ptr_vol = reinterpret_cast<volatile int32_t*>(&_num_rp);
    volatile int32_t *_num_rl_ptr_vol = reinterpret_cast<volatile int32_t*>(&_num_rl);
    volatile int32_t *_num_ready_ptr_vol = reinterpret_cast<volatile int32_t*>(&_num_ready);

    constexpr chunk_status_t _ck_busy = _ck_writing | _ck_reading | _ck_loading;

    if (_num_rp + _num_rl < swc_auto_prefetch_for_read) {
        pthread_spin_lock(&_swc_lock);
        if (_num_rp_ptr_vol[0] + _num_rl_ptr_vol[0] < swc_auto_prefetch_for_read) {
            _reading_prefetch_chunks[_num_rp++] = chunk;
            pthread_spin_unlock(&_swc_lock);
            lg_exit();
            return;
        } 
        pthread_spin_unlock(&_swc_lock);
    }

    int id = chunk->id;
    release_sync(chunk->id);

    if (_num_ready < swc_max_ready_chunks) {
        pthread_spin_lock(&_swc_lock);
        if (_num_ready_ptr_vol[0] < swc_max_ready_chunks) {
            _ready_chunks[_num_ready++] = id;
            pthread_spin_unlock(&_swc_lock);
            lg_exit();
            return;
        }
        pthread_spin_unlock(&_swc_lock);
    }
    
    pthread_spin_lock(&_locks[id % pwc_locks]);
    _chunk_status[id] |= _ck_writing;
    pthread_spin_lock(&_locks[id % pwc_locks]);
    release_del(id);
    lg_err("ready list full(%d), new chunk discarded", _num_ready);
    lg_exit();
    return;
}

template <class logger_t> chunk_t *swc_manager_tmpl<logger_t>::fetch_for_write() {
    lg_init();
    volatile int32_t *_num_wp_ptr_vol = reinterpret_cast<volatile int32_t*>(&_num_wp);
    volatile int32_t *_num_writing_ptr_vol = reinterpret_cast<volatile int32_t*>(&_num_writing);
    
    chunk_t *ret = NULL;

    #if ENABLE_PROFILING
    ev_f4w.fetch_add(1);
    int first_try = 0x2;
    #endif

    while (ret == NULL) {
        #if ENABLE_PROFILING
        first_try >>= 1;
        #endif
        if (_num_writing || _num_wp) {
            pthread_spin_lock(&_swc_lock);
            if (*_num_writing_ptr_vol) ret = _writing_chunks[--_num_writing];
            else if (*_num_wp_ptr_vol) ret = _writing_prefetch_chunks[--_num_wp];
            pthread_spin_unlock(&_swc_lock);
        }

        __prefetch_for_writing();
    }

    #if ENABLE_PROFILING
    if (first_try) ev_f4w_hit.fetch_add(1);
    #endif

    lg_exit();
    return ret;
}

template <class logger_t> chunk_t *swc_manager_tmpl<logger_t>::fetch_for_read() {
    lg_init();
    volatile int32_t *_num_rp_ptr_vol = reinterpret_cast<volatile int32_t*>(&_num_rp);

    #if ENABLE_PROFILING
    ev_f4r.fetch_add(1);
    int first_try = 0x2;
    #endif

    chunk_t *ret = NULL;

    while (ret == NULL) {
        #if ENABLE_PROFILING
        first_try >>= 1;
        #endif
        if (_num_rp) {
            pthread_spin_lock(&_swc_lock);
            if (*_num_rp_ptr_vol) ret = _reading_prefetch_chunks[--_num_rp];
            pthread_spin_unlock(&_swc_lock);
        }

        if (_num_ready) __prefetch_for_reading();
        if (_num_rl == 0 && _num_rp == 0 && _num_ready == 0) break;
    }

    #if ENABLE_PROFILING
    if (first_try) ev_f4r_hit.fetch_add(1);
    #endif

    lg_exit();
    return ret;
}

template <class logger_t> void swc_manager_tmpl<logger_t>::write_done(chunk_t *chunk) {
    lg_init();
    if (chunk->size < Pool_hd_t::chunk_max_nvecs || !swc_auto_finalize) {
        pthread_spin_lock(&_swc_lock);
        if (_num_writing < _max_cached_chunks) {
            _writing_chunks[_num_writing++] = chunk;
            pthread_spin_unlock(&_swc_lock);
            lg_exit();
            return;
        }
        pthread_spin_unlock(&_swc_lock);
    }

    chunk_finalize(chunk);
    lg_exit();
}

template <class logger_t> void swc_manager_tmpl<logger_t>::read_done(chunk_t *chunk) {
    lg_init();
    release_del(chunk->id);
    lg_exit();
}

template <class logger_t> long swc_manager_tmpl<logger_t>::num_ready() {
    pthread_spin_lock(&_swc_lock);
    long ret = _num_ready + _num_rp + _num_rl;
    pthread_spin_unlock(&_swc_lock);
    return ret;
}

template <class logger_t> long swc_manager_tmpl<logger_t>::ready_nvecs_estimate() {
    long ret = 0;

    pthread_spin_lock(&_swc_lock);
    for (int i = 0; i < _num_rp; i++) ret += _reading_prefetch_chunks[i]->size;
    for (int i = 0; i < _num_ready; i++) {
        int size = 0;
        int id = _ready_chunks[i];
        if (id >= 0 && id < _num_chunks) size = chunk_size(id);
        if (size <= Pool_hd_t::chunk_max_nvecs) ret += size;
    }
    pthread_spin_unlock(&_swc_lock);

    return ret;
}

template <class logger_t> long swc_manager_tmpl<logger_t>::num_using() {
    pthread_spin_lock(&_swc_lock);
    long ret = _num_chunks - _num_deleted_ids - _num_writing;
    pthread_spin_unlock(&_swc_lock);

    return ret;
}

#if 0
template <class logger_t> long swc_manager_tmpl<logger_t>::finalize_all_writing() {
    lg_init();
    int to_finalize_malloc_size = _num_writing + swc_auto_prefetch_for_write;
    int num_to_finalize = 0;
    chunk_t **to_finalize = (chunk_t **) malloc(to_finalize_malloc_size * sizeof(chunk_t *));
    pthread_spin_lock(&_swc_lock);
    if (_num_writing + swc_auto_prefetch_for_write > to_finalize_malloc_size) {
        lg_err("# writing chunks increased while calling finalize_all_writing, no chunk finalized");
        pthread_spin_unlock(&_swc_lock);
        free(to_finalize);
        lg_exit();
        return -1;
    }
    for (int i = 0; i < _num_writing; i++) to_finalize[num_to_finalize++] = _writing_chunks[i];
    _num_writing = 0;
    for (int i = 0; i < _num_wp; i++) to_finalize[num_to_finalize++] = _writing_prefetch_chunks[i];
    _num_wp = 0;
    pthread_spin_unlock(&_swc_lock);

    for (int i = 0; i < num_to_finalize; i++) {
        if (to_finalize[i]->size == 0) release_del(to_finalize[i]->id);
        else chunk_finalize(to_finalize[i]);
    }

    lg_exit();
    return 0;
}
#endif

#if ENABLE_PROFILING
template struct bwc_manager_tmpl<bwc_logger_t>;
template struct swc_manager_tmpl<swc_logger_t>;
#else
template struct bwc_manager_tmpl<int>;
template struct swc_manager_tmpl<int>;
#endif
