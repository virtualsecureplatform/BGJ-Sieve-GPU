// Included inside mpi_bkz after the legacy exact reader. No MPI calls in workers.
void restore_pool_fused(Pool_hd_t &pool, const state_t &state, const fs::path &source) {
    const auto started = std::chrono::steady_clock::now();
    const auto before_io = process_io();
    const int rank = mpi_sieve_rank();
    live_csd.store(state.csd);
    auto workers = [&](const char *key, int fallback) {
        const char *value = getenv(key);
        const int n = value ? atoi(value) : fallback;
        require(n >= 1 && n <= 8, std::string(key)+" must be in [1,8]");
        return n;
    };
    const int io_threads = workers("HD_MPI_RESTORE_IO_THREADS",4);
    const int uid_threads = workers("HD_MPI_RESTORE_UID_THREADS",8);
    struct item { fs::path relative; uint64_t bytes, hash; long id; };
    std::vector<item> items;
    std::ifstream manifest(source/"FILES");
    std::string name; uint64_t bytes, hash;
    const std::string prefix = pool.pwc_manager->pfx();
    while (manifest >> std::quoted(name) >> bytes >> hash) {
        fs::path relative(name);
        if (relative.begin()->string() != "pool") continue;
        const std::string leaf = relative.filename().string();
        require(leaf.size()==prefix.size()+6 && leaf.rfind(prefix,0)==0,
                "invalid fused restore chunk name");
        const std::string suffix=leaf.substr(prefix.size());
        require(suffix.find_first_not_of("0123456789abcdef")==std::string::npos,
                "invalid fused restore chunk id");
        const long id=std::stol(suffix,nullptr,16);
        require(relative==fs::path("pool")/hw::ssd_name(id)/leaf, "wrong restore shard");
        const uint64_t expected=(ONE_TIME_IO?4096:12)+Pool_hd_t::chunk_max_nvecs*uint64_t(14+pool.CSD);
        require(bytes==expected, "invalid fused restore chunk size");
        items.push_back({relative,bytes,hash,id});
    }
    require(manifest.eof(), "invalid fused restore manifest");
    std::sort(items.begin(),items.end(),[](const item &a,const item &b){return a.id<b.id;});
    std::set<fs::path> expected_paths;
    uint64_t source_bytes=0;
    for(size_t i=0;i<items.size();++i) {
        require(items[i].id==long(i), "duplicate/noncontiguous restore id");
        expected_paths.insert(items[i].relative); source_bytes+=items[i].bytes;
    }
    size_t actual=0;
    for(const auto &entry:fs::recursive_directory_iterator(source/"pool")) {
        require(!entry.is_symlink(), "snapshot pool symlink");
        if(entry.is_directory()) continue;
        require(entry.is_regular_file() && expected_paths.count(entry.path().lexically_relative(source)),
                "unlisted snapshot pool entry");
        ++actual;
    }
    require(actual==items.size() && !items.empty(), "missing snapshot chunks");
    require(items.size()<=size_t(pool.pwc_manager->max_cached_chunks()), "restore exceeds host cache");
    require(_ensure_regular_chunk_capacity(items.size()+256)>=long(items.size()+256), "restore arena capacity");
    std::atomic<uint64_t> completed{0}, vectors{0}, logical_read{0}, logical_write{0};
    std::atomic<int> stage{0};
    std::mutex mutex; std::condition_variable cv; bool done=false;
    struct monitor_t {
        std::thread thread; std::mutex &mutex; std::condition_variable &cv; bool &done;
        ~monitor_t(){ {std::lock_guard<std::mutex> lock(mutex);done=true;} cv.notify_all();thread.join(); }
    } monitor{std::thread([&]{
        std::unique_lock<std::mutex> lock(mutex);
        while(!cv.wait_for(lock,std::chrono::seconds(30),[&]{return done;})) {
            const auto io=process_io();
            printf("[MPI-RESTORE] rank=%d stage=%d chunks=%llu/%zu vectors=%llu logical_read=%llu logical_write=%llu physical_read=%llu physical_write=%llu\n",
                rank,stage.load(),(unsigned long long)completed.load(),items.size(),
                (unsigned long long)vectors.load(),(unsigned long long)logical_read.load(),
                (unsigned long long)logical_write.load(),(unsigned long long)(io.first-before_io.first),
                (unsigned long long)(io.second-before_io.second)); fflush(stdout);
        }
    }),mutex,cv,done};
    pool.pwc_manager->prepare_exact_restore(items.size());
    std::vector<std::atomic<uint64_t>> histogram(UidTable::ut_split);
    for(auto &v:histogram) v.store(0);
    memset(pool.score_stat,0,sizeof(pool.score_stat));
    stage.store(1); // read, hash, scratch copy, cache population
    checkpoint_parallel(items.size(),[&](size_t index){
        const auto &item=items[index];
        std::vector<char> data(item.bytes);
        const int fd=open((source/item.relative).c_str(),O_RDONLY|O_NOFOLLOW);
        require(fd>=0,"open restore source");
        struct close_t {int fd; ~close_t(){close(fd);}} guard{fd};
        size_t offset=0;
        while(offset<data.size()) {
            const ssize_t n=read(fd,data.data()+offset,data.size()-offset);
            if(n<0 && errno==EINTR) continue;
            require(n>0,"short restore read"); offset+=n;
        }
        char extra; require(read(fd,&extra,1)==0,"trailing restore data");
        require(hash_bytes(1469598103934665603ULL,data.data(),data.size())==item.hash,"restore checksum mismatch");
        logical_read+=data.size();
        uint16_t count; uint64_t basis_hash;
        memcpy(&count,data.data(),2); memcpy(&basis_hash,data.data()+2,8);
        require(count>0 && count<=Pool_hd_t::chunk_max_nvecs && basis_hash==state.basis_hash &&
                uint8_t(data[10])==state.left && uint8_t(data[11])==state.right,"restore chunk context mismatch");
        auto *chunk=pool.pwc_manager->exact_restore_slot(item.id);
        size_t pos=12;
        auto copy=[&](void *to,size_t n){memcpy(to,data.data()+pos,n);pos+=n;};
        copy(chunk->score,Pool_hd_t::chunk_max_nvecs*2);
        copy(chunk->norm,Pool_hd_t::chunk_max_nvecs*4);
        copy(chunk->u,Pool_hd_t::chunk_max_nvecs*8);
        copy(chunk->vec,Pool_hd_t::chunk_max_nvecs*pool.CSD);
        for(long i=0;i<Pool_hd_t::chunk_max_nvecs;++i) {
            require((chunk->score[i]!=0)==(i<count),"sparse restored records");
            if(i>=count) continue;
            const uint64_t uid=pool.uid_table->normalize(chunk->u[i]);
            require(uid%2==uint64_t(rank),"wrong restore UID owner");
            histogram[uid%UidTable::ut_split].fetch_add(1,std::memory_order_relaxed);
        }
        const auto destination=fs::path(".pool")/item.relative.lexically_relative("pool");
        std::ofstream out(destination,std::ios::binary|std::ios::trunc);
        out.write(data.data(),data.size()); out.close(); require(bool(out),"restore scratch write failed");
        logical_write+=data.size(); chunk->size=count; vectors+=count; ++completed;
    },io_threads);
    require(vectors.load()==state.local_vectors,"restore vector count mismatch");
    stage.store(2); // reserve UID shards, then insert in parallel
    checkpoint_parallel(UidTable::ut_split,[&](size_t shard){
        pool.uid_table->_tables[shard]->reserve(pool.uid_table->_size[shard]+histogram[shard].load());
    },uid_threads);
    stage.store(3);
    checkpoint_parallel(items.size(),[&](size_t index){
        auto *chunk=pool.pwc_manager->exact_restore_slot(index);
        for(long i=0;i<chunk->size;++i) {
            require(pool.uid_table->insert(chunk->u[i])==1,"duplicate restored UID");
            __atomic_fetch_add(&pool.score_stat[chunk->score[i]],1,__ATOMIC_RELAXED);
        }
    },uid_threads);
    pool.pwc_manager->publish_exact_restore();
    const auto io=process_io();
    printf("[MPI-RESTORE] rank=%d stage=loaded seconds=%.6f chunks=%zu vectors=%llu source_bytes=%llu logical_read=%llu logical_write=%llu physical_read=%llu physical_write=%llu\n",
        rank,std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count(),items.size(),
        (unsigned long long)vectors.load(),(unsigned long long)source_bytes,(unsigned long long)logical_read.load(),
        (unsigned long long)logical_write.load(),(unsigned long long)(io.first-before_io.first),
        (unsigned long long)(io.second-before_io.second)); fflush(stdout);
}
