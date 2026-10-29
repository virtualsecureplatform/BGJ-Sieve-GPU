// Quiescent transition only. Transport progress never depends on a sieve worker.
static int redistribute_bounded(Pool_hd_t *pool) {
    if(pool->mpi_compact_dense()) return -1;
    const long original=pool->pwc_manager->num_chunks();
    uint64_t owners[2]={}, totals[2]={};
    for(long id=0;id<original;++id) {
        auto *chunk=pool->pwc_manager->fetch(id);
        if(!chunk) return -1;
        for(int i=0;i<chunk->size;++i) ++owners[pool->uid_table->normalize(chunk->u[i])%2];
        pool->pwc_manager->release(id);
    }
    MPI_Allreduce(owners,totals,2,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
    const long needed=std::max(original,long((totals[g_rank]+Pool_hd_t::chunk_max_nvecs-1)/Pool_hd_t::chunk_max_nvecs))+2;
    const long available=std::min(long(PWC_DEFAULT_MAX_CACHED_CHUNKS),_regular_chunk_capacity()-256-8*pool->num_threads());
    if(!mpi_sieve_min_int(needed<=available?1:0)) {
        fprintf(stderr,"[SHARED-ARENA] redistribution infeasible rank=%d need=%ld available=%ld\n",g_rank,needed,available);
        return -1;
    }
    pool->pwc_manager->rebalance_cache_quiescent(needed);
    const int stride=pool->CSD+14;
    const size_t quota=std::max(size_t(1),std::min(g_frame_bytes,size_t(64ULL<<20))/stride);
    long cid=0; int position=0, end=0;
    uint64_t epoch=0, exported=0, received=0, duplicates=0;
    for(;;) {
        std::vector<uint8_t> outgoing; outgoing.reserve(quota*stride);
        while(cid<original && outgoing.size()/stride<quota) {
            auto *chunk=pool->pwc_manager->fetch(cid);
            if(!chunk) return -1;
            if(position==0) end=chunk->size;
            while(position<end && outgoing.size()/stride<quota) {
                const int i=position++;
                if(pool->uid_table->normalize(chunk->u[i])%2==uint64_t(g_rank)) continue;
                const size_t offset=outgoing.size(); outgoing.resize(offset+stride);
                auto *dst=outgoing.data()+offset;
                memcpy(dst,chunk->u+i,8); memcpy(dst+8,chunk->norm+i,4);
                memcpy(dst+12,chunk->score+i,2); memcpy(dst+14,chunk->vec+long(i)*pool->CSD,pool->CSD);
                if(pool->uid_table->erase(chunk->u[i])!=1) {pool->pwc_manager->release(cid);return -1;}
                --pool->score_stat[chunk->score[i]];
                chunk->score[i]=0; chunk->norm[i]=0; chunk->u[i]=0; --chunk->size;
            }
            const bool finished=position==end;
            if(finished) _normalize_chunk(chunk,pool->CSD);
            pool->pwc_manager->release_sync(cid);
            if(finished) {++cid;position=0;}
        }
        std::vector<uint8_t> incoming;
        if(exchange_bytes(outgoing.data(),outgoing.size(),incoming,tag_pool_count,tag_pool_data,pool->CSD,epoch++)) return -1;
        if(incoming.size()%stride) return -1;
        const uint64_t before=pool->pwc_manager->num_vec();
        pool->mpi_reset_append_hint();
        // Never normalize or append into the chunk whose export scan is paused.
        if(pool->mpi_append_records(incoming.data(),incoming.size()/stride,stride,cid<original?cid:-1)) return -1;
        const uint64_t inserted=pool->pwc_manager->num_vec()-before;
        exported+=outgoing.size()/stride;received+=incoming.size()/stride;
        duplicates+=incoming.size()/stride-inserted;
        if(pool->pwc_manager->allocated_cache_chunks()>needed) return -1;
        if(mpi_sieve_min_int(cid==original?1:0)) break;
    }
    if(pool->mpi_retain_owner(g_rank,g_world) || pool->mpi_compact_dense()) return -1;
    pool->pwc_manager->rebalance_cache_quiescent(std::max(1L,pool->pwc_manager->num_chunks()));
    printf("[BOUNDED-REDISTRIBUTE] rank=%d original_chunks=%ld cap=%ld exported=%llu received=%llu duplicates=%llu final_vectors=%ld\n",
        g_rank,original,needed,(unsigned long long)exported,(unsigned long long)received,
        (unsigned long long)duplicates,pool->pwc_manager->num_vec()); fflush(stdout);
    return 0;
}
