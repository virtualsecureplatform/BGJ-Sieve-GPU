#ifndef BGJ_SIEVE_CACHE_BUDGET_H
#define BGJ_SIEVE_CACHE_BUDGET_H

#include <algorithm>
#include <stdexcept>
#include <cstdlib>
#include <cstring>

inline bool bounded_shared_arena() {
    const char *p=std::getenv("HD_MPI_SHARED_ARENA_POLICY");
    if(p && std::strcmp(p,"legacy") && std::strcmp(p,"bounded"))
        throw std::invalid_argument("invalid HD_MPI_SHARED_ARENA_POLICY");
    return p && !std::strcmp(p,"bounded");
}
inline long bounded_solution_slots(long arena, long pool, long auxiliary, long preferred) {
    if(pool<0 || auxiliary<0 || pool>arena || auxiliary>arena-pool || arena-pool-auxiliary<512)
        throw std::runtime_error("shared arena cannot hold pool and minimum solution queue");
    return std::min(preferred,arena-pool-auxiliary);
}

// PWC and SWC allocate from the same regular-chunk arena. At a fixed sieve
// dimension the pool cannot grow beyond its iterator target (or the existing
// pool, if larger). Lend only that phase's unused PWC reservation to SWC.
// This does not increase the combined host budget or change the sieve target.
inline long no_spill_swc_cache_chunks(long pool_capacity, long base_swc,
                                      long phase_pool_chunks, long headroom) {
    const long unused_pool = std::max(0L, pool_capacity - phase_pool_chunks);
    const long desired_swc = phase_pool_chunks + headroom;
    return base_swc + std::min(unused_pool, std::max(0L, desired_swc - base_swc));
}

#endif
