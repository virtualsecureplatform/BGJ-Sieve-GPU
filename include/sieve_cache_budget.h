#ifndef BGJ_SIEVE_CACHE_BUDGET_H
#define BGJ_SIEVE_CACHE_BUDGET_H

#include <algorithm>

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
