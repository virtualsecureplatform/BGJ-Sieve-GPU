#ifndef __CONFIG_H
#define __CONFIG_H

///////////////// hardware config /////////////////
#define MULTI_SSD 1
#define MULTI_GPU 1
struct hw {
    static constexpr int ssd_num = 2;
    static constexpr const char *ssd_name_list[ssd_num] = {"0", "1"};
    static inline const char *ssd_name(int chunk_id) {
        return (chunk_id & 1) ? "1" : "0";
    }
    static constexpr int gpu_num = 4;
    static constexpr const int gpu_id_list[gpu_num] = {0, 1, 2, 3};
    static inline int gpu_ptr(int tid, int num_threads) {
        return (tid * gpu_num) / num_threads;
    }
    static inline int gpu_ptrl(int tid, int sid, int num_threads, int threads_per_buc) {
        if (sid == -1) return gpu_ptr(tid, num_threads);
        else return (tid * threads_per_buc + sid) % gpu_num;
    }
    static inline int gpu_id(int tid, int num_threads) {
        return (tid * gpu_num) / num_threads;
    }
};

///////////////// profiling config /////////////////

#define ENABLE_PROFILING         1
#define AUTO_REPORT_DURATION  1800
#define POOL_HD_LOG_LEVEL   ll_info
#define REDUCER_LOG_LEVEL   ll_dbg
#define BUCKETER_LOG_LEVEL  ll_dbg
#define PWC_LOG_LEVEL       ll_warn
#define BWC_LOG_LEVEL       ll_warn
#define SWC_LOG_LEVEL       ll_warn
#define DHB_LOG_LEVEL       ll_dbg
#define DHR_LOG_LEVEL       ll_dbg


///////////////// cuda config /////////////////
#define MAX_NUM_DEVICE 8
#define USE_GRAPH 0
#define USE_HUGE_PAGE 0
#define BGJL_HOST_UPK 1


///////////////// pwc config /////////////////
// Device kernels retain the 176-byte CSD16 tier.  Host cache chunks only
// store CSD bytes per vector, so their fixed slot can be smaller for the
// SVP-120/130/140 profiles used by this build.  Keeping these constants
// separate avoids changing CUDA strides while providing ~20% more host
// cache slots per GB (190 -> 158 bytes per vector).
#define POOL_VEC_MAX_DIM                176
#ifndef POOL_HOST_VEC_MAX_DIM
#define POOL_HOST_VEC_MAX_DIM           144
#endif
#define POOL_HOST_VEC_SLOT_NBYTES       (POOL_HOST_VEC_MAX_DIM + 14ULL)
#define BWC_HOST_VEC_SLOT_NBYTES        (POOL_HOST_VEC_MAX_DIM + 4ULL)
#if POOL_VEC_MAX_DIM != 176
#error "POOL_VEC_MAX_DIM != 176 needs a matching CSD16 kernel tier (bgj/dh kernels hardcode 176)"
#endif
#if POOL_HOST_VEC_MAX_DIM > POOL_VEC_MAX_DIM
#error "POOL_HOST_VEC_MAX_DIM cannot exceed the CUDA vector tier"
#endif
// DRAM cache profiles for a 125GB host (PWC/BWC/SWC _DRAM_SLIMIT):
//   SVP-120 profile: 14/24/4 GB (committed default)
//   SVP-130 profile: 16/50/12 GB — solution working set needs ~12GB by
//   CSD 116, buckets ~48GB by CSD 120
//   SVP-140 profile: 47/32/8 GB — caps bind in exact host slots
//   (158 bytes for PWC/SWC, 148 for BWC): pool must stay slot-resident or
//   full-pool scans thrash the SSD. PWC47 provides 39.0K slots, enough for a
//   size-ratio-3.2 CSD128 pool (38.8K chunks). BWC below 32GB thrashes buckets
//   from CSD ~115 (2.4-3.4x per-dim, measured). GPU UID dedup reduces the SWC
//   working set enough to use an 8GB cache. The resulting 87GB arena budget
//   leaves room for the ~8GB reducer staging and transient solution queues on
//   this 125GB host; the previous 51/32/12 profile was OOM-killed at CSD127.
// Fast persistence behavior is the DEFAULT (single-SSD tuning): lazy sync
// on, pool persisted every 6th dim, no between-sieve pwc borrow. Restore
// stock behavior with HD_LAZY_SYNC=0 HD_SYNC_EVERY=1 HD_PWC_NO_GROW=0.
//   A100x4 500GB profile: 192/96/24 GB.  The CSD135 pool occupies 106956
//   exact host chunks (128.93 GiB); the former 112GB PWC cap held CSD134 but
//   made CSD135 thrash the backing store.  The wider cap holds about 159K
//   exact chunks, enough for the projected ~143K-chunk CSD137 pool, while
//   leaving about 108GB for reducer staging, queues, the OS, and transient
//   allocations on the 500GB nodes.
// Build with -DHD_SVP140_CACHE_PROFILE=1 for the 47/32/8 GB profile, or
// -DHD_A100X4_500G_CACHE_PROFILE=1 for the 192/96/24 GB profile.
#define ONE_TIME_IO                     1
#ifndef HD_SVP140_CACHE_PROFILE
#define HD_SVP140_CACHE_PROFILE         0
#endif
#ifndef HD_A100X4_500G_CACHE_PROFILE
#define HD_A100X4_500G_CACHE_PROFILE    0
#endif
#if HD_SVP140_CACHE_PROFILE && HD_A100X4_500G_CACHE_PROFILE
#error "select only one host cache profile"
#endif
#if HD_A100X4_500G_CACHE_PROFILE
#define PWC_DRAM_SLIMIT                 (192ULL << 30)
#define BWC_DRAM_SLIMIT                 (96ULL << 30)
#define SWC_DRAM_SLIMIT                 (24ULL << 30)
#elif HD_SVP140_CACHE_PROFILE
#define PWC_DRAM_SLIMIT                 (47ULL << 30)
#define BWC_DRAM_SLIMIT                 (32ULL << 30)
#define SWC_DRAM_SLIMIT                 (8ULL << 30)
#else
#define PWC_DRAM_SLIMIT                 (14ULL << 30)
#define BWC_DRAM_SLIMIT                 (24ULL << 30)
#define SWC_DRAM_SLIMIT                 (4ULL << 30)
#endif

#define PWC_DEFAULT_LOADING_THREADS     6
#define PWC_DEFAULT_SYNCING_THREADS     6
#define PWC_SSD_SLIMIT                  (10000ULL << 30)
#define PWC_DEFAULT_MAX_CACHED_CHUNKS   (PWC_DRAM_SLIMIT / 8192ULL / POOL_HOST_VEC_SLOT_NBYTES)
#define PWC_MAX_PARALLEL_SYNC_CHUNKS    5


///////////////// bwc config /////////////////
#define BWC_DEFAULT_LOADING_THREADS     8
#define BWC_DEFAULT_SYNCING_THREADS     6
#define BWC_SSD_SLIMIT                  (32ULL << 30)
#define BWC_DEFAULT_MAX_CACHED_CHUNKS   (BWC_DRAM_SLIMIT / 8192ULL / BWC_HOST_VEC_SLOT_NBYTES)
#define BWC_MAX_PARALLEL_SYNC_CHUNKS    5
#define BWC_MAX_BUCKETS                 4192


///////////////// swc config /////////////////
#define SWC_DEFAULT_LOADING_THREADS     5
#define SWC_DEFAULT_SYNCING_THREADS     3
#define SWC_SSD_SLIMIT                  (5000ULL << 30)
#define SWC_DEFAULT_MAX_CACHED_CHUNKS   (SWC_DRAM_SLIMIT / 8192ULL / POOL_HOST_VEC_SLOT_NBYTES)
#define SWC_MAX_PARALLEL_SYNC_CHUNKS    5


///////////////// red config /////////////////
#define RED_MIN_CSD16                   128     /* change with kernel choosing code tegother */
#define RED_MAX_NUM_THREADS             64
#if HD_A100X4_500G_CACHE_PROFILE
#define RED_GRAM_SLIMIT                 (48ULL << 30)
#else
#define RED_GRAM_SLIMIT                 (22ULL << 30)
#endif

#define BGJ1_RED_DEFAULT_NUM_THREADS    32
#define BGJ2_RED_DEFAULT_NUM_THREADS    48
#define BGJ3_RED_DEFAULT_NUM_THREADS    16
#define BGJ3L_RED_DEFAULT_NUM_THREADS   6
#define BGJ4_RED_DEFAULT_NUM_THREADS    4

#define BGJ2_DEFAULT_ALPHA1             0.290               
#define BGJ2_DEFAULT_BATCH1             4096

#define BGJ3_DEFAULT_ALPHA1             0.190
#define BGJ3_DEFAULT_ALPHA2             0.295
#define BGJ3_DEFAULT_BATCH1             256
#define BGJ3_DEFAULT_BATCH2             512
#define BGJ3_DEFAULT_THREADS_PER_BUC    4

#define BGJ3L_DEFAULT_ALPHA1            0.190
#define BGJ3L_DEFAULT_ALPHA2            0.295
#define BGJ3L_DEFAULT_BATCH1            256
#define BGJ3L_DEFAULT_BATCH2            512
#define BGJ3L_DEFAULT_THREADS_PER_BUC   8

#define BGJ4_DEFAULT_ALPHA1             0.180
#define BGJ4_DEFAULT_ALPHA2             0.230
#define BGJ4_DEFAULT_ALPHA3             0.300
#define BGJ4_DEFAULT_BATCH1             128
#define BGJ4_DEFAULT_BATCH2             64
#define BGJ4_DEFAULT_BATCH3             16
#define BGJ4_DEFAULT_THREADS_PER_BUC    8


///////////////// buc config /////////////////
#define BUC_MIN_CSD16                   128     /* change with kernel choosing code tegother */
#define BUC_DEFAULT_NUM_THREADS         16
#define BUC_GRAM_SLIMIT                 (1ULL << 30)

#define BGJ1_L0_MIN_ALPHA0              0.310
#define BGJ1_L0_MAX_ALPHA0              0.310
#define BGJ2_L0_MIN_ALPHA0              0.245
#define BGJ2_L0_MAX_ALPHA0              0.245
#define BGJ3_L0_MIN_ALPHA0              0.210
#define BGJ3_L0_MAX_ALPHA0              0.210
#define BGJ3L_L0_MIN_ALPHA0             0.210
#define BGJ3L_L0_MAX_ALPHA0             0.210
#define BGJ4_L0_MIN_ALPHA0              0.165
#define BGJ4_L0_MAX_ALPHA0              0.185

#define BGJ1_L0_BATCH_RATIO             0.5
#define BGJ2_L0_BATCH_RATIO             0.5
#define BGJ3_L0_BATCH_RATIO             0.5
#define BGJ3L_L0_BATCH_RATIO            0.5
#define BGJ4_L0_BATCH_RATIO             0.5

#define BGJ_L0_MAX_BATCH0               2048    /* change with kernel choosing code tegother */
#define BGJ_L0_MIN_BATCH0               16      /* change with kernel choosing code tegother */



///////////////// bgj config /////////////////
#define BGJ_DEFAULT_SATURATION_RADIUS   4./3.
#define BGJ_DEFAULT_SATURATION_RATIO    0.375
#define BGJ_DEFAULT_IMPROVE_RATIO       0.71
#define BGJ_CENTER_IMPROVE_RATIO        0.77

#define BGJ1_SIZE_RATIO                 3.2
#define BGJ2_SIZE_RATIO                 3.2
#define BGJ3_SIZE_RATIO                 3.2
#define BGJ3L_SIZE_RATIO                3.2
#define BGJ4_SIZE_RATIO                 3.2


///////////////// ut config /////////////////
#define UT_DEFAULT_NUM_THREADS          16
#define UT_TABLE_DRAM_SLIMIT            (1500ULL << 30)
#define UT_BUFFER_DRAM_SLIMIT           (300ULL << 30)
#define UT_DEFAULT_MAX_CHUNKS           (UT_BUFFER_DRAM_SLIMIT / 8192ULL / POOL_HOST_VEC_SLOT_NBYTES)
#define UT_DEFAULT_MAX_UIDS             (UT_BUFFER_DRAM_SLIMIT / 8192ULL / 32ULL)
#define UT_DEFAULT_BATCH_RATIO          0.01


///////////////// dh config /////////////////
#define DH_MAX_BATCH                    2048
#define DH_MIN_BATCH                    256
#define DHB_DEFAULT_NUM_THREADS         24
#define DHR_DEFAULT_NUM_THREADS         32
#define DHB_GRAM_SLIMIT                 (1ULL << 30)
#define DHR_GRAM_SLIMIT                 (22ULL << 30)
#define DH_BSIZE_RATIO                  (ESD <= 40 ? 30.0 : 50.0)
#define DH_REPORT_DURATION              1800
#define SPLIT_DHR                       1

#endif
