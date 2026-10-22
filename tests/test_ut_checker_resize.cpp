// Real checker/UID code under ASan+UBSan. Only the unrelated, 150+ GiB GPU
// cache arena startup is wrapped: these tests never fetch a vector-cache chunk.
#include "../include/bgj_hd.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <new>
#include <stdexcept>
#include <vector>

static std::atomic<int> fail_after{-1}, allocator_calls{0};
extern "C" void *__real_calloc(size_t, size_t);
extern "C" void *__wrap_calloc(size_t count, size_t bytes) {
    const int remaining = fail_after.load();
    if (remaining >= 0) {
        if (!remaining) { fail_after.store(-1); return nullptr; }
        fail_after.fetch_sub(1);
    }
    return __real_calloc(count,bytes);
}
extern "C" void __wrap__Z19_start_ck_allocatorv() { allocator_calls.fetch_add(1); }

static void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

static void check_view(chunk_t *view, long capacity) {
    require(view && view[-1].size == Pool_hd_t::chunk_max_nvecs,"missing full sentinel");
    require(!view[-1].u && !view[-1].score && !view[-1].norm && !view[-1].vec,"dirty sentinel pointers");
    for (long i = 0; i < capacity; ++i)
        require(!view[i].id && !view[i].size && !view[i].u && !view[i].score &&
                !view[i].norm && !view[i].vec,"new descriptor is not zeroed");
}

static void check_storage(ut_checker_t &c) {
    check_view(c._to_check,c._max_holding); check_view(c._in_check,c._max_holding);
    if (c._type == ut_checker_t::type_sieve) {
        check_view(c._red_to_rm,c._max_holding); check_view(c._red_in_rm,c._max_holding);
    } else require(!c._red_to_rm && !c._red_in_rm,"unexpected removal storage");
}

static void verify_unchanged(ut_checker_t &c, chunk_t **views, long capacity, long batch) {
    require(c._to_check == views[0] && c._in_check == views[1] &&
            c._red_to_rm == views[2] && c._red_in_rm == views[3],"failed resize changed pointers");
    require(c._max_holding == capacity && c._exp_batch == batch,"failed resize changed configuration");
}

static void complete_uid_batch(ut_checker_t &c, Pool_hd_t &pool) {
    std::vector<uint64_t> uids(17);
    for (size_t i = 0; i < uids.size(); ++i) {
        uids[i] = 100001+i;
        pool.uid_table->insert(uids[i]);
    }
    c.set_exp_batch(4);
    chunk_t *old_to = c._to_check, *old_in = c._in_check;
    require(!c.task_commit(uids.data(),uids.size()),"UID batch submission failed");
    require(c._num_to_check == 1,"expected an untriggered partial UID batch");
    require(c.set_max_holding(32) == -1,"queued work was resized");
    require(c._to_check == old_to && c._in_check == old_in && c._num_to_check == 1,
            "rejected resize lost queued work");
    require(c._to_check[0].size == 17 && c._to_check[0].u[0] == uids[0],"UID payload changed");
    c.input_done();
    require(!c.wait_work(),"UID checking failed");
    c._ut_pool.wait_sleep();
    require(!c._running_threads && !c._num_to_check && !c._num_in_check,"checker did not drain");
    require(c._to_check == old_in && c._in_check == old_to,"real batch did not swap the views");
    for (uint64_t uid : uids)
        require(pool.uid_table->check(uid) == (c._type == ut_checker_t::type_check ? 1 : 0),
                "queued UID work was not executed exactly");
}

static void exercise(long type, Pool_hd_t &pool, swc_manager_t &solutions) {
    std::vector<uint64_t> borrowed(8192,0x123456789abcdefULL);
    {
        ut_checker_t c(type,&pool,pool.uid_table,
                       type == ut_checker_t::type_sieve ? (pwc_manager_t *)&solutions : pool.pwc_manager);
        c.set_num_threads(1);
        require(!c.set_max_holding(8),"second resize failed"); // Crashes the original implementation.
        check_storage(c);
        c.set_exp_batch(7);
        require(!c.set_max_holding(4) && c._exp_batch == 4,"shrink did not clamp batch size");
        check_storage(c);
        require(!c.set_max_holding(16) && c._exp_batch == 4,"grow changed the batch threshold");
        check_storage(c);
        chunk_t *same = c._to_check;
        require(!c.set_max_holding(16) && c._to_check == same,"same-capacity resize is not a no-op");
        require(!c.set_max_holding(8),"repeat shrink failed");
        c.set_exp_batch(3);
        chunk_t *views[] = {c._to_check,c._in_check,c._red_to_rm,c._red_in_rm};
        for (int invalid : {0,-1}) {
            require(c.set_max_holding(invalid) == -1,"invalid capacity accepted");
            verify_unchanged(c,views,8,3);
        }
        // Retained UID allocations must be freed, while borrowed cache data
        // must remain valid. Queue counts are zero: all these slots are spare.
        if (type == ut_checker_t::type_check || type == ut_checker_t::type_shrink) {
            c._to_check[0].u = (uint64_t *)std::malloc(8192*sizeof(uint64_t));
            c._in_check[7].u = (uint64_t *)std::malloc(8192*sizeof(uint64_t));
        } else {
            c._to_check[0].u = borrowed.data(); c._in_check[7].u = borrowed.data();
        }
        if (type == ut_checker_t::type_sieve) {
            c._red_to_rm[0].u = (uint64_t *)std::malloc(8192*sizeof(uint64_t));
            c._red_in_rm[7].u = (uint64_t *)std::malloc(8192*sizeof(uint64_t));
        }
        const auto retained = c._to_check[0].u;
        for (int failed_array = 0; failed_array < (type == ut_checker_t::type_sieve ? 4 : 2); ++failed_array) {
            fail_after.store(failed_array);
            const int rc = c.set_max_holding(16);
            fail_after.store(-1);
            require(rc == -2,"injected allocation failure was not reported");
            verify_unchanged(c,views,8,3);
            require(c._to_check[0].u == retained,"allocation failure lost retained UID storage");
        }
        for (long *queued : {&c._num_to_check,&c._num_in_check,&c._num_red_to_rm,&c._num_red_in_rm}) {
            *queued = 1;
            const int rc = c.set_max_holding(16);
            *queued = 0;
            require(rc == -1,"nonempty queue accepted a resize");
            verify_unchanged(c,views,8,3);
        }
        // Deterministically keep one worker active. A busy resize must return
        // immediately, not join it while holding its completion mutex.
        std::promise<void> release;
        auto released = release.get_future();
        c._running_threads.store(1);
        c._ut_pool.push([&] { released.wait(); c._running_threads.store(0); });
        const int busy_rc = c.set_max_holding(16);
        release.set_value(); c._ut_pool.wait_sleep();
        require(busy_rc == -1,"active checker accepted a resize");
        verify_unchanged(c,views,8,3);
        require(!c.set_max_holding(16),"resize after failed attempts did not recover");
        check_storage(c);
        require(borrowed[0] == 0x123456789abcdefULL,"resize freed borrowed vector-cache storage");
        if (type == ut_checker_t::type_check || type == ut_checker_t::type_shrink) {
            complete_uid_batch(c,pool);
            require(!c.set_max_holding(8),"resize after real queue swap failed");
            check_storage(c);
            complete_uid_batch(c,pool);
            require(!c.set_max_holding(32),"second post-batch resize failed");
            check_storage(c);
        }
        // Leave retained/borrowed data for the shared destructor cleanup too.
        if (type == ut_checker_t::type_check || type == ut_checker_t::type_shrink)
            c._to_check[0].u = (uint64_t *)std::malloc(8192*sizeof(uint64_t));
        else c._to_check[0].u = borrowed.data();
        if (type == ut_checker_t::type_sieve)
            c._red_to_rm[0].u = (uint64_t *)std::malloc(8192*sizeof(uint64_t));
    }
    require(borrowed.back() == 0x123456789abcdefULL,"destructor freed borrowed storage");
    const int arrays = type == ut_checker_t::type_sieve ? 4 : 2;
    for (int failed_array = 0; failed_array < arrays; ++failed_array) {
        fail_after.store(failed_array);
        bool threw = false;
        try {
            ut_checker_t c(type,&pool,pool.uid_table,
                           type == ut_checker_t::type_sieve ? (pwc_manager_t *)&solutions : pool.pwc_manager);
        } catch (const std::bad_alloc &) { threw = true; }
        fail_after.store(-1);
        require(threw,"constructor ignored an allocation failure");
    }
    printf("PASS: UID-checker resize type=%ld sentinel/grow/shrink/swap/busy/failure/ownership/destruction\n",type);
    fflush(stdout);
}

int main() {
    try {
        for (const char *kind : {".pool",".sol",".uid"})
            for (const char *ssd : {"0","1"})
                std::filesystem::create_directories(std::filesystem::path(kind)/ssd);
        {
            Pool_hd_t pool;
            pool.CSD = 60;
            swc_manager_t solutions(&pool);
            for (long type : {ut_checker_t::type_sieve,ut_checker_t::type_check,
                              ut_checker_t::type_shrink,ut_checker_t::type_others})
                exercise(type,pool,solutions);
        }
        require(allocator_calls.load() > 0,"test allocator wrapper did not intercept arena startup");
        puts("PASS: all executable UID-checker resize regressions");
        return 0;
    } catch (const std::exception &e) {
        fail_after.store(-1);
        fprintf(stderr,"FAIL: UID-checker resize: %s\n",e.what());
        return 1;
    }
}
