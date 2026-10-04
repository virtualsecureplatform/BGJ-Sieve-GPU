#include "../include/pool_hd.h"
#include <filesystem>
#include <cstdio>
#include <cstdlib>

static void require(bool ok, const char *message) {
    if (!ok) { fprintf(stderr, "FAIL: %s\n", message); abort(); }
}

static void check(chunk_t *c, long id, int pass) {
    require(c && c->size == 1, "chunk size");
    require(c->score[0] == id + pass, "score");
    require(c->norm[0] == 10000 + id + pass, "norm");
    require(c->u[0] == 100000 + id + pass, "uid");
    for (int j = 0; j < 64; ++j)
        require(c->vec[j] == (id + j + pass) % 101, "vector");
}

static void fill(chunk_t *c, long id, int pass) {
    c->size = 1;
    c->score[0] = id + pass;
    c->norm[0] = 10000 + id + pass;
    c->u[0] = 100000 + id + pass;
    for (int j = 0; j < 64; ++j) c->vec[j] = (id + j + pass) % 101;
}

int main(int argc, char **argv) {
    require(argc == 3, "usage: test_hbm_writeback BASIS generous|spill");
    Lattice_QP basis(argv[1]);
    Pool_hd_t pool(&basis);
    require(pool.set_sieving_context(100, 164) == 0, "context");
    auto *pwc = pool.pwc_manager;
    require(pwc->set_max_cached_chunks(256) == 0, "resize");
    _pwc_hbm_prepare(64);
    constexpr long count = 384;
    for (long id = 0; id < count; ++id) {
        require(pwc->create_chunk() == id, "chunk id");
        fill(pwc->fetch(id), id, 0);
        pwc->release_sync(id);
    }
    for (int pass = 1; pass <= 3; ++pass) {
        #pragma omp parallel for num_threads(8) schedule(static)
        for (long id = 0; id < count; ++id) {
            chunk_t *c = pwc->fetch(id);
            check(c, id, pass - 1);
            fill(c, id, pass);
            pwc->release_sync(id);
        }
    }
    pwc->wait_work();
    size_t files = 0;
    for (const auto &entry : std::filesystem::recursive_directory_iterator(".pool"))
        if (entry.is_regular_file()) ++files;
    if (std::string(argv[2]) == "generous") require(files == 0, "unexpected disk spill");
    else require(files > 0, "spill path not exercised");
    _pwc_hbm_report("test-before-flush", 64);
    pwc->flush();
    _pwc_hbm_report("test-after-flush", 64);
    _pwc_hbm_reset(); // Must refuse this if any authoritative dirty entry remains.
    files = 0;
    for (const auto &entry : std::filesystem::recursive_directory_iterator(".pool")) {
        if (!entry.is_regular_file()) continue;
        ++files;
        const auto name = entry.path().filename().string();
        long id = strtol(name.c_str() + name.size() - 6, nullptr, 16);
        int fd = open(entry.path().c_str(), O_RDONLY);
        require(fd >= 0, "open persisted chunk");
        uint16_t size, score;
        int32_t norm;
        uint64_t uid;
        int8_t vec[64];
        constexpr long n = Pool_hd_t::chunk_max_nvecs;
        require(pread(fd, &size, 2, 0) == 2 && size == 1, "persisted size");
        require(pread(fd, &score, 2, 12) == 2 && score == id + 3, "persisted score");
        require(pread(fd, &norm, 4, 12 + 2*n) == 4 && norm == 10000 + id + 3, "persisted norm");
        require(pread(fd, &uid, 8, 12 + 6*n) == 8 && uid == 100000 + id + 3, "persisted uid");
        require(pread(fd, vec, 64, 12 + 14*n) == 64, "persisted vector read");
        for (int j = 0; j < 64; ++j) require(vec[j] == (id+j+3)%101, "persisted vector");
        close(fd);
    }
    require(files == count, "missing persisted chunks");
    for (long id = 0; id < count; ++id) {
        check(pwc->fetch(id), id, 3);
        pwc->release(id);
    }
    pool.set_sieving_context(100, 164);
    printf("PASS: HBM write-back %s, concurrent reloads and durable flush\n", argv[2]);
}
