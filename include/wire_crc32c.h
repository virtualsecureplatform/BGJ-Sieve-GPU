#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <nmmintrin.h>

// Wire-v3 checksum, independent of how the byte stream is split into columns.
// The final short tail belongs exclusively to lane a, as in checksum64().
class wire_crc32c {
    uint64_t a_ = 0xffffffffU, b_ = 0x9e3779b9U;
    uint8_t tail_[16];
    size_t tail_size_ = 0;
    void block(const void *source) {
        uint64_t x, y;
        memcpy(&x, source, 8); memcpy(&y, static_cast<const uint8_t *>(source)+8, 8);
        a_ = _mm_crc32_u64(a_, x); b_ = _mm_crc32_u64(b_, y);
    }
public:
    void copy(void *destination, const void *source, size_t size) {
        auto *dst = static_cast<uint8_t *>(destination);
        auto *src = static_cast<const uint8_t *>(source);
        if (tail_size_) {
            size_t n = size < 16-tail_size_ ? size : 16-tail_size_;
            memcpy(dst, src, n); memcpy(tail_+tail_size_, src, n);
            dst += n; src += n; size -= n; tail_size_ += n;
            if (tail_size_ == 16) { block(tail_); tail_size_ = 0; }
        }
        while (size >= 16) {
            uint64_t x, y;
            memcpy(&x, src, 8); memcpy(&y, src+8, 8);
            memcpy(dst, &x, 8); memcpy(dst+8, &y, 8);
            a_ = _mm_crc32_u64(a_, x); b_ = _mm_crc32_u64(b_, y);
            dst += 16; src += 16; size -= 16;
        }
        if (size) { memcpy(dst, src, size); memcpy(tail_, src, size); tail_size_ = size; }
    }
    uint64_t finish() const {
        uint64_t a = a_;
        for (size_t i = 0; i < tail_size_; ++i) a = _mm_crc32_u8(a, tail_[i]);
        return ((a ^ 0xffffffffU) << 32) | (b_ ^ 0xffffffffU);
    }
    static void copy_callback(void *context, void *dst, const void *src, size_t size) {
        static_cast<wire_crc32c *>(context)->copy(dst, src, size);
    }
};
