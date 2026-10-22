#include "../include/wire_crc32c.h"
#include <algorithm>
#include <cassert>
#include <random>
#include <vector>

uint64_t reference(const uint8_t *data, size_t size) {
    uint64_t a = 0xffffffffU, b = 0x9e3779b9U;
    size_t i = 0;
    for (; i+16 <= size; i += 16) {
        uint64_t x,y; memcpy(&x,data+i,8); memcpy(&y,data+i+8,8);
        a = _mm_crc32_u64(a,x); b = _mm_crc32_u64(b,y);
    }
    for (; i < size; ++i) a = _mm_crc32_u8(a,data[i]);
    return ((a^0xffffffffU)<<32) | (b^0xffffffffU);
}
int main() {
    std::mt19937 random(1234);
    for (size_t size = 0; size < 66000; size += size < 512 ? 1 : 337) {
        std::vector<uint8_t> input(size+32), output(size+32,0);
        for (auto &byte : input) byte = random();
        for (size_t split : {size_t(1),size_t(4),size_t(15),size_t(16),size_t(17),size_t(1009)}) {
            wire_crc32c crc;
            for (size_t offset = 0; offset < size;) {
                const size_t n = std::min(split,size-offset);
                crc.copy(output.data()+1+offset,input.data()+3+offset,n);
                if (offset+n <= 32 || offset+n == size)
                    assert(crc.finish() == reference(input.data()+3,offset+n));
                offset += n;
            }
            assert(crc.finish() == reference(input.data()+3,size));
            assert(std::equal(output.begin()+1,output.begin()+1+size,input.begin()+3));
        }
    }
}
