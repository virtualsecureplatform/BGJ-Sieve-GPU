#include "../include/bgj_hd_device.h"
#include <cassert>
#include <cstdio>
#include <vector>

template<int batch, int dim> void probe(int csd, int count) {
    int8_t *centers, *vectors;
    int32_t *norms, *n;
    uint32_t *out;
    constexpr int cap = 128, vec_cap = buc_traits_t::taskVecs;
    CHECK_CUDA_ERR(cudaMalloc(&centers, batch * dim));
    CHECK_CUDA_ERR(cudaMalloc(&vectors, vec_cap * dim));
    CHECK_CUDA_ERR(cudaMalloc(&norms, vec_cap * sizeof(int32_t)));
    CHECK_CUDA_ERR(cudaMalloc(&n, sizeof(int32_t)));
    CHECK_CUDA_ERR(cudaMalloc(&out, batch * (cap+1) * sizeof(uint32_t)));
    CHECK_CUDA_ERR(cudaMemset(centers, 0, batch * dim));
    CHECK_CUDA_ERR(cudaMemset(vectors, 0, vec_cap * dim));
    CHECK_CUDA_ERR(cudaMemset(norms, 1, vec_cap * sizeof(int32_t)));
    CHECK_CUDA_ERR(cudaMemset(out, 0, batch * (cap+1) * sizeof(uint32_t)));
    CHECK_CUDA_ERR(cudaMemcpy(n, &count, sizeof(int32_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaFuncSetAttribute(_bucket_kernel<0,batch,dim>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, buc_traits_t::l0_shmem));
    _bucket_kernel<0,batch,dim><<<buc_traits_t::kernelBlocks,
        buc_traits_t::blockThreads,buc_traits_t::l0_shmem>>>(
        out, cap, nullptr, norms, centers, 0, vectors, n, .3f, csd, 8, 0);
    CHECK_CUDA_ERR(cudaGetLastError());
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    std::vector<uint32_t> sizes(batch);
    CHECK_CUDA_ERR(cudaMemcpy(sizes.data(), out, batch*sizeof(uint32_t), cudaMemcpyDeviceToHost));
    for (auto size : sizes) assert(size == 0);
    CHECK_CUDA_ERR(cudaFree(out)); CHECK_CUDA_ERR(cudaFree(n));
    CHECK_CUDA_ERR(cudaFree(norms)); CHECK_CUDA_ERR(cudaFree(vectors));
    CHECK_CUDA_ERR(cudaFree(centers));
}
template<int dim> void matrix(int csd) {
    for (int n : {0,1,63,64,65,int(buc_traits_t::taskVecs)-1,int(buc_traits_t::taskVecs)}) {
        probe<16,dim>(csd,n); probe<32,dim>(csd,n); probe<64,dim>(csd,n);
        probe<128,dim>(csd,n); probe<256,dim>(csd,n);
    }
}
int main() {
    matrix<128>(127); matrix<128>(128);
    for (int csd : {129,132,133,143,144}) matrix<144>(csd);
    puts("PASS: bucket center/tail matrix, 245 launches");
}
