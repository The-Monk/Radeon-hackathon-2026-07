// golden_gen.cu — reference outputs for the low-bit agent-reproduction rig.
// The agent never sees this file or the .bin it writes.
//   hipcc --offload-arch=gfx1201 -O3 -DFMT_Q2 golden_gen.cu -o golden_gen_q2
#include "quant_common.h"
#include <cstdio>
#include <vector>
#include <random>

__global__ void golden_kernel(const block_lowbit* __restrict__ w, const float* __restrict__ act,
                              float* __restrict__ dst, int64_t Kb)
{
    const int64_t row = blockIdx.x; const int tid = threadIdx.x;
    const block_lowbit* wr = w + row * Kb;
    float partial = 0.0f;
    for (int64_t c = tid; c < Kb; c += blockDim.x) {
        float s = 0.0f;
        for (int k = 0; k < QK_LOWBIT; ++k)
            s += lowbit_value(wr[c].qs, k) * act[c * QK_LOWBIT + k];
        partial += s * __half2float(wr[c].d);
    }
    extern __shared__ float sd[]; sd[tid] = partial; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) { if (tid < s) sd[tid] += sd[tid+s]; __syncthreads(); }
    if (tid == 0) dst[row] = sd[0];
}

int main() {
    const int N = 2048, Kb = 32;            // K = 4096
    std::mt19937 rng(20260806);
    std::uniform_int_distribution<int> byte_(0,255);
    std::uniform_real_distribution<float> af(-1.f,1.f);
    std::vector<block_lowbit> hW((size_t)N*Kb);
    std::vector<float> hA((size_t)Kb*QK_LOWBIT);
    for (auto& b : hW) { b.d = __float2half(0.02f); for (auto& q : b.qs) q = (uint8_t)byte_(rng); }
    for (auto& a : hA) a = af(rng);

    block_lowbit* dW; float *dA, *dO;
    hipMalloc(&dW, hW.size()*sizeof(block_lowbit)); hipMalloc(&dA, hA.size()*4); hipMalloc(&dO, (size_t)N*4);
    hipMemcpy(dW, hW.data(), hW.size()*sizeof(block_lowbit), hipMemcpyHostToDevice);
    hipMemcpy(dA, hA.data(), hA.size()*4, hipMemcpyHostToDevice);
    golden_kernel<<<N, 64, 64*sizeof(float)>>>(dW, dA, dO, Kb);
    hipDeviceSynchronize();
    std::vector<float> out(N); hipMemcpy(out.data(), dO, (size_t)N*4, hipMemcpyDeviceToHost);

    char p[64];
    snprintf(p,sizeof p,"golden_%s_w.bin",FMT_NAME); FILE* f=fopen(p,"wb"); fwrite(hW.data(),sizeof(block_lowbit),hW.size(),f); fclose(f);
    snprintf(p,sizeof p,"golden_%s_a.bin",FMT_NAME); f=fopen(p,"wb"); fwrite(hA.data(),4,hA.size(),f); fclose(f);
    snprintf(p,sizeof p,"golden_%s_o.bin",FMT_NAME); f=fopen(p,"wb"); fwrite(out.data(),4,out.size(),f); fclose(f);
    printf("%s golden: N=%d Kb=%d first=%.6f last=%.6f\n", FMT_NAME, N, Kb, out[0], out[N-1]);
    return 0;
}
