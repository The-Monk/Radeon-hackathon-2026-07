// quant_common.h — shared definitions for the Q1_0 / Q2_0 agent-reproduction rig.
//
// Both formats are low-bit weight encodings from the roc8 fork, with a per-block
// fp16 scale and float activations. Compile with -DFMT_Q1 or -DFMT_Q2.
//
//   Q1_0  QK=128, qs[16]  1 bit/elem   bit i of byte gb -> elem 8*gb+i
//                                      bit set -> +1, clear -> -1
//   Q2_0  QK=128, qs[32]  2 bits/elem  code=(byte>>2e)&3 -> elem 4*gb+e
//                                      value = code-1  (0->-1, 1->0, 2->+1)
//
// Layouts and decode rules taken from the fork: ggml-common.h for the structs,
// mul_mat_q2_0_gemv.cuh for the bit conventions, ggml-quants.c for the CPU
// dequantiser. They are not re-derived here.

#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

#define QK_LOWBIT 128

#if defined(FMT_Q1)
  #define FMT_NAME "Q1_0"
  #define QS_BYTES (QK_LOWBIT / 8)     // 16
#elif defined(FMT_Q2)
  #define FMT_NAME "Q2_0"
  #define QS_BYTES (QK_LOWBIT / 4)     // 32
#else
  #error "define FMT_Q1 or FMT_Q2"
#endif

struct block_lowbit {
    __half  d;
    uint8_t qs[QS_BYTES];
};

// Decode element k (0..127) of a block to its logical value.
__host__ __device__ __forceinline__ float lowbit_value(const uint8_t* qs, int k)
{
#if defined(FMT_Q1)
    const int gb  = k / 8;
    const int bit = k % 8;
    return ((qs[gb] >> bit) & 1) ? 1.0f : -1.0f;
#else
    const int gb   = k / 4;
    const int e    = k % 4;
    const int code = (qs[gb] >> (2 * e)) & 3;
    return (float)(code - 1);          // 0->-1, 1->0, 2->+1 (3 unused)
#endif
}
