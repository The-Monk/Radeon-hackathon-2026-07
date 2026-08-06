// quant_common.h — shared definitions for the quant-format agent-reproduction rig.
//
// Layouts and decode rules are taken from the roc8 fork (ggml-common.h for the
// structs, mul_mat_q2_0_gemv.cuh for the low-bit conventions, ggml-quants.c for
// the CPU dequantisers). They are not re-derived here.
//
//   Q1_0  QK=128 qs[16]  1 bit   bit k%8 of byte k/8; set -> +1, clear -> -1
//   Q2_0  QK=128 qs[32]  2 bits  code=(byte>>2*(k%4))&3 -> value code-1
//   Q4_0  QK=32  qs[16]  nibble  low nibble of byte j = elem j,
//                                high nibble of byte j = elem j+16; value n-8
//   Q8_0  QK=32  qs[32]  int8    value = qs[k] directly
//
// Compile with -DFMT_Q1 / -DFMT_Q2 / -DFMT_Q4 / -DFMT_Q8.

#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

#if   defined(FMT_Q1)
  #define FMT_NAME "Q1_0"
  #define QK_LOWBIT 128
  #define QS_BYTES  16
  typedef uint8_t qs_t;
#elif defined(FMT_Q2)
  #define FMT_NAME "Q2_0"
  #define QK_LOWBIT 128
  #define QS_BYTES  32
  typedef uint8_t qs_t;
#elif defined(FMT_Q4)
  #define FMT_NAME "Q4_0"
  #define QK_LOWBIT 32
  #define QS_BYTES  16
  typedef uint8_t qs_t;
#elif defined(FMT_Q8)
  #define FMT_NAME "Q8_0"
  #define QK_LOWBIT 32
  #define QS_BYTES  32
  typedef int8_t  qs_t;
#else
  #error "define FMT_Q1, FMT_Q2, FMT_Q4 or FMT_Q8"
#endif

struct block_lowbit {
    __half d;
    qs_t   qs[QS_BYTES];
};

// Decode element k (0..QK_LOWBIT-1) of a block to its logical value.
__host__ __device__ __forceinline__ float lowbit_value(const qs_t* qs, int k)
{
#if   defined(FMT_Q1)
    return ((qs[k / 8] >> (k % 8)) & 1) ? 1.0f : -1.0f;
#elif defined(FMT_Q2)
    const int code = (qs[k / 4] >> (2 * (k % 4))) & 3;
    return (float)(code - 1);                       // 0->-1, 1->0, 2->+1
#elif defined(FMT_Q4)
    const int j = k % 16;
    const int n = (k < 16) ? (qs[j] & 0x0F) : (qs[j] >> 4);
    return (float)(n - 8);                          // unsigned 0..15 -> -8..+7
#else   // FMT_Q8
    return (float)qs[k];
#endif
}
