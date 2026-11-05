#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void rms_kernel(const half_t* __restrict__ X, half_t* __restrict__ Y);
extern "C" __global__ void __launch_bounds__(128, 1) rms_kernel(const half_t* __restrict__ X, half_t* __restrict__ Y) {
  half_t x[44];
  #pragma unroll
  for (int i = 0; i < 11; ++i) {
    *(uint2*)(x + (i * 4)) = *(uint2*)(X + (((((int)blockIdx.x) * 5632) + (i * 512)) + (((int)threadIdx.x) * 4)));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 11; ++i_1) {
    *(uint2*)(Y + (((((int)blockIdx.x) * 5632) + (i_1 * 512)) + (((int)threadIdx.x) * 4))) = *(uint2*)(x + (i_1 * 4));
  }
}

