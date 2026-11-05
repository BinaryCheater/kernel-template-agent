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
extern "C" __global__ void __launch_bounds__(512, 1) rms_kernel(const half_t* __restrict__ X, half_t* __restrict__ Y) {
  half_t x[7];
  #pragma unroll
  for (int i = 0; i < 7; ++i) {
    x[i] = X[(((((int)blockIdx.x) * 3584) + (i * 512)) + ((int)threadIdx.x))];
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 7; ++i_1) {
    Y[(((((int)blockIdx.x) * 3584) + (i_1 * 512)) + ((int)threadIdx.x))] = x[i_1];
  }
}

