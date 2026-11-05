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

extern "C" __global__ void rms_kernel(const half_t* __restrict__ W, const half_t* __restrict__ X, half_t* __restrict__ Y);
extern "C" __global__ void __launch_bounds__(64, 1) rms_kernel(const half_t* __restrict__ W, const half_t* __restrict__ X, half_t* __restrict__ Y) {
  half_t x[35];
  float sq[35];
  float ss[1];
  extern __shared__ __align__(1024) float workspace[];
  #pragma unroll
  for (int i = 0; i < 35; ++i) {
    x[i] = X[(((((int)blockIdx.x) * 2240) + (i * 64)) + ((int)threadIdx.x))];
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 35; ++i_1) {
    sq[i_1] = (((float)x[i_1]) * ((float)x[i_1]));
  }
  ss[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv = 0; rv < 35; ++rv) {
    ss[0] = (ss[0] + sq[rv]);
  }
  ss[0] = tl::AllReduce<tl::SumOp, 64, 1, 0, tl::NamedBarrier<64>>::run(ss[0], (&(workspace[0])));
  ss[0] = rsqrtf(((ss[0] / 0x1.18p+11f/*2.240000e+03*/) + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/));
  #pragma unroll
  for (int i_2 = 0; i_2 < 35; ++i_2) {
    float wj = ((float)W[((i_2 * 64) + ((int)threadIdx.x))]);
    x[i_2] = ((half_t)((((float)x[i_2]) * ss[0]) * wj));
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 35; ++i_3) {
    Y[(((((int)blockIdx.x) * 2240) + (i_3 * 64)) + ((int)threadIdx.x))] = x[i_3];
  }
}

