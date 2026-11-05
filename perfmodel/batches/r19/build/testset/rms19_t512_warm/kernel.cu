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
extern "C" __global__ void __launch_bounds__(512, 1) rms_kernel(const half_t* __restrict__ W, const half_t* __restrict__ X, half_t* __restrict__ Y) {
  half_t x[13];
  float sq[13];
  float ss[1];
  extern __shared__ __align__(1024) float workspace[];
  #pragma unroll
  for (int i = 0; i < 13; ++i) {
    x[i] = X[(((((int)blockIdx.x) * 6656) + (i * 512)) + ((int)threadIdx.x))];
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 13; ++i_1) {
    sq[i_1] = (((float)x[i_1]) * ((float)x[i_1]));
  }
  ss[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv = 0; rv < 13; ++rv) {
    ss[0] = (ss[0] + sq[rv]);
  }
  ss[0] = tl::AllReduce<tl::SumOp, 512, 1, 0, tl::NamedBarrier<512>>::run(ss[0], (&(workspace[0])));
  ss[0] = rsqrtf(((ss[0] / 0x1.ap+12f/*6.656000e+03*/) + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/));
  #pragma unroll
  for (int i_2 = 0; i_2 < 13; ++i_2) {
    float wj = ((float)W[((i_2 * 512) + ((int)threadIdx.x))]);
    x[i_2] = ((half_t)((((float)x[i_2]) * ss[0]) * wj));
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 13; ++i_3) {
    Y[(((((int)blockIdx.x) * 6656) + (i_3 * 512)) + ((int)threadIdx.x))] = x[i_3];
  }
}

