#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <math_constants.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void comb_kernel(const half_t* __restrict__ Opart, half_t* __restrict__ Out, const half_t* __restrict__ glse);
extern "C" __global__ void __launch_bounds__(128, 1) comb_kernel(const half_t* __restrict__ Opart, half_t* __restrict__ Out, const half_t* __restrict__ glse) {
  float mx[1];
  float sm[1];
  float acc[1];
  mx[0] = -CUDART_INF_F;
  for (int k = 0; k < 8; ++k) {
    mx[0] = max(mx[0], ((float)glse[((((int)blockIdx.x) * 8) + k)]));
  }
  sm[0] = 0x0p+0f/*0.000000e+00*/;
  acc[0] = 0x0p+0f/*0.000000e+00*/;
  for (int k_1 = 0; k_1 < 8; ++k_1) {
    sm[0] = (sm[0] + exp2f((((float)glse[((((int)blockIdx.x) * 8) + k_1)]) - mx[0])));
    acc[0] = (acc[0] + (exp2f((((float)glse[((((int)blockIdx.x) * 8) + k_1)]) - mx[0])) * ((float)Opart[(((((int)blockIdx.x) * 1024) + (k_1 * 128)) + ((int)threadIdx.x))])));
  }
  Out[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = ((half_t)(acc[0] / sm[0]));
}

