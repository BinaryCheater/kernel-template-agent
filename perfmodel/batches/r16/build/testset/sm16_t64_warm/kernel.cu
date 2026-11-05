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

extern "C" __global__ void smax_kernel(const half_t* __restrict__ X, half_t* __restrict__ Y);
extern "C" __global__ void __launch_bounds__(64, 1) smax_kernel(const half_t* __restrict__ X, half_t* __restrict__ Y) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* workspace = ((void*)((char*)buf_dyn_shmem + 0));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 0));
  half_t x[35];
  float s[35];
  float mx[1];
  float sm[1];
  #pragma unroll
  for (int i = 0; i < 35; ++i) {
    x[i] = X[(((((int)blockIdx.x) * 2240) + (i * 64)) + ((int)threadIdx.x))];
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 35; ++i_1) {
    s[i_1] = (((float)x[i_1]) * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/);
  }
  mx[0] = -CUDART_INF_F;
  #pragma unroll
  for (int rv = 0; rv < 35; ++rv) {
    mx[0] = max(mx[0], s[rv]);
  }
  mx[0] = tl::AllReduce<tl::MaxOp, 64, 1, 0, tl::NamedBarrier<64>>::run(mx[0], (&(((float*)workspace_1)[0])));
  #pragma unroll
  for (int i_2 = 0; i_2 < 35; ++i_2) {
    s[i_2] = exp2f((s[i_2] - mx[0]));
  }
  sm[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv_1 = 0; rv_1 < 35; ++rv_1) {
    sm[0] = (sm[0] + s[rv_1]);
  }
  __syncthreads();
  sm[0] = tl::AllReduce<tl::SumOp, 64, 1, 0, tl::NamedBarrier<64>>::run(sm[0], (&(((float*)workspace)[0])));
  #pragma unroll
  for (int i_3 = 0; i_3 < 35; ++i_3) {
    x[i_3] = ((half_t)(s[i_3] / sm[0]));
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 35; ++i_4) {
    Y[(((((int)blockIdx.x) * 2240) + (i_4 * 64)) + ((int)threadIdx.x))] = x[i_4];
  }
}

