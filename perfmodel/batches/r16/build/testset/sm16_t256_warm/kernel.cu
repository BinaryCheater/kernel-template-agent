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
extern "C" __global__ void __launch_bounds__(256, 1) smax_kernel(const half_t* __restrict__ X, half_t* __restrict__ Y) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* workspace = ((void*)((char*)buf_dyn_shmem + 0));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 0));
  half_t x[14];
  float s[14];
  float mx[1];
  float sm[1];
  #pragma unroll
  for (int i = 0; i < 7; ++i) {
    *(uint1*)(x + (i * 2)) = *(uint1*)(X + (((((int)blockIdx.x) * 3584) + (i * 512)) + (((int)threadIdx.x) * 2)));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 7; ++i_1) {
    float broadcast_var = 0x1.7154764ee6c2fp+0f/*1.442695e+00*/;
    float2 __1;
      float2 __2;
      uint1 v_ = *(uint1*)(x + (i_1 * 2));
      ((float2*)(&__2))[0] = __half22float2(((half2*)(&v_))[0]);
      float2 v__1 = make_float2(broadcast_var, broadcast_var);
      *(float2*)(&(__1.x)) = tl::mul2(*(float2*)(&(__2.x)), *(float2*)(&(v__1.x)));
    *(float2*)(s + (i_1 * 2)) = __1;
  }
  mx[0] = -CUDART_INF_F;
  #pragma unroll
  for (int rv = 0; rv < 14; ++rv) {
    mx[0] = max(mx[0], s[(((rv % 7) * 2) + (rv / 7))]);
  }
  mx[0] = tl::AllReduce<tl::MaxOp, 256, 1, 0, tl::NamedBarrier<256>>::run(mx[0], (&(((float*)workspace_1)[0])));
  #pragma unroll
  for (int i_2 = 0; i_2 < 7; ++i_2) {
    float2 __3;
    float2 __4;
      float2 v__2 = *(float2*)(s + (i_2 * 2));
      float2 v__3 = make_float2(mx[0], mx[0]);
      *(float2*)(&(__4.x)) = tl::sub2(*(float2*)(&(v__2.x)), *(float2*)(&(v__3.x)));
    __3.x = exp2f(__4.x);
    __3.y = exp2f(__4.y);
    *(float2*)(s + (i_2 * 2)) = __3;
  }
  sm[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv_1 = 0; rv_1 < 14; ++rv_1) {
    sm[0] = (sm[0] + s[(((rv_1 % 7) * 2) + (rv_1 / 7))]);
  }
  __syncthreads();
  sm[0] = tl::AllReduce<tl::SumOp, 256, 1, 0, tl::NamedBarrier<256>>::run(sm[0], (&(((float*)workspace)[0])));
  #pragma unroll
  for (int i_3 = 0; i_3 < 7; ++i_3) {
    uint1 __5;
    float2 __6;
      float2 v__4 = *(float2*)(s + (i_3 * 2));
      float2 v__5 = make_float2(sm[0], sm[0]);
      __6.x = (v__4.x/v__5.x);
      __6.y = (v__4.y/v__5.y);
    ((half2*)(&__5))[0] = __float22half2_rn(((float2*)(&__6))[0]);
    *(uint1*)(x + (i_3 * 2)) = __5;
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 7; ++i_4) {
    *(uint1*)(Y + (((((int)blockIdx.x) * 3584) + (i_4 * 512)) + (((int)threadIdx.x) * 2))) = *(uint1*)(x + (i_4 * 2));
  }
}

