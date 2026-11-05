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
extern "C" __global__ void __launch_bounds__(128, 1) rms_kernel(const half_t* __restrict__ W, const half_t* __restrict__ X, half_t* __restrict__ Y) {
  half_t x[36];
  float sq[36];
  float ss[1];
  extern __shared__ __align__(1024) float workspace[];
  #pragma unroll
  for (int i = 0; i < 9; ++i) {
    *(uint2*)(x + (i * 4)) = *(uint2*)(X + (((((int)blockIdx.x) * 4608) + (i * 512)) + (((int)threadIdx.x) * 4)));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 9; ++i_1) {
    float4 __1;
      float4 __2;
      uint2 v_ = *(uint2*)(x + (i_1 * 4));
      ((float2*)(&__2))[0] = __half22float2(((half2*)(&v_))[0]);
      ((float2*)(&__2))[1] = __half22float2(((half2*)(&v_))[1]);
      float4 __3;
      ((float2*)(&__3))[0] = __half22float2(((half2*)(&v_))[0]);
      ((float2*)(&__3))[1] = __half22float2(((half2*)(&v_))[1]);
      *(float2*)(&(__1.x)) = tl::mul2(*(float2*)(&(__2.x)), *(float2*)(&(__3.x)));
      *(float2*)(&(__1.z)) = tl::mul2(*(float2*)(&(__2.z)), *(float2*)(&(__3.z)));
    *(float4*)(sq + (i_1 * 4)) = __1;
  }
  ss[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv = 0; rv < 36; ++rv) {
    ss[0] = (ss[0] + sq[(((rv % 9) * 4) + (rv / 9))]);
  }
  ss[0] = tl::AllReduce<tl::SumOp, 128, 1, 0, tl::NamedBarrier<128>>::run(ss[0], (&(workspace[0])));
  ss[0] = rsqrtf(((ss[0] / 0x1.2p+12f/*4.608000e+03*/) + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/));
  #pragma unroll
  for (int i_2 = 0; i_2 < 9; ++i_2) {
    half_t W_local_cast[4];
    *(uint2*)(W_local_cast + 0) = *(uint2*)(W + ((i_2 * 512) + (((int)threadIdx.x) * 4)));
    uint2 __4;
    float4 __5;
      float4 __6;
        float4 __7;
        uint2 v__1 = *(uint2*)(x + (i_2 * 4));
        ((float2*)(&__7))[0] = __half22float2(((half2*)(&v__1))[0]);
        ((float2*)(&__7))[1] = __half22float2(((half2*)(&v__1))[1]);
        float4 v__2 = make_float4(ss[0], ss[0], ss[0], ss[0]);
        *(float2*)(&(__6.x)) = tl::mul2(*(float2*)(&(__7.x)), *(float2*)(&(v__2.x)));
        *(float2*)(&(__6.z)) = tl::mul2(*(float2*)(&(__7.z)), *(float2*)(&(v__2.z)));
      float4 __8;
      uint2 v__3 = *(uint2*)(W_local_cast + 0);
      ((float2*)(&__8))[0] = __half22float2(((half2*)(&v__3))[0]);
      ((float2*)(&__8))[1] = __half22float2(((half2*)(&v__3))[1]);
      *(float2*)(&(__5.x)) = tl::mul2(*(float2*)(&(__6.x)), *(float2*)(&(__8.x)));
      *(float2*)(&(__5.z)) = tl::mul2(*(float2*)(&(__6.z)), *(float2*)(&(__8.z)));
    ((half2*)(&__4))[0] = __float22half2_rn(((float2*)(&__5))[0]);
    ((half2*)(&__4))[1] = __float22half2_rn(((float2*)(&__5))[1]);
    *(uint2*)(x + (i_2 * 4)) = __4;
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 9; ++i_3) {
    *(uint2*)(Y + (((((int)blockIdx.x) * 4608) + (i_3 * 512)) + (((int)threadIdx.x) * 4))) = *(uint2*)(x + (i_3 * 4));
  }
}

