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

extern "C" __global__ void rms_kernel(half_t* __restrict__ Y);
extern "C" __global__ void __launch_bounds__(128, 1) rms_kernel(half_t* __restrict__ Y) {
  half_t x[44];
  float sq[44];
  float ss[1];
  extern __shared__ __align__(1024) float workspace[];
  #pragma unroll
  for (int i = 0; i < 11; ++i) {
    uint2 __1;
    int4 v_ = make_int4((((i * 512) + (((int)threadIdx.x) * 4)))+(1*0), (((i * 512) + (((int)threadIdx.x) * 4)))+(1*1), (((i * 512) + (((int)threadIdx.x) * 4)))+(1*2), (((i * 512) + (((int)threadIdx.x) * 4)))+(1*3));
    ((half2*)(&(__1.x)))->x = (half_t)(v_.x);
    ((half2*)(&(__1.x)))->y = (half_t)(v_.y);
    ((half2*)(&(__1.y)))->x = (half_t)(v_.z);
    ((half2*)(&(__1.y)))->y = (half_t)(v_.w);
    *(uint2*)(x + (i * 4)) = __1;
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 11; ++i_1) {
    float4 __2;
      float4 __3;
      uint2 v__1 = *(uint2*)(x + (i_1 * 4));
      ((float2*)(&__3))[0] = __half22float2(((half2*)(&v__1))[0]);
      ((float2*)(&__3))[1] = __half22float2(((half2*)(&v__1))[1]);
      float4 __4;
      ((float2*)(&__4))[0] = __half22float2(((half2*)(&v__1))[0]);
      ((float2*)(&__4))[1] = __half22float2(((half2*)(&v__1))[1]);
      *(float2*)(&(__2.x)) = tl::mul2(*(float2*)(&(__3.x)), *(float2*)(&(__4.x)));
      *(float2*)(&(__2.z)) = tl::mul2(*(float2*)(&(__3.z)), *(float2*)(&(__4.z)));
    *(float4*)(sq + (i_1 * 4)) = __2;
  }
  ss[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv = 0; rv < 44; ++rv) {
    ss[0] = (ss[0] + sq[(((rv % 11) * 4) + (rv / 11))]);
  }
  ss[0] = tl::AllReduce<tl::SumOp, 128, 1, 0, tl::NamedBarrier<128>>::run(ss[0], (&(workspace[0])));
  ss[0] = rsqrtf(((ss[0] / 0x1.6p+12f/*5.632000e+03*/) + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/));
  #pragma unroll
  for (int i_2 = 0; i_2 < 11; ++i_2) {
    float4 __5;
    uint2 v__2 = *(uint2*)(x + (i_2 * 4));
    ((float2*)(&__5))[0] = __half22float2(((half2*)(&v__2))[0]);
    ((float2*)(&__5))[1] = __half22float2(((half2*)(&v__2))[1]);
    float4 wj = __5;
    uint2 __6;
    float4 __7;
      float4 __8;
        float4 __9;
        ((float2*)(&__9))[0] = __half22float2(((half2*)(&v__2))[0]);
        ((float2*)(&__9))[1] = __half22float2(((half2*)(&v__2))[1]);
        float4 v__3 = make_float4(ss[0], ss[0], ss[0], ss[0]);
        *(float2*)(&(__8.x)) = tl::mul2(*(float2*)(&(__9.x)), *(float2*)(&(v__3.x)));
        *(float2*)(&(__8.z)) = tl::mul2(*(float2*)(&(__9.z)), *(float2*)(&(v__3.z)));
      *(float2*)(&(__7.x)) = tl::mul2(*(float2*)(&(__8.x)), *(float2*)(&(wj.x)));
      *(float2*)(&(__7.z)) = tl::mul2(*(float2*)(&(__8.z)), *(float2*)(&(wj.z)));
    ((half2*)(&__6))[0] = __float22half2_rn(((float2*)(&__7))[0]);
    ((half2*)(&__6))[1] = __float22half2_rn(((float2*)(&__7))[1]);
    *(uint2*)(x + (i_2 * 4)) = __6;
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 11; ++i_3) {
    *(uint2*)(Y + (((((int)blockIdx.x) * 5632) + (i_3 * 512)) + (((int)threadIdx.x) * 4))) = *(uint2*)(x + (i_3 * 4));
  }
}

