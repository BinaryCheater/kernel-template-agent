#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/copy_sm90.h>
#include <tl_templates/cuda/copy_sm100.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void rms_kernel(const half_t* __restrict__ W, const half_t* __restrict__ X, half_t* __restrict__ Y);
extern "C" __global__ void __launch_bounds__(16, 1) rms_kernel(const half_t* __restrict__ W, const half_t* __restrict__ X, half_t* __restrict__ Y) {
  half_t x[352];
  float sq[352];
  float ss[1];
  #pragma unroll
  for (int i = 0; i < 22; ++i) {
    *(ulonglong4*)(x + (i * 16)) = tl::load_global_256(&(*(ulonglong4*)(X + (((((int)blockIdx.x) * 5632) + (i * 256)) + (((int)threadIdx.x) * 16)))));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 88; ++i_1) {
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
  for (int rv = 0; rv < 352; ++rv) {
    ss[0] = (ss[0] + sq[(((rv % 22) * 16) + (rv / 22))]);
  }
  ss[0] = tl::AllReduce<tl::SumOp, 16, 1, 0, tl::NamedBarrier<16>>::run(ss[0]);
  ss[0] = rsqrtf(((ss[0] / 0x1.6p+12f/*5.632000e+03*/) + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/));
  #pragma unroll
  for (int i_2 = 0; i_2 < 22; ++i_2) {
    half_t W_local_cast[16];
    *(ulonglong4*)(W_local_cast + 0) = tl::load_global_256(&(*(ulonglong4*)(W + ((i_2 * 256) + (((int)threadIdx.x) * 16)))));
    for (int vec = 0; vec < 4; ++vec) {
      uint2 __4;
      float4 __5;
        float4 __6;
          float4 __7;
          uint2 v__1 = *(uint2*)(x + ((i_2 * 16) + (vec * 4)));
          ((float2*)(&__7))[0] = __half22float2(((half2*)(&v__1))[0]);
          ((float2*)(&__7))[1] = __half22float2(((half2*)(&v__1))[1]);
          float4 v__2 = make_float4(ss[0], ss[0], ss[0], ss[0]);
          *(float2*)(&(__6.x)) = tl::mul2(*(float2*)(&(__7.x)), *(float2*)(&(v__2.x)));
          *(float2*)(&(__6.z)) = tl::mul2(*(float2*)(&(__7.z)), *(float2*)(&(v__2.z)));
        float4 __8;
        uint2 v__3 = *(uint2*)(W_local_cast + (vec * 4));
        ((float2*)(&__8))[0] = __half22float2(((half2*)(&v__3))[0]);
        ((float2*)(&__8))[1] = __half22float2(((half2*)(&v__3))[1]);
        *(float2*)(&(__5.x)) = tl::mul2(*(float2*)(&(__6.x)), *(float2*)(&(__8.x)));
        *(float2*)(&(__5.z)) = tl::mul2(*(float2*)(&(__6.z)), *(float2*)(&(__8.z)));
      ((half2*)(&__4))[0] = __float22half2_rn(((float2*)(&__5))[0]);
      ((half2*)(&__4))[1] = __float22half2_rn(((float2*)(&__5))[1]);
      *(uint2*)(x + ((i_2 * 16) + (vec * 4))) = __4;
    }
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 22; ++i_3) {
    tl::store_global_256(&(*(ulonglong4*)(Y + (((((int)blockIdx.x) * 5632) + (i_3 * 256)) + (((int)threadIdx.x) * 16)))), *(ulonglong4*)(x + (i_3 * 16)));
  }
}

