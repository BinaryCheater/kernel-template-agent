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

extern "C" __global__ void rms_kernel(const float* __restrict__ W, const float* __restrict__ X, float* __restrict__ Y);
extern "C" __global__ void __launch_bounds__(128, 1) rms_kernel(const float* __restrict__ W, const float* __restrict__ X, float* __restrict__ Y) {
  float x[32];
  float sq[32];
  float ss[1];
  extern __shared__ __align__(1024) float workspace[];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    *(ulonglong4*)(x + (i * 8)) = tl::load_global_256(&(*(ulonglong4*)(X + (((((int)blockIdx.x) * 4096) + (i * 1024)) + (((int)threadIdx.x) * 8)))));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    float4 __1;
      float4 v_ = *(float4*)(x + (i_1 * 4));
      *(float2*)(&(__1.x)) = tl::mul2(*(float2*)(&(v_.x)), *(float2*)(&(v_.x)));
      *(float2*)(&(__1.z)) = tl::mul2(*(float2*)(&(v_.z)), *(float2*)(&(v_.z)));
    *(float4*)(sq + (i_1 * 4)) = __1;
  }
  ss[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv = 0; rv < 32; ++rv) {
    ss[0] = (ss[0] + sq[(((rv & 3) * 8) + (rv >> 2))]);
  }
  ss[0] = tl::AllReduce<tl::SumOp, 128, 1, 0, tl::NamedBarrier<128>>::run(ss[0], (&(workspace[0])));
  ss[0] = rsqrtf(((ss[0] / 0x1p+12f/*4.096000e+03*/) + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/));
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    ulonglong4 __2;
      ulonglong4 __3;
        ulonglong4 v__1 = *(ulonglong4*)(x + (i_2 * 8));
        ulonglong4 v__2 = make_ulonglong4(*(unsigned long long*)&make_float2(ss[0], ss[0]), *(unsigned long long*)&make_float2(ss[0], ss[0]), *(unsigned long long*)&make_float2(ss[0], ss[0]), *(unsigned long long*)&make_float2(ss[0], ss[0]));
        *(float2*)(&(__3.x)) = tl::mul2(*(float2*)(&(v__1.x)), *(float2*)(&(v__2.x)));
        *(float2*)(&(__3.y)) = tl::mul2(*(float2*)(&(v__1.y)), *(float2*)(&(v__2.y)));
        *(float2*)(&(__3.z)) = tl::mul2(*(float2*)(&(v__1.z)), *(float2*)(&(v__2.z)));
        *(float2*)(&(__3.w)) = tl::mul2(*(float2*)(&(v__1.w)), *(float2*)(&(v__2.w)));
      ulonglong4 v__3 = tl::load_global_256(&(*(ulonglong4*)(W + ((i_2 * 1024) + (((int)threadIdx.x) * 8)))));
      *(float2*)(&(__2.x)) = tl::mul2(*(float2*)(&(__3.x)), *(float2*)(&(v__3.x)));
      *(float2*)(&(__2.y)) = tl::mul2(*(float2*)(&(__3.y)), *(float2*)(&(v__3.y)));
      *(float2*)(&(__2.z)) = tl::mul2(*(float2*)(&(__3.z)), *(float2*)(&(v__3.z)));
      *(float2*)(&(__2.w)) = tl::mul2(*(float2*)(&(__3.w)), *(float2*)(&(v__3.w)));
    *(ulonglong4*)(x + (i_2 * 8)) = __2;
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 4; ++i_3) {
    tl::store_global_256(&(*(ulonglong4*)(Y + (((((int)blockIdx.x) * 4096) + (i_3 * 1024)) + (((int)threadIdx.x) * 8)))), *(ulonglong4*)(x + (i_3 * 8)));
  }
}

