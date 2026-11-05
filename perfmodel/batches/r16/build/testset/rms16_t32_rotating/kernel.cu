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
extern "C" __global__ void __launch_bounds__(32, 1) rms_kernel(const half_t* __restrict__ W, const half_t* __restrict__ X, half_t* __restrict__ Y) {
  half_t x[70];
  float sq[70];
  float ss[1];
  #pragma unroll
  for (int i = 0; i < 35; ++i) {
    *(uint1*)(x + (i * 2)) = *(uint1*)(X + (((((int)blockIdx.x) * 2240) + (i * 64)) + (((int)threadIdx.x) * 2)));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 35; ++i_1) {
    float2 __1;
      float2 __2;
      uint1 v_ = *(uint1*)(x + (i_1 * 2));
      ((float2*)(&__2))[0] = __half22float2(((half2*)(&v_))[0]);
      float2 __3;
      ((float2*)(&__3))[0] = __half22float2(((half2*)(&v_))[0]);
      *(float2*)(&(__1.x)) = tl::mul2(*(float2*)(&(__2.x)), *(float2*)(&(__3.x)));
    *(float2*)(sq + (i_1 * 2)) = __1;
  }
  ss[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv = 0; rv < 70; ++rv) {
    ss[0] = (ss[0] + sq[(((rv % 35) * 2) + (rv / 35))]);
  }
  ss[0] = tl::AllReduce<tl::SumOp, 32, 1, 0, tl::NamedBarrier<32>>::run(ss[0]);
  ss[0] = rsqrtf(((ss[0] / 0x1.18p+11f/*2.240000e+03*/) + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/));
  #pragma unroll
  for (int i_2 = 0; i_2 < 35; ++i_2) {
    half_t W_local_cast[2];
    *(uint1*)(W_local_cast + 0) = *(uint1*)(W + ((i_2 * 64) + (((int)threadIdx.x) * 2)));
    uint1 __4;
    float2 __5;
      float2 __6;
        float2 __7;
        uint1 v__1 = *(uint1*)(x + (i_2 * 2));
        ((float2*)(&__7))[0] = __half22float2(((half2*)(&v__1))[0]);
        float2 v__2 = make_float2(ss[0], ss[0]);
        *(float2*)(&(__6.x)) = tl::mul2(*(float2*)(&(__7.x)), *(float2*)(&(v__2.x)));
      float2 __8;
      uint1 v__3 = *(uint1*)(W_local_cast + 0);
      ((float2*)(&__8))[0] = __half22float2(((half2*)(&v__3))[0]);
      *(float2*)(&(__5.x)) = tl::mul2(*(float2*)(&(__6.x)), *(float2*)(&(__8.x)));
    ((half2*)(&__4))[0] = __float22half2_rn(((float2*)(&__5))[0]);
    *(uint1*)(x + (i_2 * 2)) = __4;
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 35; ++i_3) {
    *(uint1*)(Y + (((((int)blockIdx.x) * 2240) + (i_3 * 64)) + (((int)threadIdx.x) * 2))) = *(uint1*)(x + (i_3 * 2));
  }
}

