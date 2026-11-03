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
  half_t x[8];
  float sq[8];
  float2 ss_pack[1];
  float ss[1];
  extern __shared__ __align__(1024) float workspace[];
  *(uint4*)(x + 0) = *(uint4*)(X + ((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 8)));
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    float4 __1;
      float4 __2;
      uint2 v_ = *(uint2*)(x + (i * 4));
      ((float2*)(&__2))[0] = __half22float2(((half2*)(&v_))[0]);
      ((float2*)(&__2))[1] = __half22float2(((half2*)(&v_))[1]);
      float4 __3;
      ((float2*)(&__3))[0] = __half22float2(((half2*)(&v_))[0]);
      ((float2*)(&__3))[1] = __half22float2(((half2*)(&v_))[1]);
      *(float2*)(&(__1.x)) = tl::mul2(*(float2*)(&(__2.x)), *(float2*)(&(__3.x)));
      *(float2*)(&(__1.z)) = tl::mul2(*(float2*)(&(__2.z)), *(float2*)(&(__3.z)));
    *(float4*)(sq + (i * 4)) = __1;
  }
  float broadcast_var = 0x0p+0f/*0.000000e+00*/;
  ss_pack[0] = make_float2(broadcast_var, broadcast_var);
  #pragma unroll
  for (int rv = 0; rv < 4; ++rv) {
    ss_pack[0] = tl::add2(ss_pack[0], *(float2*)(sq + (rv * 2)));
  }
  ss[0] = ((ss_pack[0]).x + (ss_pack[0]).y);
  ss[0] = tl::AllReduce<tl::SumOp, 128, 1, 0, tl::NamedBarrier<128>>::run(ss[0], (&(workspace[0])));
  ss[0] = rsqrtf(((ss[0] / 0x1p+10f/*1.024000e+03*/) + 0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/));
  {
    half_t W_local_cast[8];
    *(uint4*)(W_local_cast + 0) = *(uint4*)(W + (((int)threadIdx.x) * 8));
    for (int i_1 = 0; i_1 < 2; ++i_1) {
      uint2 __4;
      float4 __5;
        float4 __6;
          float4 __7;
          uint2 v__1 = *(uint2*)(x + (i_1 * 4));
          ((float2*)(&__7))[0] = __half22float2(((half2*)(&v__1))[0]);
          ((float2*)(&__7))[1] = __half22float2(((half2*)(&v__1))[1]);
          float4 v__2 = make_float4(ss[0], ss[0], ss[0], ss[0]);
          *(float2*)(&(__6.x)) = tl::mul2(*(float2*)(&(__7.x)), *(float2*)(&(v__2.x)));
          *(float2*)(&(__6.z)) = tl::mul2(*(float2*)(&(__7.z)), *(float2*)(&(v__2.z)));
        float4 __8;
        uint2 v__3 = *(uint2*)(W_local_cast + (i_1 * 4));
        ((float2*)(&__8))[0] = __half22float2(((half2*)(&v__3))[0]);
        ((float2*)(&__8))[1] = __half22float2(((half2*)(&v__3))[1]);
        *(float2*)(&(__5.x)) = tl::mul2(*(float2*)(&(__6.x)), *(float2*)(&(__8.x)));
        *(float2*)(&(__5.z)) = tl::mul2(*(float2*)(&(__6.z)), *(float2*)(&(__8.z)));
      ((half2*)(&__4))[0] = __float22half2_rn(((float2*)(&__5))[0]);
      ((half2*)(&__4))[1] = __float22half2_rn(((float2*)(&__5))[1]);
      *(uint2*)(x + (i_1 * 4)) = __4;
    }
  }
  *(uint4*)(Y + ((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 8))) = *(uint4*)(x + 0);
}

