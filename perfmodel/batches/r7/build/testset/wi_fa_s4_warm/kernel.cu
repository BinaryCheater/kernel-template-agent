#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/copy.h>
#include <math_constants.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void fa_kernel(const half_t* __restrict__ K, half_t* __restrict__ Out, const half_t* __restrict__ Q, const half_t* __restrict__ V);
extern "C" __global__ void __launch_bounds__(128, 1) fa_kernel(const half_t* __restrict__ K, half_t* __restrict__ Out, const half_t* __restrict__ Q, const half_t* __restrict__ V) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* O_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* Q_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* K_shared = ((void*)((char*)buf_dyn_shmem + 16384));
  void* V_shared = ((void*)((char*)buf_dyn_shmem + 81920));
  float acc_o[64];
  float lsum[2];
  float m[2];
  float acc_s[32];
  float m_prev[2];
  float sc[2];
  half_t acc_s_cast[32];
  float ssum[2];
  float m_clear[2];
  float m_clear_1[2];
  float m_clear_2[2];
  float m_clear_3[2];
  float m_clear_4[2];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    *(uint4*)(((half_t*)Q_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(Q + (((((((int)blockIdx.x) * 65536) + (i * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 16; ++i_1) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_o + (i_1 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(lsum + 0) = make_float2(broadcast_var_1, broadcast_var_1);
  float broadcast_var_2 = -CUDART_INF_F;
  *(float2*)(m + 0) = make_float2(broadcast_var_2, broadcast_var_2);
  __syncthreads();
  #pragma unroll
  for (int i_2 = 0; i_2 < 8; ++i_2) {
    tl::cp_async_gs<16>((&(((half_t*)K_shared)[((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_2 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(K[((((i_2 * 8192) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_3 = 0; i_3 < 8; ++i_3) {
    tl::cp_async_gs<16>((&(((half_t*)V_shared)[((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_3 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(V[((((i_3 * 8192) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_4 = 0; i_4 < 8; ++i_4) {
    tl::cp_async_gs<16>((&(((half_t*)K_shared)[(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_4 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(K[(((((i_4 * 8192) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 65536)])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_5 = 0; i_5 < 8; ++i_5) {
    tl::cp_async_gs<16>((&(((half_t*)V_shared)[(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_5 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(V[(((((i_5 * 8192) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 65536)])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_6 = 0; i_6 < 8; ++i_6) {
    tl::cp_async_gs<16>((&(((half_t*)K_shared)[(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_6 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)])), (&(K[(((((i_6 * 8192) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 131072)])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_7 = 0; i_7 < 8; ++i_7) {
    tl::cp_async_gs<16>((&(((half_t*)V_shared)[(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_7 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)])), (&(V[(((((i_7 * 8192) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 131072)])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_8 = 0; i_8 < 8; ++i_8) {
    tl::cp_async_gs<16>((&(((half_t*)K_shared)[(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_8 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 24576)])), (&(K[(((((i_8 * 8192) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 196608)])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_9 = 0; i_9 < 8; ++i_9) {
    tl::cp_async_gs<16>((&(((half_t*)V_shared)[(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_9 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 24576)])), (&(V[(((((i_9 * 8192) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 196608)])));
  }
  tl::cp_async_commit();
  for (int k = 0; k < 28; ++k) {
    #pragma unroll
    for (int i_10 = 0; i_10 < 8; ++i_10) {
      float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_s + (i_10 * 4)) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
    }
    tl::cp_async_wait<7>();
    __syncthreads();
    {
      half_t A_local[8];
      half_t B_local[32];
      for (int ki = 0; ki < 8; ++ki) {
        tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[(((((((ki >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local[0])));
        #pragma unroll
        for (int i_11 = 0; i_11 < 4; ++i_11) {
          tl::ptx_ldmatrix_x4((&(((half_t*)K_shared)[(((((((((k & 3) * 8192) + ((ki >> 2) * 4096)) + (i_11 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[(i_11 * 8)])));
        }
        for (int j = 0; j < 4; ++j) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (j * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_12 = 0; i_12 < 8; ++i_12) {
      tl::cp_async_gs<16>((&(((half_t*)K_shared)[((((((((k & 3) * 8192) + (((((int)threadIdx.x) & 15) >> 3) * 4096)) + (i_12 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(K[((((((k * 65536) + (i_12 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 262144)])));
    }
    tl::cp_async_commit();
    *(float2*)(m_prev + 0) = *(float2*)(m + 0);
    float broadcast_var_4 = -CUDART_INF_F;
    *(float2*)(m + 0) = make_float2(broadcast_var_4, broadcast_var_4);
    #pragma unroll
    for (int i_13 = 0; i_13 < 2; ++i_13) {
      m_clear[i_13] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 16; ++rv) {
        m_clear[i_13] = max(m_clear[i_13], acc_s[((((rv & 7) * 4) + (i_13 * 2)) + (rv >> 3))]);
      }
      m_clear[i_13] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(m_clear[i_13]);
      m[i_13] = max(m[i_13], m_clear[i_13]);
    }
    float broadcast_var_5 = 0x1.0527dbd5cafffp-3f/*1.275174e-01*/;
    float broadcast_var_6 = 0x1.0527dbd5cafffp-3f/*1.275174e-01*/;
    float2 __1;
    float2 __2;
      float2 __3;
        float2 v_ = *(float2*)(m_prev + 0);
        float2 v__1 = make_float2(broadcast_var_5, broadcast_var_5);
        *(float2*)(&(__3.x)) = tl::mul2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)));
      float2 __4;
        float2 v__2 = *(float2*)(m + 0);
        float2 v__3 = make_float2(broadcast_var_6, broadcast_var_6);
        *(float2*)(&(__4.x)) = tl::mul2(*(float2*)(&(v__2.x)), *(float2*)(&(v__3.x)));
      *(float2*)(&(__2.x)) = tl::sub2(*(float2*)(&(__3.x)), *(float2*)(&(__4.x)));
    __1.x = exp2f(__2.x);
    __1.y = exp2f(__2.y);
    *(float2*)(sc + 0) = __1;
    #pragma unroll
    for (int i_14 = 0; i_14 < 32; ++i_14) {
      float2 __5;
        float2 v__4 = *(float2*)(acc_o + (i_14 * 2));
        float2 v__5 = make_float2(sc[(i_14 & 1)], sc[(i_14 & 1)]);
        *(float2*)(&(__5.x)) = tl::mul2(*(float2*)(&(v__4.x)), *(float2*)(&(v__5.x)));
      *(float2*)(acc_o + (i_14 * 2)) = __5;
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 16; ++i_15) {
      float broadcast_var_7 = 0x1.0527dbd5cafffp-3f/*1.275174e-01*/;
      float2 __6;
      float2 __7;
        float2 __8;
          float2 v__6 = *(float2*)(acc_s + (i_15 * 2));
          float2 v__7 = make_float2(broadcast_var_7, broadcast_var_7);
          *(float2*)(&(__8.x)) = tl::mul2(*(float2*)(&(v__6.x)), *(float2*)(&(v__7.x)));
        float2 v__8 = make_float2((m[(i_15 & 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/), (m[(i_15 & 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/));
        *(float2*)(&(__7.x)) = tl::sub2(*(float2*)(&(__8.x)), *(float2*)(&(v__8.x)));
      __6.x = exp2f(__7.x);
      __6.y = exp2f(__7.y);
      *(float2*)(acc_s + (i_15 * 2)) = __6;
    }
    #pragma unroll
    for (int i_16 = 0; i_16 < 8; ++i_16) {
      uint2 __9;
      float4 v__9 = *(float4*)(acc_s + (i_16 * 4));
      ((half2*)(&__9))[0] = __float22half2_rn(((float2*)(&v__9))[0]);
      ((half2*)(&__9))[1] = __float22half2_rn(((float2*)(&v__9))[1]);
      *(uint2*)(acc_s_cast + (i_16 * 4)) = __9;
    }
    tl::cp_async_wait<7>();
    __syncthreads();
    {
      half_t B_local_1[64];
      for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
        #pragma unroll
        for (int i_17 = 0; i_17 < 8; ++i_17) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)V_shared)[((((((((k & 3) * 8192) + ((i_17 >> 2) * 4096)) + (ki_1 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_17 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_17 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_1[(i_17 * 8)])));
        }
        for (int j_1 = 0; j_1 < 8; ++j_1) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_1 * 8)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_1 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_1 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_18 = 0; i_18 < 8; ++i_18) {
      tl::cp_async_gs<16>((&(((half_t*)V_shared)[((((((((k & 3) * 8192) + (((((int)threadIdx.x) & 15) >> 3) * 4096)) + (i_18 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(V[((((((k * 65536) + (i_18 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 262144)])));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_19 = 0; i_19 < 2; ++i_19) {
      ssum[i_19] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
        ssum[i_19] = (ssum[i_19] + acc_s[((((rv_1 & 7) * 4) + (i_19 * 2)) + (rv_1 >> 3))]);
      }
      ssum[i_19] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(ssum[i_19]);
    }
    float2 __10;
      float2 v__10 = *(float2*)(lsum + 0);
      float2 v__11 = *(float2*)(sc + 0);
      float2 v__12 = *(float2*)(ssum + 0);
      *(float2*)(&(__10.x)) = tl::fma2(*(float2*)(&(v__10.x)), *(float2*)(&(v__11.x)), *(float2*)(&(v__12.x)));
    *(float2*)(lsum + 0) = __10;
  }
  #pragma unroll
  for (int i_20 = 0; i_20 < 8; ++i_20) {
    float broadcast_var_8 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_s + (i_20 * 4)) = make_float4(broadcast_var_8, broadcast_var_8, broadcast_var_8, broadcast_var_8);
  }
  tl::cp_async_wait<7>();
  __syncthreads();
  {
    half_t A_local_1[8];
    half_t B_local_2[32];
    for (int ki_2 = 0; ki_2 < 8; ++ki_2) {
      tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[(((((((ki_2 >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_1[0])));
      #pragma unroll
      for (int i_21 = 0; i_21 < 4; ++i_21) {
        tl::ptx_ldmatrix_x4((&(((half_t*)K_shared)[((((((((ki_2 >> 2) * 4096) + (i_21 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_2[(i_21 * 8)])));
      }
      for (int j_2 = 0; j_2 < 4; ++j_2) {
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (j_2 * 8)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_2 + (j_2 * 8)));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((j_2 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_2 + ((j_2 * 8) + 4)));
      }
    }
  }
  *(float2*)(m_prev + 0) = *(float2*)(m + 0);
  float broadcast_var_9 = -CUDART_INF_F;
  *(float2*)(m + 0) = make_float2(broadcast_var_9, broadcast_var_9);
  #pragma unroll
  for (int i_22 = 0; i_22 < 2; ++i_22) {
    m_clear_1[i_22] = -CUDART_INF_F;
    #pragma unroll
    for (int rv_2 = 0; rv_2 < 16; ++rv_2) {
      m_clear_1[i_22] = max(m_clear_1[i_22], acc_s[((((rv_2 & 7) * 4) + (i_22 * 2)) + (rv_2 >> 3))]);
    }
    m_clear_1[i_22] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(m_clear_1[i_22]);
    m[i_22] = max(m[i_22], m_clear_1[i_22]);
  }
  float broadcast_var_10 = 0x1.0527dbd5cafffp-3f/*1.275174e-01*/;
  float broadcast_var_11 = 0x1.0527dbd5cafffp-3f/*1.275174e-01*/;
  float2 __11;
  float2 __12;
    float2 __13;
      float2 v__13 = *(float2*)(m_prev + 0);
      float2 v__14 = make_float2(broadcast_var_10, broadcast_var_10);
      *(float2*)(&(__13.x)) = tl::mul2(*(float2*)(&(v__13.x)), *(float2*)(&(v__14.x)));
    float2 __14;
      float2 v__15 = *(float2*)(m + 0);
      float2 v__16 = make_float2(broadcast_var_11, broadcast_var_11);
      *(float2*)(&(__14.x)) = tl::mul2(*(float2*)(&(v__15.x)), *(float2*)(&(v__16.x)));
    *(float2*)(&(__12.x)) = tl::sub2(*(float2*)(&(__13.x)), *(float2*)(&(__14.x)));
  __11.x = exp2f(__12.x);
  __11.y = exp2f(__12.y);
  *(float2*)(sc + 0) = __11;
  #pragma unroll
  for (int i_23 = 0; i_23 < 32; ++i_23) {
    float2 __15;
      float2 v__17 = *(float2*)(acc_o + (i_23 * 2));
      float2 v__18 = make_float2(sc[(i_23 & 1)], sc[(i_23 & 1)]);
      *(float2*)(&(__15.x)) = tl::mul2(*(float2*)(&(v__17.x)), *(float2*)(&(v__18.x)));
    *(float2*)(acc_o + (i_23 * 2)) = __15;
  }
  #pragma unroll
  for (int i_24 = 0; i_24 < 16; ++i_24) {
    float broadcast_var_12 = 0x1.0527dbd5cafffp-3f/*1.275174e-01*/;
    float2 __16;
    float2 __17;
      float2 __18;
        float2 v__19 = *(float2*)(acc_s + (i_24 * 2));
        float2 v__20 = make_float2(broadcast_var_12, broadcast_var_12);
        *(float2*)(&(__18.x)) = tl::mul2(*(float2*)(&(v__19.x)), *(float2*)(&(v__20.x)));
      float2 v__21 = make_float2((m[(i_24 & 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/), (m[(i_24 & 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/));
      *(float2*)(&(__17.x)) = tl::sub2(*(float2*)(&(__18.x)), *(float2*)(&(v__21.x)));
    __16.x = exp2f(__17.x);
    __16.y = exp2f(__17.y);
    *(float2*)(acc_s + (i_24 * 2)) = __16;
  }
  #pragma unroll
  for (int i_25 = 0; i_25 < 8; ++i_25) {
    uint2 __19;
    float4 v__22 = *(float4*)(acc_s + (i_25 * 4));
    ((half2*)(&__19))[0] = __float22half2_rn(((float2*)(&v__22))[0]);
    ((half2*)(&__19))[1] = __float22half2_rn(((float2*)(&v__22))[1]);
    *(uint2*)(acc_s_cast + (i_25 * 4)) = __19;
  }
  tl::cp_async_wait<6>();
  __syncthreads();
  {
    half_t B_local_3[64];
    for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
      #pragma unroll
      for (int i_26 = 0; i_26 < 8; ++i_26) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)V_shared)[(((((((i_26 >> 2) * 4096) + (ki_3 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_26 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_26 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_3[(i_26 * 8)])));
      }
      for (int j_3 = 0; j_3 < 8; ++j_3) {
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_3 * 8)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_3 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + (j_3 * 8)));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_3 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_3 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + ((j_3 * 8) + 4)));
      }
    }
  }
  #pragma unroll
  for (int i_27 = 0; i_27 < 2; ++i_27) {
    ssum[i_27] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_3 = 0; rv_3 < 16; ++rv_3) {
      ssum[i_27] = (ssum[i_27] + acc_s[((((rv_3 & 7) * 4) + (i_27 * 2)) + (rv_3 >> 3))]);
    }
    ssum[i_27] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(ssum[i_27]);
  }
  float2 __20;
    float2 v__23 = *(float2*)(lsum + 0);
    float2 v__24 = *(float2*)(sc + 0);
    float2 v__25 = *(float2*)(ssum + 0);
    *(float2*)(&(__20.x)) = tl::fma2(*(float2*)(&(v__23.x)), *(float2*)(&(v__24.x)), *(float2*)(&(v__25.x)));
  *(float2*)(lsum + 0) = __20;
  #pragma unroll
  for (int i_28 = 0; i_28 < 8; ++i_28) {
    float broadcast_var_13 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_s + (i_28 * 4)) = make_float4(broadcast_var_13, broadcast_var_13, broadcast_var_13, broadcast_var_13);
  }
  tl::cp_async_wait<5>();
  __syncthreads();
  {
    half_t A_local_2[8];
    half_t B_local_4[32];
    for (int ki_4 = 0; ki_4 < 8; ++ki_4) {
      tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[(((((((ki_4 >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_4 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_2[0])));
      #pragma unroll
      for (int i_29 = 0; i_29 < 4; ++i_29) {
        tl::ptx_ldmatrix_x4((&(((half_t*)K_shared)[(((((((((ki_4 >> 2) * 4096) + (i_29 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_4 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(B_local_4[(i_29 * 8)])));
      }
      for (int j_4 = 0; j_4 < 4; ++j_4) {
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (j_4 * 8)), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_4 + (j_4 * 8)));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((j_4 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_4 + ((j_4 * 8) + 4)));
      }
    }
  }
  *(float2*)(m_prev + 0) = *(float2*)(m + 0);
  float broadcast_var_14 = -CUDART_INF_F;
  *(float2*)(m + 0) = make_float2(broadcast_var_14, broadcast_var_14);
  #pragma unroll
  for (int i_30 = 0; i_30 < 2; ++i_30) {
    m_clear_2[i_30] = -CUDART_INF_F;
    #pragma unroll
    for (int rv_4 = 0; rv_4 < 16; ++rv_4) {
      m_clear_2[i_30] = max(m_clear_2[i_30], acc_s[((((rv_4 & 7) * 4) + (i_30 * 2)) + (rv_4 >> 3))]);
    }
    m_clear_2[i_30] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(m_clear_2[i_30]);
    m[i_30] = max(m[i_30], m_clear_2[i_30]);
  }
  float broadcast_var_15 = 0x1.0527dbd5cafffp-3f/*1.275174e-01*/;
  float broadcast_var_16 = 0x1.0527dbd5cafffp-3f/*1.275174e-01*/;
  float2 __21;
  float2 __22;
    float2 __23;
      float2 v__26 = *(float2*)(m_prev + 0);
      float2 v__27 = make_float2(broadcast_var_15, broadcast_var_15);
      *(float2*)(&(__23.x)) = tl::mul2(*(float2*)(&(v__26.x)), *(float2*)(&(v__27.x)));
    float2 __24;
      float2 v__28 = *(float2*)(m + 0);
      float2 v__29 = make_float2(broadcast_var_16, broadcast_var_16);
      *(float2*)(&(__24.x)) = tl::mul2(*(float2*)(&(v__28.x)), *(float2*)(&(v__29.x)));
    *(float2*)(&(__22.x)) = tl::sub2(*(float2*)(&(__23.x)), *(float2*)(&(__24.x)));
  __21.x = exp2f(__22.x);
  __21.y = exp2f(__22.y);
  *(float2*)(sc + 0) = __21;
  #pragma unroll
  for (int i_31 = 0; i_31 < 32; ++i_31) {
    float2 __25;
      float2 v__30 = *(float2*)(acc_o + (i_31 * 2));
      float2 v__31 = make_float2(sc[(i_31 & 1)], sc[(i_31 & 1)]);
      *(float2*)(&(__25.x)) = tl::mul2(*(float2*)(&(v__30.x)), *(float2*)(&(v__31.x)));
    *(float2*)(acc_o + (i_31 * 2)) = __25;
  }
  #pragma unroll
  for (int i_32 = 0; i_32 < 16; ++i_32) {
    float broadcast_var_17 = 0x1.0527dbd5cafffp-3f/*1.275174e-01*/;
    float2 __26;
    float2 __27;
      float2 __28;
        float2 v__32 = *(float2*)(acc_s + (i_32 * 2));
        float2 v__33 = make_float2(broadcast_var_17, broadcast_var_17);
        *(float2*)(&(__28.x)) = tl::mul2(*(float2*)(&(v__32.x)), *(float2*)(&(v__33.x)));
      float2 v__34 = make_float2((m[(i_32 & 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/), (m[(i_32 & 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/));
      *(float2*)(&(__27.x)) = tl::sub2(*(float2*)(&(__28.x)), *(float2*)(&(v__34.x)));
    __26.x = exp2f(__27.x);
    __26.y = exp2f(__27.y);
    *(float2*)(acc_s + (i_32 * 2)) = __26;
  }
  #pragma unroll
  for (int i_33 = 0; i_33 < 8; ++i_33) {
    uint2 __29;
    float4 v__35 = *(float4*)(acc_s + (i_33 * 4));
    ((half2*)(&__29))[0] = __float22half2_rn(((float2*)(&v__35))[0]);
    ((half2*)(&__29))[1] = __float22half2_rn(((float2*)(&v__35))[1]);
    *(uint2*)(acc_s_cast + (i_33 * 4)) = __29;
  }
  tl::cp_async_wait<4>();
  __syncthreads();
  {
    half_t B_local_5[64];
    for (int ki_5 = 0; ki_5 < 4; ++ki_5) {
      #pragma unroll
      for (int i_34 = 0; i_34 < 8; ++i_34) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)V_shared)[((((((((i_34 >> 2) * 4096) + (ki_5 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_34 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_34 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)])), (&(B_local_5[(i_34 * 8)])));
      }
      for (int j_5 = 0; j_5 < 8; ++j_5) {
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_5 * 8)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_5 * 8)), reinterpret_cast<const unsigned*>(B_local_5 + (j_5 * 8)));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_5 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_5 * 8)), reinterpret_cast<const unsigned*>(B_local_5 + ((j_5 * 8) + 4)));
      }
    }
  }
  #pragma unroll
  for (int i_35 = 0; i_35 < 2; ++i_35) {
    ssum[i_35] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_5 = 0; rv_5 < 16; ++rv_5) {
      ssum[i_35] = (ssum[i_35] + acc_s[((((rv_5 & 7) * 4) + (i_35 * 2)) + (rv_5 >> 3))]);
    }
    ssum[i_35] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(ssum[i_35]);
  }
  float2 __30;
    float2 v__36 = *(float2*)(lsum + 0);
    float2 v__37 = *(float2*)(sc + 0);
    float2 v__38 = *(float2*)(ssum + 0);
    *(float2*)(&(__30.x)) = tl::fma2(*(float2*)(&(v__36.x)), *(float2*)(&(v__37.x)), *(float2*)(&(v__38.x)));
  *(float2*)(lsum + 0) = __30;
  #pragma unroll
  for (int i_36 = 0; i_36 < 8; ++i_36) {
    float broadcast_var_18 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_s + (i_36 * 4)) = make_float4(broadcast_var_18, broadcast_var_18, broadcast_var_18, broadcast_var_18);
  }
  tl::cp_async_wait<3>();
  __syncthreads();
  {
    half_t A_local_3[8];
    half_t B_local_6[32];
    for (int ki_6 = 0; ki_6 < 8; ++ki_6) {
      tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[(((((((ki_6 >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_6 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_6 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_3[0])));
      #pragma unroll
      for (int i_37 = 0; i_37 < 4; ++i_37) {
        tl::ptx_ldmatrix_x4((&(((half_t*)K_shared)[(((((((((ki_6 >> 2) * 4096) + (i_37 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_6 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_6 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)])), (&(B_local_6[(i_37 * 8)])));
      }
      for (int j_6 = 0; j_6 < 4; ++j_6) {
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (j_6 * 8)), reinterpret_cast<const unsigned*>(A_local_3 + 0), reinterpret_cast<const unsigned*>(B_local_6 + (j_6 * 8)));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((j_6 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_3 + 0), reinterpret_cast<const unsigned*>(B_local_6 + ((j_6 * 8) + 4)));
      }
    }
  }
  *(float2*)(m_prev + 0) = *(float2*)(m + 0);
  float broadcast_var_19 = -CUDART_INF_F;
  *(float2*)(m + 0) = make_float2(broadcast_var_19, broadcast_var_19);
  #pragma unroll
  for (int i_38 = 0; i_38 < 2; ++i_38) {
    m_clear_3[i_38] = -CUDART_INF_F;
    #pragma unroll
    for (int rv_6 = 0; rv_6 < 16; ++rv_6) {
      m_clear_3[i_38] = max(m_clear_3[i_38], acc_s[((((rv_6 & 7) * 4) + (i_38 * 2)) + (rv_6 >> 3))]);
    }
    m_clear_3[i_38] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(m_clear_3[i_38]);
    m[i_38] = max(m[i_38], m_clear_3[i_38]);
  }
  float broadcast_var_20 = 0x1.0527dbd5cafffp-3f/*1.275174e-01*/;
  float broadcast_var_21 = 0x1.0527dbd5cafffp-3f/*1.275174e-01*/;
  float2 __31;
  float2 __32;
    float2 __33;
      float2 v__39 = *(float2*)(m_prev + 0);
      float2 v__40 = make_float2(broadcast_var_20, broadcast_var_20);
      *(float2*)(&(__33.x)) = tl::mul2(*(float2*)(&(v__39.x)), *(float2*)(&(v__40.x)));
    float2 __34;
      float2 v__41 = *(float2*)(m + 0);
      float2 v__42 = make_float2(broadcast_var_21, broadcast_var_21);
      *(float2*)(&(__34.x)) = tl::mul2(*(float2*)(&(v__41.x)), *(float2*)(&(v__42.x)));
    *(float2*)(&(__32.x)) = tl::sub2(*(float2*)(&(__33.x)), *(float2*)(&(__34.x)));
  __31.x = exp2f(__32.x);
  __31.y = exp2f(__32.y);
  *(float2*)(sc + 0) = __31;
  #pragma unroll
  for (int i_39 = 0; i_39 < 32; ++i_39) {
    float2 __35;
      float2 v__43 = *(float2*)(acc_o + (i_39 * 2));
      float2 v__44 = make_float2(sc[(i_39 & 1)], sc[(i_39 & 1)]);
      *(float2*)(&(__35.x)) = tl::mul2(*(float2*)(&(v__43.x)), *(float2*)(&(v__44.x)));
    *(float2*)(acc_o + (i_39 * 2)) = __35;
  }
  #pragma unroll
  for (int i_40 = 0; i_40 < 16; ++i_40) {
    float broadcast_var_22 = 0x1.0527dbd5cafffp-3f/*1.275174e-01*/;
    float2 __36;
    float2 __37;
      float2 __38;
        float2 v__45 = *(float2*)(acc_s + (i_40 * 2));
        float2 v__46 = make_float2(broadcast_var_22, broadcast_var_22);
        *(float2*)(&(__38.x)) = tl::mul2(*(float2*)(&(v__45.x)), *(float2*)(&(v__46.x)));
      float2 v__47 = make_float2((m[(i_40 & 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/), (m[(i_40 & 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/));
      *(float2*)(&(__37.x)) = tl::sub2(*(float2*)(&(__38.x)), *(float2*)(&(v__47.x)));
    __36.x = exp2f(__37.x);
    __36.y = exp2f(__37.y);
    *(float2*)(acc_s + (i_40 * 2)) = __36;
  }
  #pragma unroll
  for (int i_41 = 0; i_41 < 8; ++i_41) {
    uint2 __39;
    float4 v__48 = *(float4*)(acc_s + (i_41 * 4));
    ((half2*)(&__39))[0] = __float22half2_rn(((float2*)(&v__48))[0]);
    ((half2*)(&__39))[1] = __float22half2_rn(((float2*)(&v__48))[1]);
    *(uint2*)(acc_s_cast + (i_41 * 4)) = __39;
  }
  tl::cp_async_wait<2>();
  __syncthreads();
  {
    half_t B_local_7[64];
    for (int ki_7 = 0; ki_7 < 4; ++ki_7) {
      #pragma unroll
      for (int i_42 = 0; i_42 < 8; ++i_42) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)V_shared)[((((((((i_42 >> 2) * 4096) + (ki_7 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_42 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_42 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)])), (&(B_local_7[(i_42 * 8)])));
      }
      for (int j_7 = 0; j_7 < 8; ++j_7) {
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_7 * 8)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_7 * 8)), reinterpret_cast<const unsigned*>(B_local_7 + (j_7 * 8)));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_7 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_7 * 8)), reinterpret_cast<const unsigned*>(B_local_7 + ((j_7 * 8) + 4)));
      }
    }
  }
  #pragma unroll
  for (int i_43 = 0; i_43 < 2; ++i_43) {
    ssum[i_43] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_7 = 0; rv_7 < 16; ++rv_7) {
      ssum[i_43] = (ssum[i_43] + acc_s[((((rv_7 & 7) * 4) + (i_43 * 2)) + (rv_7 >> 3))]);
    }
    ssum[i_43] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(ssum[i_43]);
  }
  float2 __40;
    float2 v__49 = *(float2*)(lsum + 0);
    float2 v__50 = *(float2*)(sc + 0);
    float2 v__51 = *(float2*)(ssum + 0);
    *(float2*)(&(__40.x)) = tl::fma2(*(float2*)(&(v__49.x)), *(float2*)(&(v__50.x)), *(float2*)(&(v__51.x)));
  *(float2*)(lsum + 0) = __40;
  #pragma unroll
  for (int i_44 = 0; i_44 < 8; ++i_44) {
    float broadcast_var_23 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_s + (i_44 * 4)) = make_float4(broadcast_var_23, broadcast_var_23, broadcast_var_23, broadcast_var_23);
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  {
    half_t A_local_4[8];
    half_t B_local_8[32];
    for (int ki_8 = 0; ki_8 < 8; ++ki_8) {
      tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[(((((((ki_8 >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_8 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_8 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_4[0])));
      #pragma unroll
      for (int i_45 = 0; i_45 < 4; ++i_45) {
        tl::ptx_ldmatrix_x4((&(((half_t*)K_shared)[(((((((((ki_8 >> 2) * 4096) + (i_45 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_8 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_8 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 24576)])), (&(B_local_8[(i_45 * 8)])));
      }
      for (int j_8 = 0; j_8 < 4; ++j_8) {
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (j_8 * 8)), reinterpret_cast<const unsigned*>(A_local_4 + 0), reinterpret_cast<const unsigned*>(B_local_8 + (j_8 * 8)));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((j_8 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_4 + 0), reinterpret_cast<const unsigned*>(B_local_8 + ((j_8 * 8) + 4)));
      }
    }
  }
  *(float2*)(m_prev + 0) = *(float2*)(m + 0);
  float broadcast_var_24 = -CUDART_INF_F;
  *(float2*)(m + 0) = make_float2(broadcast_var_24, broadcast_var_24);
  #pragma unroll
  for (int i_46 = 0; i_46 < 2; ++i_46) {
    m_clear_4[i_46] = -CUDART_INF_F;
    #pragma unroll
    for (int rv_8 = 0; rv_8 < 16; ++rv_8) {
      m_clear_4[i_46] = max(m_clear_4[i_46], acc_s[((((rv_8 & 7) * 4) + (i_46 * 2)) + (rv_8 >> 3))]);
    }
    m_clear_4[i_46] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(m_clear_4[i_46]);
    m[i_46] = max(m[i_46], m_clear_4[i_46]);
  }
  float broadcast_var_25 = 0x1.0527dbd5cafffp-3f/*1.275174e-01*/;
  float broadcast_var_26 = 0x1.0527dbd5cafffp-3f/*1.275174e-01*/;
  float2 __41;
  float2 __42;
    float2 __43;
      float2 v__52 = *(float2*)(m_prev + 0);
      float2 v__53 = make_float2(broadcast_var_25, broadcast_var_25);
      *(float2*)(&(__43.x)) = tl::mul2(*(float2*)(&(v__52.x)), *(float2*)(&(v__53.x)));
    float2 __44;
      float2 v__54 = *(float2*)(m + 0);
      float2 v__55 = make_float2(broadcast_var_26, broadcast_var_26);
      *(float2*)(&(__44.x)) = tl::mul2(*(float2*)(&(v__54.x)), *(float2*)(&(v__55.x)));
    *(float2*)(&(__42.x)) = tl::sub2(*(float2*)(&(__43.x)), *(float2*)(&(__44.x)));
  __41.x = exp2f(__42.x);
  __41.y = exp2f(__42.y);
  *(float2*)(sc + 0) = __41;
  #pragma unroll
  for (int i_47 = 0; i_47 < 32; ++i_47) {
    float2 __45;
      float2 v__56 = *(float2*)(acc_o + (i_47 * 2));
      float2 v__57 = make_float2(sc[(i_47 & 1)], sc[(i_47 & 1)]);
      *(float2*)(&(__45.x)) = tl::mul2(*(float2*)(&(v__56.x)), *(float2*)(&(v__57.x)));
    *(float2*)(acc_o + (i_47 * 2)) = __45;
  }
  #pragma unroll
  for (int i_48 = 0; i_48 < 16; ++i_48) {
    float broadcast_var_27 = 0x1.0527dbd5cafffp-3f/*1.275174e-01*/;
    float2 __46;
    float2 __47;
      float2 __48;
        float2 v__58 = *(float2*)(acc_s + (i_48 * 2));
        float2 v__59 = make_float2(broadcast_var_27, broadcast_var_27);
        *(float2*)(&(__48.x)) = tl::mul2(*(float2*)(&(v__58.x)), *(float2*)(&(v__59.x)));
      float2 v__60 = make_float2((m[(i_48 & 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/), (m[(i_48 & 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/));
      *(float2*)(&(__47.x)) = tl::sub2(*(float2*)(&(__48.x)), *(float2*)(&(v__60.x)));
    __46.x = exp2f(__47.x);
    __46.y = exp2f(__47.y);
    *(float2*)(acc_s + (i_48 * 2)) = __46;
  }
  #pragma unroll
  for (int i_49 = 0; i_49 < 8; ++i_49) {
    uint2 __49;
    float4 v__61 = *(float4*)(acc_s + (i_49 * 4));
    ((half2*)(&__49))[0] = __float22half2_rn(((float2*)(&v__61))[0]);
    ((half2*)(&__49))[1] = __float22half2_rn(((float2*)(&v__61))[1]);
    *(uint2*)(acc_s_cast + (i_49 * 4)) = __49;
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    half_t B_local_9[64];
    for (int ki_9 = 0; ki_9 < 4; ++ki_9) {
      #pragma unroll
      for (int i_50 = 0; i_50 < 8; ++i_50) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)V_shared)[((((((((i_50 >> 2) * 4096) + (ki_9 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_50 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_50 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 24576)])), (&(B_local_9[(i_50 * 8)])));
      }
      for (int j_9 = 0; j_9 < 8; ++j_9) {
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_9 * 8)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_9 * 8)), reinterpret_cast<const unsigned*>(B_local_9 + (j_9 * 8)));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_9 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_9 * 8)), reinterpret_cast<const unsigned*>(B_local_9 + ((j_9 * 8) + 4)));
      }
    }
  }
  #pragma unroll
  for (int i_51 = 0; i_51 < 2; ++i_51) {
    ssum[i_51] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_9 = 0; rv_9 < 16; ++rv_9) {
      ssum[i_51] = (ssum[i_51] + acc_s[((((rv_9 & 7) * 4) + (i_51 * 2)) + (rv_9 >> 3))]);
    }
    ssum[i_51] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(ssum[i_51]);
  }
  float2 __50;
    float2 v__62 = *(float2*)(lsum + 0);
    float2 v__63 = *(float2*)(sc + 0);
    float2 v__64 = *(float2*)(ssum + 0);
    *(float2*)(&(__50.x)) = tl::fma2(*(float2*)(&(v__62.x)), *(float2*)(&(v__63.x)), *(float2*)(&(v__64.x)));
  *(float2*)(lsum + 0) = __50;
  #pragma unroll
  for (int i_52 = 0; i_52 < 32; ++i_52) {
    float2 __51;
      float2 v__65 = *(float2*)(acc_o + (i_52 * 2));
      float2 v__66 = make_float2(lsum[(i_52 & 1)], lsum[(i_52 & 1)]);
      __51.x = (v__65.x/v__66.x);
      __51.y = (v__65.y/v__66.y);
    *(float2*)(acc_o + (i_52 * 2)) = __51;
  }
  __syncthreads();
  #pragma unroll
  for (int i_53 = 0; i_53 < 8; ++i_53) {
    tl::ptx_stmatrix_m8n8_x4((&(((half_t*)O_shared)[(((((((int)threadIdx.x) >> 5) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + (i_53 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), __pack_half2(((half_t)acc_o[(i_53 * 8)]), ((half_t)acc_o[((i_53 * 8) + 1)])), __pack_half2(((half_t)acc_o[((i_53 * 8) + 2)]), ((half_t)acc_o[((i_53 * 8) + 3)])), __pack_half2(((half_t)acc_o[((i_53 * 8) + 4)]), ((half_t)acc_o[((i_53 * 8) + 5)])), __pack_half2(((half_t)acc_o[((i_53 * 8) + 6)]), ((half_t)acc_o[((i_53 * 8) + 7)])));
  }
  __syncthreads();
  #pragma unroll
  for (int i_54 = 0; i_54 < 8; ++i_54) {
    *(uint4*)(Out + (((((((int)blockIdx.x) * 65536) + (i_54 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8))) = *(uint4*)(((half_t*)O_shared) + ((i_54 * 1024) + (((int)threadIdx.x) * 8)));
  }
}

