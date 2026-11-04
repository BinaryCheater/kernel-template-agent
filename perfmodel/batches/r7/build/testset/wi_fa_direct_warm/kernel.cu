#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
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
  void* V_shared = ((void*)((char*)buf_dyn_shmem + 32768));
  float acc_o[64];
  float lsum[2];
  float m[2];
  float acc_s[32];
  float m_prev[2];
  float m_clear[2];
  float sc[2];
  half_t acc_s_cast[32];
  float ssum[2];
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
  for (int k = 0; k < 32; ++k) {
    __syncthreads();
    #pragma unroll
    for (int i_2 = 0; i_2 < 8; ++i_2) {
      *(uint4*)(((half_t*)K_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_2 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(K + (((((k * 65536) + (i_2 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 8; ++i_3) {
      float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_s + (i_3 * 4)) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
    }
    {
      half_t A_local[8];
      half_t B_local[32];
      __syncthreads();
      for (int ki = 0; ki < 8; ++ki) {
        tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[(((((((ki >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local[0])));
        #pragma unroll
        for (int i_4 = 0; i_4 < 4; ++i_4) {
          tl::ptx_ldmatrix_x4((&(((half_t*)K_shared)[((((((((ki >> 2) * 4096) + (i_4 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[(i_4 * 8)])));
        }
        for (int j = 0; j < 4; ++j) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (j * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
        }
      }
    }
    *(float2*)(m_prev + 0) = *(float2*)(m + 0);
    float broadcast_var_4 = -CUDART_INF_F;
    *(float2*)(m + 0) = make_float2(broadcast_var_4, broadcast_var_4);
    #pragma unroll
    for (int i_5 = 0; i_5 < 2; ++i_5) {
      m_clear[i_5] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 16; ++rv) {
        m_clear[i_5] = max(m_clear[i_5], acc_s[((((rv & 7) * 4) + (i_5 * 2)) + (rv >> 3))]);
      }
      m_clear[i_5] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(m_clear[i_5]);
      m[i_5] = max(m[i_5], m_clear[i_5]);
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
    for (int i_6 = 0; i_6 < 32; ++i_6) {
      float2 __5;
        float2 v__4 = *(float2*)(acc_o + (i_6 * 2));
        float2 v__5 = make_float2(sc[(i_6 & 1)], sc[(i_6 & 1)]);
        *(float2*)(&(__5.x)) = tl::mul2(*(float2*)(&(v__4.x)), *(float2*)(&(v__5.x)));
      *(float2*)(acc_o + (i_6 * 2)) = __5;
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 16; ++i_7) {
      float broadcast_var_7 = 0x1.0527dbd5cafffp-3f/*1.275174e-01*/;
      float2 __6;
      float2 __7;
        float2 __8;
          float2 v__6 = *(float2*)(acc_s + (i_7 * 2));
          float2 v__7 = make_float2(broadcast_var_7, broadcast_var_7);
          *(float2*)(&(__8.x)) = tl::mul2(*(float2*)(&(v__6.x)), *(float2*)(&(v__7.x)));
        float2 v__8 = make_float2((m[(i_7 & 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/), (m[(i_7 & 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/));
        *(float2*)(&(__7.x)) = tl::sub2(*(float2*)(&(__8.x)), *(float2*)(&(v__8.x)));
      __6.x = exp2f(__7.x);
      __6.y = exp2f(__7.y);
      *(float2*)(acc_s + (i_7 * 2)) = __6;
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 8; ++i_8) {
      uint2 __9;
      float4 v__9 = *(float4*)(acc_s + (i_8 * 4));
      ((half2*)(&__9))[0] = __float22half2_rn(((float2*)(&v__9))[0]);
      ((half2*)(&__9))[1] = __float22half2_rn(((float2*)(&v__9))[1]);
      *(uint2*)(acc_s_cast + (i_8 * 4)) = __9;
    }
    __syncthreads();
    #pragma unroll
    for (int i_9 = 0; i_9 < 8; ++i_9) {
      *(uint4*)(((half_t*)V_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_9 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(V + (((((k * 65536) + (i_9 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
    }
    {
      half_t B_local_1[64];
      __syncthreads();
      for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
        #pragma unroll
        for (int i_10 = 0; i_10 < 8; ++i_10) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)V_shared)[(((((((i_10 >> 2) * 4096) + (ki_1 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_10 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_10 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_1[(i_10 * 8)])));
        }
        for (int j_1 = 0; j_1 < 8; ++j_1) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_1 * 8)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_1 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_1 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
        }
      }
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 2; ++i_11) {
      ssum[i_11] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
        ssum[i_11] = (ssum[i_11] + acc_s[((((rv_1 & 7) * 4) + (i_11 * 2)) + (rv_1 >> 3))]);
      }
      ssum[i_11] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(ssum[i_11]);
    }
    float2 __10;
      float2 v__10 = *(float2*)(lsum + 0);
      float2 v__11 = *(float2*)(sc + 0);
      float2 v__12 = *(float2*)(ssum + 0);
      *(float2*)(&(__10.x)) = tl::fma2(*(float2*)(&(v__10.x)), *(float2*)(&(v__11.x)), *(float2*)(&(v__12.x)));
    *(float2*)(lsum + 0) = __10;
  }
  #pragma unroll
  for (int i_12 = 0; i_12 < 32; ++i_12) {
    float2 __11;
      float2 v__13 = *(float2*)(acc_o + (i_12 * 2));
      float2 v__14 = make_float2(lsum[(i_12 & 1)], lsum[(i_12 & 1)]);
      __11.x = (v__13.x/v__14.x);
      __11.y = (v__13.y/v__14.y);
    *(float2*)(acc_o + (i_12 * 2)) = __11;
  }
  __syncthreads();
  #pragma unroll
  for (int i_13 = 0; i_13 < 8; ++i_13) {
    tl::ptx_stmatrix_m8n8_x4((&(((half_t*)O_shared)[(((((((int)threadIdx.x) >> 5) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + (i_13 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), __pack_half2(((half_t)acc_o[(i_13 * 8)]), ((half_t)acc_o[((i_13 * 8) + 1)])), __pack_half2(((half_t)acc_o[((i_13 * 8) + 2)]), ((half_t)acc_o[((i_13 * 8) + 3)])), __pack_half2(((half_t)acc_o[((i_13 * 8) + 4)]), ((half_t)acc_o[((i_13 * 8) + 5)])), __pack_half2(((half_t)acc_o[((i_13 * 8) + 6)]), ((half_t)acc_o[((i_13 * 8) + 7)])));
  }
  __syncthreads();
  #pragma unroll
  for (int i_14 = 0; i_14 < 8; ++i_14) {
    *(uint4*)(Out + (((((((int)blockIdx.x) * 65536) + (i_14 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8))) = *(uint4*)(((half_t*)O_shared) + ((i_14 * 1024) + (((int)threadIdx.x) * 8)));
  }
}

