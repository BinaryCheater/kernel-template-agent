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
extern "C" __global__ void __launch_bounds__(64, 1) fa_kernel(const half_t* __restrict__ K, half_t* __restrict__ Out, const half_t* __restrict__ Q, const half_t* __restrict__ V) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* O_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* Q_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* K_shared = ((void*)((char*)buf_dyn_shmem + 14336));
  void* V_shared = ((void*)((char*)buf_dyn_shmem + 43008));
  float acc_o[112];
  float lsum[4];
  float m[4];
  float acc_s[64];
  float m_prev[4];
  float sc[4];
  half_t acc_s_cast[64];
  float ssum[4];
  float m_clear[4];
  float m_clear_1[4];
  float m_clear_2[4];
  #pragma unroll
  for (int i = 0; i < 14; ++i) {
    half_t broadcast_var = half_t(0x0p+0f/*0.000000e+00*/);
    uint4 condval;
    if ((((((int)blockIdx.x) * 4) + (((i * 32) + (((int)threadIdx.x) >> 1)) / 112)) < 169)) {
      condval = *(uint4*)(Q + ((((((int)blockIdx.x) * 93184) + ((((i * 32) + (((int)threadIdx.x) >> 1)) / 7) * 1456)) + (((int)blockIdx.y) * 112)) + ((((i * 64) + ((int)threadIdx.x)) % 14) * 8)));
    } else {
      condval = make_uint4(__pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var));
    }
    *(uint4*)(((half_t*)Q_shared) + ((((((i * 32) + (((int)threadIdx.x) >> 1)) % 7) * 1024) + ((((i * 32) + (((int)threadIdx.x) >> 1)) / 7) * 16)) + (((((((i * 8) + (((int)threadIdx.x) >> 3)) % 14) / 7) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval;
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 28; ++i_1) {
    float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_o + (i_1 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
  }
  float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
  *(float4*)(lsum + 0) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
  float broadcast_var_3 = -CUDART_INF_F;
  *(float4*)(m + 0) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
  __syncthreads();
  #pragma unroll
  for (int i_2 = 0; i_2 < 14; ++i_2) {
    tl::cp_async_gs<16>((&(((half_t*)K_shared)[((((((i_2 * 32) + (((int)threadIdx.x) >> 1)) % 7) * 1024) + ((((i_2 * 32) + (((int)threadIdx.x) >> 1)) / 7) * 16)) + (((((((i_2 * 8) + (((int)threadIdx.x) >> 3)) % 14) / 7) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(K[((((((i_2 * 32) + (((int)threadIdx.x) >> 1)) / 7) * 1456) + (((int)blockIdx.y) * 112)) + ((((i_2 * 64) + ((int)threadIdx.x)) % 14) * 8))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_3 = 0; i_3 < 14; ++i_3) {
    tl::cp_async_gs<16>((&(((half_t*)V_shared)[((((((i_3 * 32) + (((int)threadIdx.x) >> 1)) % 7) * 1024) + ((((i_3 * 32) + (((int)threadIdx.x) >> 1)) / 7) * 16)) + (((((((i_3 * 8) + (((int)threadIdx.x) >> 3)) % 14) / 7) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(V[((((((i_3 * 32) + (((int)threadIdx.x) >> 1)) / 7) * 1456) + (((int)blockIdx.y) * 112)) + ((((i_3 * 64) + ((int)threadIdx.x)) % 14) * 8))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_4 = 0; i_4 < 14; ++i_4) {
    tl::cp_async_gs<16>((&(((half_t*)K_shared)[(((((((i_4 * 32) + (((int)threadIdx.x) >> 1)) % 7) * 1024) + ((((i_4 * 32) + (((int)threadIdx.x) >> 1)) / 7) * 16)) + (((((((i_4 * 8) + (((int)threadIdx.x) >> 3)) % 14) / 7) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 7168)])), (&(K[(((((((i_4 * 32) + (((int)threadIdx.x) >> 1)) / 7) * 1456) + (((int)blockIdx.y) * 112)) + ((((i_4 * 64) + ((int)threadIdx.x)) % 14) * 8)) + 93184)])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_5 = 0; i_5 < 14; ++i_5) {
    tl::cp_async_gs<16>((&(((half_t*)V_shared)[(((((((i_5 * 32) + (((int)threadIdx.x) >> 1)) % 7) * 1024) + ((((i_5 * 32) + (((int)threadIdx.x) >> 1)) / 7) * 16)) + (((((((i_5 * 8) + (((int)threadIdx.x) >> 3)) % 14) / 7) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 7168)])), (&(V[(((((((i_5 * 32) + (((int)threadIdx.x) >> 1)) / 7) * 1456) + (((int)blockIdx.y) * 112)) + ((((i_5 * 64) + ((int)threadIdx.x)) % 14) * 8)) + 93184)])));
  }
  tl::cp_async_commit();
  for (int k = 0; k < 41; ++k) {
    #pragma unroll
    for (int i_6 = 0; i_6 < 16; ++i_6) {
      float broadcast_var_4 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_s + (i_6 * 4)) = make_float4(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4);
    }
    tl::cp_async_wait<3>();
    __syncthreads();
    {
      half_t A_local[16];
      half_t B_local[32];
      for (int ki = 0; ki < 7; ++ki) {
        #pragma unroll
        for (int i_7 = 0; i_7 < 2; ++i_7) {
          tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[(((((ki * 1024) + ((((int)threadIdx.x) >> 5) * 512)) + (i_7 * 256)) + ((((int)threadIdx.x) & 15) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(A_local[(i_7 * 8)])));
        }
        #pragma unroll
        for (int i_8 = 0; i_8 < 4; ++i_8) {
          tl::ptx_ldmatrix_x4((&(((half_t*)K_shared)[(((((((k & 1) * 7168) + (ki * 1024)) + (i_8 * 256)) + (((((int)threadIdx.x) & 31) >> 4) * 128)) + ((((int)threadIdx.x) & 7) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(B_local[(i_8 * 8)])));
        }
        for (int i_9 = 0; i_9 < 2; ++i_9) {
          for (int j = 0; j < 4; ++j) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((i_9 * 32) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_9 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (((i_9 * 32) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_9 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_10 = 0; i_10 < 14; ++i_10) {
      tl::cp_async_gs_conditional<16>((&(((half_t*)K_shared)[(((((k & 1) * 7168) + ((((i_10 * 32) + (((int)threadIdx.x) >> 1)) % 7) * 1024)) + ((((i_10 * 32) + (((int)threadIdx.x) >> 1)) / 7) * 16)) + (((((((i_10 * 8) + (((int)threadIdx.x) >> 3)) % 14) / 7) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(K[(((((k * 93184) + ((((i_10 * 32) + (((int)threadIdx.x) >> 1)) / 7) * 1456)) + (((int)blockIdx.y) * 112)) + ((((i_10 * 64) + ((int)threadIdx.x)) % 14) * 8)) + 186368)])), ((((k * 4) + (((i_10 * 32) + (((int)threadIdx.x) >> 1)) / 112)) < 161) && (((k * 4) + (((i_10 * 32) + (((int)threadIdx.x) >> 1)) / 112)) < 161)));
    }
    tl::cp_async_commit();
    *(float4*)(m_prev + 0) = *(float4*)(m + 0);
    float broadcast_var_5 = -CUDART_INF_F;
    *(float4*)(m + 0) = make_float4(broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5);
    #pragma unroll
    for (int i_11 = 0; i_11 < 4; ++i_11) {
      m_clear[i_11] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 16; ++rv) {
        m_clear[i_11] = max(m_clear[i_11], acc_s[(((((i_11 >> 1) * 32) + ((rv & 7) * 4)) + ((i_11 & 1) * 2)) + (rv >> 3))]);
      }
      m_clear[i_11] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<64>>::run(m_clear[i_11]);
      m[i_11] = max(m[i_11], m_clear[i_11]);
    }
    float broadcast_var_6 = 0x1.172feb597d322p-3f/*1.363219e-01*/;
    float broadcast_var_7 = 0x1.172feb597d322p-3f/*1.363219e-01*/;
    float4 __1;
    float4 __2;
      float4 __3;
        float4 v_ = *(float4*)(m_prev + 0);
        float4 v__1 = make_float4(broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6);
        *(float2*)(&(__3.x)) = tl::mul2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)));
        *(float2*)(&(__3.z)) = tl::mul2(*(float2*)(&(v_.z)), *(float2*)(&(v__1.z)));
      float4 __4;
        float4 v__2 = *(float4*)(m + 0);
        float4 v__3 = make_float4(broadcast_var_7, broadcast_var_7, broadcast_var_7, broadcast_var_7);
        *(float2*)(&(__4.x)) = tl::mul2(*(float2*)(&(v__2.x)), *(float2*)(&(v__3.x)));
        *(float2*)(&(__4.z)) = tl::mul2(*(float2*)(&(v__2.z)), *(float2*)(&(v__3.z)));
      *(float2*)(&(__2.x)) = tl::sub2(*(float2*)(&(__3.x)), *(float2*)(&(__4.x)));
      *(float2*)(&(__2.z)) = tl::sub2(*(float2*)(&(__3.z)), *(float2*)(&(__4.z)));
    __1.x = exp2f(__2.x);
    __1.y = exp2f(__2.y);
    __1.z = exp2f(__2.z);
    __1.w = exp2f(__2.w);
    *(float4*)(sc + 0) = __1;
    #pragma unroll
    for (int i_12 = 0; i_12 < 56; ++i_12) {
      float2 __5;
        float2 v__4 = *(float2*)(acc_o + (i_12 * 2));
        float2 v__5 = make_float2(sc[(((i_12 / 28) * 2) + (i_12 & 1))], sc[(((i_12 / 28) * 2) + (i_12 & 1))]);
        *(float2*)(&(__5.x)) = tl::mul2(*(float2*)(&(v__4.x)), *(float2*)(&(v__5.x)));
      *(float2*)(acc_o + (i_12 * 2)) = __5;
    }
    #pragma unroll
    for (int i_13 = 0; i_13 < 32; ++i_13) {
      float broadcast_var_8 = 0x1.172feb597d322p-3f/*1.363219e-01*/;
      float2 __6;
      float2 __7;
        float2 __8;
          float2 v__6 = *(float2*)(acc_s + (i_13 * 2));
          float2 v__7 = make_float2(broadcast_var_8, broadcast_var_8);
          *(float2*)(&(__8.x)) = tl::mul2(*(float2*)(&(v__6.x)), *(float2*)(&(v__7.x)));
        float2 v__8 = make_float2((m[(((i_13 >> 4) * 2) + (i_13 & 1))] * 0x1.172feb597d322p-3f/*1.363219e-01*/), (m[(((i_13 >> 4) * 2) + (i_13 & 1))] * 0x1.172feb597d322p-3f/*1.363219e-01*/));
        *(float2*)(&(__7.x)) = tl::sub2(*(float2*)(&(__8.x)), *(float2*)(&(v__8.x)));
      __6.x = exp2f(__7.x);
      __6.y = exp2f(__7.y);
      *(float2*)(acc_s + (i_13 * 2)) = __6;
    }
    #pragma unroll
    for (int i_14 = 0; i_14 < 16; ++i_14) {
      uint2 __9;
      float4 v__9 = *(float4*)(acc_s + (((((i_14 & 3) >> 1) * 32) + ((i_14 >> 2) * 8)) + ((i_14 & 1) * 4)));
      ((half2*)(&__9))[0] = __float22half2_rn(((float2*)(&v__9))[0]);
      ((half2*)(&__9))[1] = __float22half2_rn(((float2*)(&v__9))[1]);
      *(uint2*)(acc_s_cast + (i_14 * 4)) = __9;
    }
    tl::cp_async_wait<3>();
    __syncthreads();
    {
      half_t B_local_1[56];
      for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
        #pragma unroll
        for (int i_15 = 0; i_15 < 7; ++i_15) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)V_shared)[((((((k & 1) * 7168) + (i_15 * 1024)) + (ki_1 * 256)) + ((((int)threadIdx.x) & 15) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(B_local_1[(i_15 * 8)])));
        }
        for (int i_16 = 0; i_16 < 2; ++i_16) {
          for (int j_1 = 0; j_1 < 7; ++j_1) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((i_16 * 56) + (j_1 * 8))), reinterpret_cast<const unsigned*>(acc_s_cast + ((ki_1 * 16) + (i_16 * 8))), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (((i_16 * 56) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(acc_s_cast + ((ki_1 * 16) + (i_16 * 8))), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
          }
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_17 = 0; i_17 < 14; ++i_17) {
      tl::cp_async_gs_conditional<16>((&(((half_t*)V_shared)[(((((k & 1) * 7168) + ((((i_17 * 32) + (((int)threadIdx.x) >> 1)) % 7) * 1024)) + ((((i_17 * 32) + (((int)threadIdx.x) >> 1)) / 7) * 16)) + (((((((i_17 * 8) + (((int)threadIdx.x) >> 3)) % 14) / 7) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(V[(((((k * 93184) + ((((i_17 * 32) + (((int)threadIdx.x) >> 1)) / 7) * 1456)) + (((int)blockIdx.y) * 112)) + ((((i_17 * 64) + ((int)threadIdx.x)) % 14) * 8)) + 186368)])), ((((k * 4) + (((i_17 * 32) + (((int)threadIdx.x) >> 1)) / 112)) < 161) && (((k * 4) + (((i_17 * 32) + (((int)threadIdx.x) >> 1)) / 112)) < 161)));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_18 = 0; i_18 < 4; ++i_18) {
      ssum[i_18] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
        ssum[i_18] = (ssum[i_18] + acc_s[(((((i_18 >> 1) * 32) + ((rv_1 & 7) * 4)) + ((i_18 & 1) * 2)) + (rv_1 >> 3))]);
      }
      ssum[i_18] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<64>>::run(ssum[i_18]);
    }
    float4 __10;
      float4 v__10 = *(float4*)(lsum + 0);
      float4 v__11 = *(float4*)(sc + 0);
      float4 v__12 = *(float4*)(ssum + 0);
      *(float2*)(&(__10.x)) = tl::fma2(*(float2*)(&(v__10.x)), *(float2*)(&(v__11.x)), *(float2*)(&(v__12.x)));
      *(float2*)(&(__10.z)) = tl::fma2(*(float2*)(&(v__10.z)), *(float2*)(&(v__11.z)), *(float2*)(&(v__12.z)));
    *(float4*)(lsum + 0) = __10;
  }
  #pragma unroll
  for (int i_19 = 0; i_19 < 16; ++i_19) {
    float broadcast_var_9 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_s + (i_19 * 4)) = make_float4(broadcast_var_9, broadcast_var_9, broadcast_var_9, broadcast_var_9);
  }
  tl::cp_async_wait<3>();
  __syncthreads();
  {
    half_t A_local_1[16];
    half_t B_local_2[32];
    for (int ki_2 = 0; ki_2 < 7; ++ki_2) {
      #pragma unroll
      for (int i_20 = 0; i_20 < 2; ++i_20) {
        tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[(((((ki_2 * 1024) + ((((int)threadIdx.x) >> 5) * 512)) + (i_20 * 256)) + ((((int)threadIdx.x) & 15) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(A_local_1[(i_20 * 8)])));
      }
      #pragma unroll
      for (int i_21 = 0; i_21 < 4; ++i_21) {
        tl::ptx_ldmatrix_x4((&(((half_t*)K_shared)[((((((ki_2 * 1024) + (i_21 * 256)) + (((((int)threadIdx.x) & 31) >> 4) * 128)) + ((((int)threadIdx.x) & 7) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 7168)])), (&(B_local_2[(i_21 * 8)])));
      }
      for (int i_22 = 0; i_22 < 2; ++i_22) {
        for (int j_2 = 0; j_2 < 4; ++j_2) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((i_22 * 32) + (j_2 * 8))), reinterpret_cast<const unsigned*>(A_local_1 + (i_22 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + (j_2 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (((i_22 * 32) + (j_2 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + (i_22 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + ((j_2 * 8) + 4)));
        }
      }
    }
  }
  *(float4*)(m_prev + 0) = *(float4*)(m + 0);
  float broadcast_var_10 = -CUDART_INF_F;
  *(float4*)(m + 0) = make_float4(broadcast_var_10, broadcast_var_10, broadcast_var_10, broadcast_var_10);
  #pragma unroll
  for (int i_23 = 0; i_23 < 4; ++i_23) {
    m_clear_1[i_23] = -CUDART_INF_F;
    #pragma unroll
    for (int rv_2 = 0; rv_2 < 16; ++rv_2) {
      m_clear_1[i_23] = max(m_clear_1[i_23], acc_s[(((((i_23 >> 1) * 32) + ((rv_2 & 7) * 4)) + ((i_23 & 1) * 2)) + (rv_2 >> 3))]);
    }
    m_clear_1[i_23] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<64>>::run(m_clear_1[i_23]);
    m[i_23] = max(m[i_23], m_clear_1[i_23]);
  }
  float broadcast_var_11 = 0x1.172feb597d322p-3f/*1.363219e-01*/;
  float broadcast_var_12 = 0x1.172feb597d322p-3f/*1.363219e-01*/;
  float4 __11;
  float4 __12;
    float4 __13;
      float4 v__13 = *(float4*)(m_prev + 0);
      float4 v__14 = make_float4(broadcast_var_11, broadcast_var_11, broadcast_var_11, broadcast_var_11);
      *(float2*)(&(__13.x)) = tl::mul2(*(float2*)(&(v__13.x)), *(float2*)(&(v__14.x)));
      *(float2*)(&(__13.z)) = tl::mul2(*(float2*)(&(v__13.z)), *(float2*)(&(v__14.z)));
    float4 __14;
      float4 v__15 = *(float4*)(m + 0);
      float4 v__16 = make_float4(broadcast_var_12, broadcast_var_12, broadcast_var_12, broadcast_var_12);
      *(float2*)(&(__14.x)) = tl::mul2(*(float2*)(&(v__15.x)), *(float2*)(&(v__16.x)));
      *(float2*)(&(__14.z)) = tl::mul2(*(float2*)(&(v__15.z)), *(float2*)(&(v__16.z)));
    *(float2*)(&(__12.x)) = tl::sub2(*(float2*)(&(__13.x)), *(float2*)(&(__14.x)));
    *(float2*)(&(__12.z)) = tl::sub2(*(float2*)(&(__13.z)), *(float2*)(&(__14.z)));
  __11.x = exp2f(__12.x);
  __11.y = exp2f(__12.y);
  __11.z = exp2f(__12.z);
  __11.w = exp2f(__12.w);
  *(float4*)(sc + 0) = __11;
  #pragma unroll
  for (int i_24 = 0; i_24 < 56; ++i_24) {
    float2 __15;
      float2 v__17 = *(float2*)(acc_o + (i_24 * 2));
      float2 v__18 = make_float2(sc[(((i_24 / 28) * 2) + (i_24 & 1))], sc[(((i_24 / 28) * 2) + (i_24 & 1))]);
      *(float2*)(&(__15.x)) = tl::mul2(*(float2*)(&(v__17.x)), *(float2*)(&(v__18.x)));
    *(float2*)(acc_o + (i_24 * 2)) = __15;
  }
  #pragma unroll
  for (int i_25 = 0; i_25 < 32; ++i_25) {
    float broadcast_var_13 = 0x1.172feb597d322p-3f/*1.363219e-01*/;
    float2 __16;
    float2 __17;
      float2 __18;
        float2 v__19 = *(float2*)(acc_s + (i_25 * 2));
        float2 v__20 = make_float2(broadcast_var_13, broadcast_var_13);
        *(float2*)(&(__18.x)) = tl::mul2(*(float2*)(&(v__19.x)), *(float2*)(&(v__20.x)));
      float2 v__21 = make_float2((m[(((i_25 >> 4) * 2) + (i_25 & 1))] * 0x1.172feb597d322p-3f/*1.363219e-01*/), (m[(((i_25 >> 4) * 2) + (i_25 & 1))] * 0x1.172feb597d322p-3f/*1.363219e-01*/));
      *(float2*)(&(__17.x)) = tl::sub2(*(float2*)(&(__18.x)), *(float2*)(&(v__21.x)));
    __16.x = exp2f(__17.x);
    __16.y = exp2f(__17.y);
    *(float2*)(acc_s + (i_25 * 2)) = __16;
  }
  #pragma unroll
  for (int i_26 = 0; i_26 < 16; ++i_26) {
    uint2 __19;
    float4 v__22 = *(float4*)(acc_s + (((((i_26 & 3) >> 1) * 32) + ((i_26 >> 2) * 8)) + ((i_26 & 1) * 4)));
    ((half2*)(&__19))[0] = __float22half2_rn(((float2*)(&v__22))[0]);
    ((half2*)(&__19))[1] = __float22half2_rn(((float2*)(&v__22))[1]);
    *(uint2*)(acc_s_cast + (i_26 * 4)) = __19;
  }
  tl::cp_async_wait<2>();
  __syncthreads();
  {
    half_t B_local_3[56];
    for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
      #pragma unroll
      for (int i_27 = 0; i_27 < 7; ++i_27) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)V_shared)[(((((i_27 * 1024) + (ki_3 * 256)) + ((((int)threadIdx.x) & 15) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 7168)])), (&(B_local_3[(i_27 * 8)])));
      }
      for (int i_28 = 0; i_28 < 2; ++i_28) {
        for (int j_3 = 0; j_3 < 7; ++j_3) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((i_28 * 56) + (j_3 * 8))), reinterpret_cast<const unsigned*>(acc_s_cast + ((ki_3 * 16) + (i_28 * 8))), reinterpret_cast<const unsigned*>(B_local_3 + (j_3 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (((i_28 * 56) + (j_3 * 8)) + 4)), reinterpret_cast<const unsigned*>(acc_s_cast + ((ki_3 * 16) + (i_28 * 8))), reinterpret_cast<const unsigned*>(B_local_3 + ((j_3 * 8) + 4)));
        }
      }
    }
  }
  #pragma unroll
  for (int i_29 = 0; i_29 < 4; ++i_29) {
    ssum[i_29] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_3 = 0; rv_3 < 16; ++rv_3) {
      ssum[i_29] = (ssum[i_29] + acc_s[(((((i_29 >> 1) * 32) + ((rv_3 & 7) * 4)) + ((i_29 & 1) * 2)) + (rv_3 >> 3))]);
    }
    ssum[i_29] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<64>>::run(ssum[i_29]);
  }
  float4 __20;
    float4 v__23 = *(float4*)(lsum + 0);
    float4 v__24 = *(float4*)(sc + 0);
    float4 v__25 = *(float4*)(ssum + 0);
    *(float2*)(&(__20.x)) = tl::fma2(*(float2*)(&(v__23.x)), *(float2*)(&(v__24.x)), *(float2*)(&(v__25.x)));
    *(float2*)(&(__20.z)) = tl::fma2(*(float2*)(&(v__23.z)), *(float2*)(&(v__24.z)), *(float2*)(&(v__25.z)));
  *(float4*)(lsum + 0) = __20;
  #pragma unroll
  for (int i_30 = 0; i_30 < 16; ++i_30) {
    float broadcast_var_14 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_s + (i_30 * 4)) = make_float4(broadcast_var_14, broadcast_var_14, broadcast_var_14, broadcast_var_14);
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  {
    half_t A_local_2[16];
    half_t B_local_4[32];
    for (int ki_4 = 0; ki_4 < 7; ++ki_4) {
      #pragma unroll
      for (int i_31 = 0; i_31 < 2; ++i_31) {
        tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[(((((ki_4 * 1024) + ((((int)threadIdx.x) >> 5) * 512)) + (i_31 * 256)) + ((((int)threadIdx.x) & 15) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(A_local_2[(i_31 * 8)])));
      }
      #pragma unroll
      for (int i_32 = 0; i_32 < 4; ++i_32) {
        tl::ptx_ldmatrix_x4((&(((half_t*)K_shared)[(((((ki_4 * 1024) + (i_32 * 256)) + (((((int)threadIdx.x) & 31) >> 4) * 128)) + ((((int)threadIdx.x) & 7) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(B_local_4[(i_32 * 8)])));
      }
      for (int i_33 = 0; i_33 < 2; ++i_33) {
        for (int j_4 = 0; j_4 < 4; ++j_4) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((i_33 * 32) + (j_4 * 8))), reinterpret_cast<const unsigned*>(A_local_2 + (i_33 * 8)), reinterpret_cast<const unsigned*>(B_local_4 + (j_4 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (((i_33 * 32) + (j_4 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + (i_33 * 8)), reinterpret_cast<const unsigned*>(B_local_4 + ((j_4 * 8) + 4)));
        }
      }
    }
  }
  *(float4*)(m_prev + 0) = *(float4*)(m + 0);
  float broadcast_var_15 = -CUDART_INF_F;
  *(float4*)(m + 0) = make_float4(broadcast_var_15, broadcast_var_15, broadcast_var_15, broadcast_var_15);
  #pragma unroll
  for (int i_34 = 0; i_34 < 4; ++i_34) {
    m_clear_2[i_34] = -CUDART_INF_F;
    #pragma unroll
    for (int rv_4 = 0; rv_4 < 16; ++rv_4) {
      m_clear_2[i_34] = max(m_clear_2[i_34], acc_s[(((((i_34 >> 1) * 32) + ((rv_4 & 7) * 4)) + ((i_34 & 1) * 2)) + (rv_4 >> 3))]);
    }
    m_clear_2[i_34] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<64>>::run(m_clear_2[i_34]);
    m[i_34] = max(m[i_34], m_clear_2[i_34]);
  }
  float broadcast_var_16 = 0x1.172feb597d322p-3f/*1.363219e-01*/;
  float broadcast_var_17 = 0x1.172feb597d322p-3f/*1.363219e-01*/;
  float4 __21;
  float4 __22;
    float4 __23;
      float4 v__26 = *(float4*)(m_prev + 0);
      float4 v__27 = make_float4(broadcast_var_16, broadcast_var_16, broadcast_var_16, broadcast_var_16);
      *(float2*)(&(__23.x)) = tl::mul2(*(float2*)(&(v__26.x)), *(float2*)(&(v__27.x)));
      *(float2*)(&(__23.z)) = tl::mul2(*(float2*)(&(v__26.z)), *(float2*)(&(v__27.z)));
    float4 __24;
      float4 v__28 = *(float4*)(m + 0);
      float4 v__29 = make_float4(broadcast_var_17, broadcast_var_17, broadcast_var_17, broadcast_var_17);
      *(float2*)(&(__24.x)) = tl::mul2(*(float2*)(&(v__28.x)), *(float2*)(&(v__29.x)));
      *(float2*)(&(__24.z)) = tl::mul2(*(float2*)(&(v__28.z)), *(float2*)(&(v__29.z)));
    *(float2*)(&(__22.x)) = tl::sub2(*(float2*)(&(__23.x)), *(float2*)(&(__24.x)));
    *(float2*)(&(__22.z)) = tl::sub2(*(float2*)(&(__23.z)), *(float2*)(&(__24.z)));
  __21.x = exp2f(__22.x);
  __21.y = exp2f(__22.y);
  __21.z = exp2f(__22.z);
  __21.w = exp2f(__22.w);
  *(float4*)(sc + 0) = __21;
  #pragma unroll
  for (int i_35 = 0; i_35 < 56; ++i_35) {
    float2 __25;
      float2 v__30 = *(float2*)(acc_o + (i_35 * 2));
      float2 v__31 = make_float2(sc[(((i_35 / 28) * 2) + (i_35 & 1))], sc[(((i_35 / 28) * 2) + (i_35 & 1))]);
      *(float2*)(&(__25.x)) = tl::mul2(*(float2*)(&(v__30.x)), *(float2*)(&(v__31.x)));
    *(float2*)(acc_o + (i_35 * 2)) = __25;
  }
  #pragma unroll
  for (int i_36 = 0; i_36 < 32; ++i_36) {
    float broadcast_var_18 = 0x1.172feb597d322p-3f/*1.363219e-01*/;
    float2 __26;
    float2 __27;
      float2 __28;
        float2 v__32 = *(float2*)(acc_s + (i_36 * 2));
        float2 v__33 = make_float2(broadcast_var_18, broadcast_var_18);
        *(float2*)(&(__28.x)) = tl::mul2(*(float2*)(&(v__32.x)), *(float2*)(&(v__33.x)));
      float2 v__34 = make_float2((m[(((i_36 >> 4) * 2) + (i_36 & 1))] * 0x1.172feb597d322p-3f/*1.363219e-01*/), (m[(((i_36 >> 4) * 2) + (i_36 & 1))] * 0x1.172feb597d322p-3f/*1.363219e-01*/));
      *(float2*)(&(__27.x)) = tl::sub2(*(float2*)(&(__28.x)), *(float2*)(&(v__34.x)));
    __26.x = exp2f(__27.x);
    __26.y = exp2f(__27.y);
    *(float2*)(acc_s + (i_36 * 2)) = __26;
  }
  #pragma unroll
  for (int i_37 = 0; i_37 < 16; ++i_37) {
    uint2 __29;
    float4 v__35 = *(float4*)(acc_s + (((((i_37 & 3) >> 1) * 32) + ((i_37 >> 2) * 8)) + ((i_37 & 1) * 4)));
    ((half2*)(&__29))[0] = __float22half2_rn(((float2*)(&v__35))[0]);
    ((half2*)(&__29))[1] = __float22half2_rn(((float2*)(&v__35))[1]);
    *(uint2*)(acc_s_cast + (i_37 * 4)) = __29;
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    half_t B_local_5[56];
    for (int ki_5 = 0; ki_5 < 4; ++ki_5) {
      #pragma unroll
      for (int i_38 = 0; i_38 < 7; ++i_38) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)V_shared)[((((i_38 * 1024) + (ki_5 * 256)) + ((((int)threadIdx.x) & 15) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(B_local_5[(i_38 * 8)])));
      }
      for (int i_39 = 0; i_39 < 2; ++i_39) {
        for (int j_5 = 0; j_5 < 7; ++j_5) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((i_39 * 56) + (j_5 * 8))), reinterpret_cast<const unsigned*>(acc_s_cast + ((ki_5 * 16) + (i_39 * 8))), reinterpret_cast<const unsigned*>(B_local_5 + (j_5 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (((i_39 * 56) + (j_5 * 8)) + 4)), reinterpret_cast<const unsigned*>(acc_s_cast + ((ki_5 * 16) + (i_39 * 8))), reinterpret_cast<const unsigned*>(B_local_5 + ((j_5 * 8) + 4)));
        }
      }
    }
  }
  #pragma unroll
  for (int i_40 = 0; i_40 < 4; ++i_40) {
    ssum[i_40] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_5 = 0; rv_5 < 16; ++rv_5) {
      ssum[i_40] = (ssum[i_40] + acc_s[(((((i_40 >> 1) * 32) + ((rv_5 & 7) * 4)) + ((i_40 & 1) * 2)) + (rv_5 >> 3))]);
    }
    ssum[i_40] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<64>>::run(ssum[i_40]);
  }
  float4 __30;
    float4 v__36 = *(float4*)(lsum + 0);
    float4 v__37 = *(float4*)(sc + 0);
    float4 v__38 = *(float4*)(ssum + 0);
    *(float2*)(&(__30.x)) = tl::fma2(*(float2*)(&(v__36.x)), *(float2*)(&(v__37.x)), *(float2*)(&(v__38.x)));
    *(float2*)(&(__30.z)) = tl::fma2(*(float2*)(&(v__36.z)), *(float2*)(&(v__37.z)), *(float2*)(&(v__38.z)));
  *(float4*)(lsum + 0) = __30;
  #pragma unroll
  for (int i_41 = 0; i_41 < 56; ++i_41) {
    float2 __31;
      float2 v__39 = *(float2*)(acc_o + (i_41 * 2));
      float2 v__40 = make_float2(lsum[(((i_41 / 28) * 2) + (i_41 & 1))], lsum[(((i_41 / 28) * 2) + (i_41 & 1))]);
      __31.x = (v__39.x/v__40.x);
      __31.y = (v__39.y/v__40.y);
    *(float2*)(acc_o + (i_41 * 2)) = __31;
  }
  __syncthreads();
  #pragma unroll
  for (int i_42 = 0; i_42 < 14; ++i_42) {
    tl::ptx_stmatrix_m8n8_x4((&(((half_t*)O_shared)[((((((((int)threadIdx.x) >> 5) * 3584) + ((i_42 / 7) * 1792)) + ((((int)threadIdx.x) & 15) * 112)) + ((i_42 % 7) * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), __pack_half2(((half_t)acc_o[(i_42 * 8)]), ((half_t)acc_o[((i_42 * 8) + 1)])), __pack_half2(((half_t)acc_o[((i_42 * 8) + 2)]), ((half_t)acc_o[((i_42 * 8) + 3)])), __pack_half2(((half_t)acc_o[((i_42 * 8) + 4)]), ((half_t)acc_o[((i_42 * 8) + 5)])), __pack_half2(((half_t)acc_o[((i_42 * 8) + 6)]), ((half_t)acc_o[((i_42 * 8) + 7)])));
  }
  __syncthreads();
  #pragma unroll
  for (int i_43 = 0; i_43 < 14; ++i_43) {
    if (((((int)blockIdx.x) * 4) + (((i_43 * 32) + (((int)threadIdx.x) >> 1)) / 112)) < 169) {
      *(uint4*)(Out + ((((((int)blockIdx.x) * 93184) + ((((i_43 * 32) + (((int)threadIdx.x) >> 1)) / 7) * 1456)) + (((int)blockIdx.y) * 112)) + ((((i_43 * 64) + ((int)threadIdx.x)) % 14) * 8))) = *(uint4*)(((half_t*)O_shared) + ((i_43 * 512) + (((int)threadIdx.x) * 8)));
    }
  }
}

