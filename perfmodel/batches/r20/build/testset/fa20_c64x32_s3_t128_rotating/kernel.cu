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
  void* K_shared = ((void*)((char*)buf_dyn_shmem + 14336));
  void* V_shared = ((void*)((char*)buf_dyn_shmem + 35840));
  float acc_o[56];
  float lsum[2];
  float m[2];
  float acc_s[16];
  float m_prev[2];
  float sc[2];
  half_t acc_s_cast[16];
  float ssum[2];
  float m_clear[2];
  float m_clear_1[2];
  float m_clear_2[2];
  float m_clear_3[2];
  #pragma unroll
  for (int i = 0; i < 7; ++i) {
    half_t broadcast_var = half_t(0x0p+0f/*0.000000e+00*/);
    uint4 condval;
    if ((((((int)blockIdx.x) * 2) + (((i * 64) + (((int)threadIdx.x) >> 1)) / 224)) < 65)) {
      condval = *(uint4*)(Q + ((((((int)blockIdx.x) * 71680) + ((((i * 64) + (((int)threadIdx.x) >> 1)) / 7) * 1120)) + (((int)blockIdx.y) * 112)) + ((((i * 128) + ((int)threadIdx.x)) % 14) * 8)));
    } else {
      condval = make_uint4(__pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var));
    }
    *(uint4*)(((half_t*)Q_shared) + ((((((i * 64) + (((int)threadIdx.x) >> 1)) % 7) * 1024) + ((((i * 64) + (((int)threadIdx.x) >> 1)) / 7) * 16)) + (((((((i * 16) + (((int)threadIdx.x) >> 3)) % 14) / 7) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval;
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 14; ++i_1) {
    float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_o + (i_1 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
  }
  float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(lsum + 0) = make_float2(broadcast_var_2, broadcast_var_2);
  float broadcast_var_3 = -CUDART_INF_F;
  *(float2*)(m + 0) = make_float2(broadcast_var_3, broadcast_var_3);
  __syncthreads();
  #pragma unroll
  for (int i_2 = 0; i_2 < 7; ++i_2) {
    tl::cp_async_gs<8>((&(((half_t*)K_shared)[(((((((i_2 * 32) + (((int)threadIdx.x) >> 2)) % 7) * 512) + ((((i_2 * 32) + (((int)threadIdx.x) >> 2)) / 7) * 16)) + (((((((i_2 * 8) + (((int)threadIdx.x) >> 4)) % 14) / 7) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 1) * 4))])), (&(K[((((((i_2 * 32) + (((int)threadIdx.x) >> 2)) / 7) * 1120) + (((int)blockIdx.y) * 112)) + ((((i_2 * 128) + ((int)threadIdx.x)) % 28) * 4))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_3 = 0; i_3 < 7; ++i_3) {
    tl::cp_async_gs<8>((&(((half_t*)V_shared)[(((((((i_3 * 32) + (((int)threadIdx.x) >> 2)) % 7) * 512) + ((((i_3 * 32) + (((int)threadIdx.x) >> 2)) / 7) * 16)) + (((((((i_3 * 8) + (((int)threadIdx.x) >> 4)) % 14) / 7) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 1) * 4))])), (&(V[((((((i_3 * 32) + (((int)threadIdx.x) >> 2)) / 7) * 1120) + (((int)blockIdx.y) * 112)) + ((((i_3 * 128) + ((int)threadIdx.x)) % 28) * 4))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_4 = 0; i_4 < 7; ++i_4) {
    tl::cp_async_gs<8>((&(((half_t*)K_shared)[((((((((i_4 * 32) + (((int)threadIdx.x) >> 2)) % 7) * 512) + ((((i_4 * 32) + (((int)threadIdx.x) >> 2)) / 7) * 16)) + (((((((i_4 * 8) + (((int)threadIdx.x) >> 4)) % 14) / 7) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 3584)])), (&(K[(((((((i_4 * 32) + (((int)threadIdx.x) >> 2)) / 7) * 1120) + (((int)blockIdx.y) * 112)) + ((((i_4 * 128) + ((int)threadIdx.x)) % 28) * 4)) + 35840)])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_5 = 0; i_5 < 7; ++i_5) {
    tl::cp_async_gs<8>((&(((half_t*)V_shared)[((((((((i_5 * 32) + (((int)threadIdx.x) >> 2)) % 7) * 512) + ((((i_5 * 32) + (((int)threadIdx.x) >> 2)) / 7) * 16)) + (((((((i_5 * 8) + (((int)threadIdx.x) >> 4)) % 14) / 7) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 3584)])), (&(V[(((((((i_5 * 32) + (((int)threadIdx.x) >> 2)) / 7) * 1120) + (((int)blockIdx.y) * 112)) + ((((i_5 * 128) + ((int)threadIdx.x)) % 28) * 4)) + 35840)])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_6 = 0; i_6 < 7; ++i_6) {
    tl::cp_async_gs<8>((&(((half_t*)K_shared)[((((((((i_6 * 32) + (((int)threadIdx.x) >> 2)) % 7) * 512) + ((((i_6 * 32) + (((int)threadIdx.x) >> 2)) / 7) * 16)) + (((((((i_6 * 8) + (((int)threadIdx.x) >> 4)) % 14) / 7) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 7168)])), (&(K[(((((((i_6 * 32) + (((int)threadIdx.x) >> 2)) / 7) * 1120) + (((int)blockIdx.y) * 112)) + ((((i_6 * 128) + ((int)threadIdx.x)) % 28) * 4)) + 71680)])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_7 = 0; i_7 < 7; ++i_7) {
    tl::cp_async_gs<8>((&(((half_t*)V_shared)[((((((((i_7 * 32) + (((int)threadIdx.x) >> 2)) % 7) * 512) + ((((i_7 * 32) + (((int)threadIdx.x) >> 2)) / 7) * 16)) + (((((((i_7 * 8) + (((int)threadIdx.x) >> 4)) % 14) / 7) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 7168)])), (&(V[(((((((i_7 * 32) + (((int)threadIdx.x) >> 2)) / 7) * 1120) + (((int)blockIdx.y) * 112)) + ((((i_7 * 128) + ((int)threadIdx.x)) % 28) * 4)) + 71680)])));
  }
  tl::cp_async_commit();
  for (int k = 0; k < 62; ++k) {
    #pragma unroll
    for (int i_8 = 0; i_8 < 4; ++i_8) {
      float broadcast_var_4 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_s + (i_8 * 4)) = make_float4(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4);
    }
    tl::cp_async_wait<5>();
    __syncthreads();
    {
      half_t A_local[8];
      half_t B_local[16];
      for (int ki = 0; ki < 7; ++ki) {
        tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[((((ki * 1024) + ((((int)threadIdx.x) >> 5) * 256)) + ((((int)threadIdx.x) & 15) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(A_local[0])));
        #pragma unroll
        for (int i_9 = 0; i_9 < 2; ++i_9) {
          tl::ptx_ldmatrix_x4((&(((half_t*)K_shared)[(((((((k % 3) * 3584) + (ki * 512)) + (i_9 * 256)) + (((((int)threadIdx.x) & 31) >> 4) * 128)) + ((((int)threadIdx.x) & 7) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(B_local[(i_9 * 8)])));
        }
        for (int j = 0; j < 2; ++j) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (j * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_10 = 0; i_10 < 7; ++i_10) {
      tl::cp_async_gs<8>((&(((half_t*)K_shared)[((((((k % 3) * 3584) + ((((i_10 * 32) + (((int)threadIdx.x) >> 2)) % 7) * 512)) + ((((i_10 * 32) + (((int)threadIdx.x) >> 2)) / 7) * 16)) + (((((((i_10 * 8) + (((int)threadIdx.x) >> 4)) % 14) / 7) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 1) * 4))])), (&(K[(((((k * 35840) + ((((i_10 * 32) + (((int)threadIdx.x) >> 2)) / 7) * 1120)) + (((int)blockIdx.y) * 112)) + ((((i_10 * 128) + ((int)threadIdx.x)) % 28) * 4)) + 107520)])));
    }
    tl::cp_async_commit();
    *(float2*)(m_prev + 0) = *(float2*)(m + 0);
    float broadcast_var_5 = -CUDART_INF_F;
    *(float2*)(m + 0) = make_float2(broadcast_var_5, broadcast_var_5);
    #pragma unroll
    for (int i_11 = 0; i_11 < 2; ++i_11) {
      m_clear[i_11] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 8; ++rv) {
        m_clear[i_11] = max(m_clear[i_11], acc_s[((((rv & 3) * 4) + (i_11 * 2)) + (rv >> 2))]);
      }
      m_clear[i_11] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(m_clear[i_11]);
      m[i_11] = max(m[i_11], m_clear[i_11]);
    }
    float broadcast_var_6 = 0x1.172feb597d322p-3f/*1.363219e-01*/;
    float broadcast_var_7 = 0x1.172feb597d322p-3f/*1.363219e-01*/;
    float2 __1;
    float2 __2;
      float2 __3;
        float2 v_ = *(float2*)(m_prev + 0);
        float2 v__1 = make_float2(broadcast_var_6, broadcast_var_6);
        *(float2*)(&(__3.x)) = tl::mul2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)));
      float2 __4;
        float2 v__2 = *(float2*)(m + 0);
        float2 v__3 = make_float2(broadcast_var_7, broadcast_var_7);
        *(float2*)(&(__4.x)) = tl::mul2(*(float2*)(&(v__2.x)), *(float2*)(&(v__3.x)));
      *(float2*)(&(__2.x)) = tl::sub2(*(float2*)(&(__3.x)), *(float2*)(&(__4.x)));
    __1.x = exp2f(__2.x);
    __1.y = exp2f(__2.y);
    *(float2*)(sc + 0) = __1;
    #pragma unroll
    for (int i_12 = 0; i_12 < 28; ++i_12) {
      float2 __5;
        float2 v__4 = *(float2*)(acc_o + (i_12 * 2));
        float2 v__5 = make_float2(sc[(i_12 & 1)], sc[(i_12 & 1)]);
        *(float2*)(&(__5.x)) = tl::mul2(*(float2*)(&(v__4.x)), *(float2*)(&(v__5.x)));
      *(float2*)(acc_o + (i_12 * 2)) = __5;
    }
    #pragma unroll
    for (int i_13 = 0; i_13 < 8; ++i_13) {
      float broadcast_var_8 = 0x1.172feb597d322p-3f/*1.363219e-01*/;
      float2 __6;
      float2 __7;
        float2 __8;
          float2 v__6 = *(float2*)(acc_s + (i_13 * 2));
          float2 v__7 = make_float2(broadcast_var_8, broadcast_var_8);
          *(float2*)(&(__8.x)) = tl::mul2(*(float2*)(&(v__6.x)), *(float2*)(&(v__7.x)));
        float2 v__8 = make_float2((m[(i_13 & 1)] * 0x1.172feb597d322p-3f/*1.363219e-01*/), (m[(i_13 & 1)] * 0x1.172feb597d322p-3f/*1.363219e-01*/));
        *(float2*)(&(__7.x)) = tl::sub2(*(float2*)(&(__8.x)), *(float2*)(&(v__8.x)));
      __6.x = exp2f(__7.x);
      __6.y = exp2f(__7.y);
      *(float2*)(acc_s + (i_13 * 2)) = __6;
    }
    #pragma unroll
    for (int i_14 = 0; i_14 < 4; ++i_14) {
      uint2 __9;
      float4 v__9 = *(float4*)(acc_s + (i_14 * 4));
      ((half2*)(&__9))[0] = __float22half2_rn(((float2*)(&v__9))[0]);
      ((half2*)(&__9))[1] = __float22half2_rn(((float2*)(&v__9))[1]);
      *(uint2*)(acc_s_cast + (i_14 * 4)) = __9;
    }
    tl::cp_async_wait<5>();
    __syncthreads();
    {
      half_t B_local_1[56];
      for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
        #pragma unroll
        for (int i_15 = 0; i_15 < 7; ++i_15) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)V_shared)[((((((k % 3) * 3584) + (i_15 * 512)) + (ki_1 * 256)) + ((((int)threadIdx.x) & 15) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(B_local_1[(i_15 * 8)])));
        }
        for (int j_1 = 0; j_1 < 7; ++j_1) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_1 * 8)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_1 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_1 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_16 = 0; i_16 < 7; ++i_16) {
      tl::cp_async_gs<8>((&(((half_t*)V_shared)[((((((k % 3) * 3584) + ((((i_16 * 32) + (((int)threadIdx.x) >> 2)) % 7) * 512)) + ((((i_16 * 32) + (((int)threadIdx.x) >> 2)) / 7) * 16)) + (((((((i_16 * 8) + (((int)threadIdx.x) >> 4)) % 14) / 7) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 1) * 4))])), (&(V[(((((k * 35840) + ((((i_16 * 32) + (((int)threadIdx.x) >> 2)) / 7) * 1120)) + (((int)blockIdx.y) * 112)) + ((((i_16 * 128) + ((int)threadIdx.x)) % 28) * 4)) + 107520)])));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_17 = 0; i_17 < 2; ++i_17) {
      ssum[i_17] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 8; ++rv_1) {
        ssum[i_17] = (ssum[i_17] + acc_s[((((rv_1 & 3) * 4) + (i_17 * 2)) + (rv_1 >> 2))]);
      }
      ssum[i_17] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(ssum[i_17]);
    }
    float2 __10;
      float2 v__10 = *(float2*)(lsum + 0);
      float2 v__11 = *(float2*)(sc + 0);
      float2 v__12 = *(float2*)(ssum + 0);
      *(float2*)(&(__10.x)) = tl::fma2(*(float2*)(&(v__10.x)), *(float2*)(&(v__11.x)), *(float2*)(&(v__12.x)));
    *(float2*)(lsum + 0) = __10;
  }
  #pragma unroll
  for (int i_18 = 0; i_18 < 4; ++i_18) {
    float broadcast_var_9 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_s + (i_18 * 4)) = make_float4(broadcast_var_9, broadcast_var_9, broadcast_var_9, broadcast_var_9);
  }
  tl::cp_async_wait<5>();
  __syncthreads();
  {
    half_t A_local_1[8];
    half_t B_local_2[16];
    for (int ki_2 = 0; ki_2 < 7; ++ki_2) {
      tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[((((ki_2 * 1024) + ((((int)threadIdx.x) >> 5) * 256)) + ((((int)threadIdx.x) & 15) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(A_local_1[0])));
      #pragma unroll
      for (int i_19 = 0; i_19 < 2; ++i_19) {
        tl::ptx_ldmatrix_x4((&(((half_t*)K_shared)[((((((ki_2 * 512) + (i_19 * 256)) + (((((int)threadIdx.x) & 31) >> 4) * 128)) + ((((int)threadIdx.x) & 7) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 7168)])), (&(B_local_2[(i_19 * 8)])));
      }
      for (int j_2 = 0; j_2 < 2; ++j_2) {
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (j_2 * 8)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_2 + (j_2 * 8)));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((j_2 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_2 + ((j_2 * 8) + 4)));
      }
    }
  }
  *(float2*)(m_prev + 0) = *(float2*)(m + 0);
  float broadcast_var_10 = -CUDART_INF_F;
  *(float2*)(m + 0) = make_float2(broadcast_var_10, broadcast_var_10);
  #pragma unroll
  for (int i_20 = 0; i_20 < 2; ++i_20) {
    m_clear_1[i_20] = -CUDART_INF_F;
    #pragma unroll
    for (int rv_2 = 0; rv_2 < 8; ++rv_2) {
      m_clear_1[i_20] = max(m_clear_1[i_20], acc_s[((((rv_2 & 3) * 4) + (i_20 * 2)) + (rv_2 >> 2))]);
    }
    m_clear_1[i_20] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(m_clear_1[i_20]);
    m[i_20] = max(m[i_20], m_clear_1[i_20]);
  }
  float broadcast_var_11 = 0x1.172feb597d322p-3f/*1.363219e-01*/;
  float broadcast_var_12 = 0x1.172feb597d322p-3f/*1.363219e-01*/;
  float2 __11;
  float2 __12;
    float2 __13;
      float2 v__13 = *(float2*)(m_prev + 0);
      float2 v__14 = make_float2(broadcast_var_11, broadcast_var_11);
      *(float2*)(&(__13.x)) = tl::mul2(*(float2*)(&(v__13.x)), *(float2*)(&(v__14.x)));
    float2 __14;
      float2 v__15 = *(float2*)(m + 0);
      float2 v__16 = make_float2(broadcast_var_12, broadcast_var_12);
      *(float2*)(&(__14.x)) = tl::mul2(*(float2*)(&(v__15.x)), *(float2*)(&(v__16.x)));
    *(float2*)(&(__12.x)) = tl::sub2(*(float2*)(&(__13.x)), *(float2*)(&(__14.x)));
  __11.x = exp2f(__12.x);
  __11.y = exp2f(__12.y);
  *(float2*)(sc + 0) = __11;
  #pragma unroll
  for (int i_21 = 0; i_21 < 28; ++i_21) {
    float2 __15;
      float2 v__17 = *(float2*)(acc_o + (i_21 * 2));
      float2 v__18 = make_float2(sc[(i_21 & 1)], sc[(i_21 & 1)]);
      *(float2*)(&(__15.x)) = tl::mul2(*(float2*)(&(v__17.x)), *(float2*)(&(v__18.x)));
    *(float2*)(acc_o + (i_21 * 2)) = __15;
  }
  #pragma unroll
  for (int i_22 = 0; i_22 < 8; ++i_22) {
    float broadcast_var_13 = 0x1.172feb597d322p-3f/*1.363219e-01*/;
    float2 __16;
    float2 __17;
      float2 __18;
        float2 v__19 = *(float2*)(acc_s + (i_22 * 2));
        float2 v__20 = make_float2(broadcast_var_13, broadcast_var_13);
        *(float2*)(&(__18.x)) = tl::mul2(*(float2*)(&(v__19.x)), *(float2*)(&(v__20.x)));
      float2 v__21 = make_float2((m[(i_22 & 1)] * 0x1.172feb597d322p-3f/*1.363219e-01*/), (m[(i_22 & 1)] * 0x1.172feb597d322p-3f/*1.363219e-01*/));
      *(float2*)(&(__17.x)) = tl::sub2(*(float2*)(&(__18.x)), *(float2*)(&(v__21.x)));
    __16.x = exp2f(__17.x);
    __16.y = exp2f(__17.y);
    *(float2*)(acc_s + (i_22 * 2)) = __16;
  }
  #pragma unroll
  for (int i_23 = 0; i_23 < 4; ++i_23) {
    uint2 __19;
    float4 v__22 = *(float4*)(acc_s + (i_23 * 4));
    ((half2*)(&__19))[0] = __float22half2_rn(((float2*)(&v__22))[0]);
    ((half2*)(&__19))[1] = __float22half2_rn(((float2*)(&v__22))[1]);
    *(uint2*)(acc_s_cast + (i_23 * 4)) = __19;
  }
  tl::cp_async_wait<4>();
  __syncthreads();
  {
    half_t B_local_3[56];
    for (int ki_3 = 0; ki_3 < 2; ++ki_3) {
      #pragma unroll
      for (int i_24 = 0; i_24 < 7; ++i_24) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)V_shared)[(((((i_24 * 512) + (ki_3 * 256)) + ((((int)threadIdx.x) & 15) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 7168)])), (&(B_local_3[(i_24 * 8)])));
      }
      for (int j_3 = 0; j_3 < 7; ++j_3) {
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_3 * 8)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_3 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + (j_3 * 8)));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_3 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_3 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + ((j_3 * 8) + 4)));
      }
    }
  }
  #pragma unroll
  for (int i_25 = 0; i_25 < 2; ++i_25) {
    ssum[i_25] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_3 = 0; rv_3 < 8; ++rv_3) {
      ssum[i_25] = (ssum[i_25] + acc_s[((((rv_3 & 3) * 4) + (i_25 * 2)) + (rv_3 >> 2))]);
    }
    ssum[i_25] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(ssum[i_25]);
  }
  float2 __20;
    float2 v__23 = *(float2*)(lsum + 0);
    float2 v__24 = *(float2*)(sc + 0);
    float2 v__25 = *(float2*)(ssum + 0);
    *(float2*)(&(__20.x)) = tl::fma2(*(float2*)(&(v__23.x)), *(float2*)(&(v__24.x)), *(float2*)(&(v__25.x)));
  *(float2*)(lsum + 0) = __20;
  #pragma unroll
  for (int i_26 = 0; i_26 < 4; ++i_26) {
    float broadcast_var_14 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_s + (i_26 * 4)) = make_float4(broadcast_var_14, broadcast_var_14, broadcast_var_14, broadcast_var_14);
  }
  tl::cp_async_wait<3>();
  __syncthreads();
  {
    half_t A_local_2[8];
    half_t B_local_4[16];
    for (int ki_4 = 0; ki_4 < 7; ++ki_4) {
      tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[((((ki_4 * 1024) + ((((int)threadIdx.x) >> 5) * 256)) + ((((int)threadIdx.x) & 15) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(A_local_2[0])));
      #pragma unroll
      for (int i_27 = 0; i_27 < 2; ++i_27) {
        tl::ptx_ldmatrix_x4((&(((half_t*)K_shared)[(((((ki_4 * 512) + (i_27 * 256)) + (((((int)threadIdx.x) & 31) >> 4) * 128)) + ((((int)threadIdx.x) & 7) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(B_local_4[(i_27 * 8)])));
      }
      for (int j_4 = 0; j_4 < 2; ++j_4) {
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (j_4 * 8)), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_4 + (j_4 * 8)));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((j_4 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_4 + ((j_4 * 8) + 4)));
      }
    }
  }
  *(float2*)(m_prev + 0) = *(float2*)(m + 0);
  float broadcast_var_15 = -CUDART_INF_F;
  *(float2*)(m + 0) = make_float2(broadcast_var_15, broadcast_var_15);
  #pragma unroll
  for (int i_28 = 0; i_28 < 2; ++i_28) {
    m_clear_2[i_28] = -CUDART_INF_F;
    #pragma unroll
    for (int rv_4 = 0; rv_4 < 8; ++rv_4) {
      m_clear_2[i_28] = max(m_clear_2[i_28], acc_s[((((rv_4 & 3) * 4) + (i_28 * 2)) + (rv_4 >> 2))]);
    }
    m_clear_2[i_28] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(m_clear_2[i_28]);
    m[i_28] = max(m[i_28], m_clear_2[i_28]);
  }
  float broadcast_var_16 = 0x1.172feb597d322p-3f/*1.363219e-01*/;
  float broadcast_var_17 = 0x1.172feb597d322p-3f/*1.363219e-01*/;
  float2 __21;
  float2 __22;
    float2 __23;
      float2 v__26 = *(float2*)(m_prev + 0);
      float2 v__27 = make_float2(broadcast_var_16, broadcast_var_16);
      *(float2*)(&(__23.x)) = tl::mul2(*(float2*)(&(v__26.x)), *(float2*)(&(v__27.x)));
    float2 __24;
      float2 v__28 = *(float2*)(m + 0);
      float2 v__29 = make_float2(broadcast_var_17, broadcast_var_17);
      *(float2*)(&(__24.x)) = tl::mul2(*(float2*)(&(v__28.x)), *(float2*)(&(v__29.x)));
    *(float2*)(&(__22.x)) = tl::sub2(*(float2*)(&(__23.x)), *(float2*)(&(__24.x)));
  __21.x = exp2f(__22.x);
  __21.y = exp2f(__22.y);
  *(float2*)(sc + 0) = __21;
  #pragma unroll
  for (int i_29 = 0; i_29 < 28; ++i_29) {
    float2 __25;
      float2 v__30 = *(float2*)(acc_o + (i_29 * 2));
      float2 v__31 = make_float2(sc[(i_29 & 1)], sc[(i_29 & 1)]);
      *(float2*)(&(__25.x)) = tl::mul2(*(float2*)(&(v__30.x)), *(float2*)(&(v__31.x)));
    *(float2*)(acc_o + (i_29 * 2)) = __25;
  }
  #pragma unroll
  for (int i_30 = 0; i_30 < 8; ++i_30) {
    float broadcast_var_18 = 0x1.172feb597d322p-3f/*1.363219e-01*/;
    float2 __26;
    float2 __27;
      float2 __28;
        float2 v__32 = *(float2*)(acc_s + (i_30 * 2));
        float2 v__33 = make_float2(broadcast_var_18, broadcast_var_18);
        *(float2*)(&(__28.x)) = tl::mul2(*(float2*)(&(v__32.x)), *(float2*)(&(v__33.x)));
      float2 v__34 = make_float2((m[(i_30 & 1)] * 0x1.172feb597d322p-3f/*1.363219e-01*/), (m[(i_30 & 1)] * 0x1.172feb597d322p-3f/*1.363219e-01*/));
      *(float2*)(&(__27.x)) = tl::sub2(*(float2*)(&(__28.x)), *(float2*)(&(v__34.x)));
    __26.x = exp2f(__27.x);
    __26.y = exp2f(__27.y);
    *(float2*)(acc_s + (i_30 * 2)) = __26;
  }
  #pragma unroll
  for (int i_31 = 0; i_31 < 4; ++i_31) {
    uint2 __29;
    float4 v__35 = *(float4*)(acc_s + (i_31 * 4));
    ((half2*)(&__29))[0] = __float22half2_rn(((float2*)(&v__35))[0]);
    ((half2*)(&__29))[1] = __float22half2_rn(((float2*)(&v__35))[1]);
    *(uint2*)(acc_s_cast + (i_31 * 4)) = __29;
  }
  tl::cp_async_wait<2>();
  __syncthreads();
  {
    half_t B_local_5[56];
    for (int ki_5 = 0; ki_5 < 2; ++ki_5) {
      #pragma unroll
      for (int i_32 = 0; i_32 < 7; ++i_32) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)V_shared)[((((i_32 * 512) + (ki_5 * 256)) + ((((int)threadIdx.x) & 15) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(B_local_5[(i_32 * 8)])));
      }
      for (int j_5 = 0; j_5 < 7; ++j_5) {
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_5 * 8)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_5 * 8)), reinterpret_cast<const unsigned*>(B_local_5 + (j_5 * 8)));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_5 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_5 * 8)), reinterpret_cast<const unsigned*>(B_local_5 + ((j_5 * 8) + 4)));
      }
    }
  }
  #pragma unroll
  for (int i_33 = 0; i_33 < 2; ++i_33) {
    ssum[i_33] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_5 = 0; rv_5 < 8; ++rv_5) {
      ssum[i_33] = (ssum[i_33] + acc_s[((((rv_5 & 3) * 4) + (i_33 * 2)) + (rv_5 >> 2))]);
    }
    ssum[i_33] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(ssum[i_33]);
  }
  float2 __30;
    float2 v__36 = *(float2*)(lsum + 0);
    float2 v__37 = *(float2*)(sc + 0);
    float2 v__38 = *(float2*)(ssum + 0);
    *(float2*)(&(__30.x)) = tl::fma2(*(float2*)(&(v__36.x)), *(float2*)(&(v__37.x)), *(float2*)(&(v__38.x)));
  *(float2*)(lsum + 0) = __30;
  #pragma unroll
  for (int i_34 = 0; i_34 < 4; ++i_34) {
    float broadcast_var_19 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_s + (i_34 * 4)) = make_float4(broadcast_var_19, broadcast_var_19, broadcast_var_19, broadcast_var_19);
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  {
    half_t A_local_3[8];
    half_t B_local_6[16];
    for (int ki_6 = 0; ki_6 < 7; ++ki_6) {
      tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[((((ki_6 * 1024) + ((((int)threadIdx.x) >> 5) * 256)) + ((((int)threadIdx.x) & 15) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(A_local_3[0])));
      #pragma unroll
      for (int i_35 = 0; i_35 < 2; ++i_35) {
        tl::ptx_ldmatrix_x4((&(((half_t*)K_shared)[((((((ki_6 * 512) + (i_35 * 256)) + (((((int)threadIdx.x) & 31) >> 4) * 128)) + ((((int)threadIdx.x) & 7) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 3584)])), (&(B_local_6[(i_35 * 8)])));
      }
      for (int j_6 = 0; j_6 < 2; ++j_6) {
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (j_6 * 8)), reinterpret_cast<const unsigned*>(A_local_3 + 0), reinterpret_cast<const unsigned*>(B_local_6 + (j_6 * 8)));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((j_6 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_3 + 0), reinterpret_cast<const unsigned*>(B_local_6 + ((j_6 * 8) + 4)));
      }
    }
  }
  *(float2*)(m_prev + 0) = *(float2*)(m + 0);
  float broadcast_var_20 = -CUDART_INF_F;
  *(float2*)(m + 0) = make_float2(broadcast_var_20, broadcast_var_20);
  #pragma unroll
  for (int i_36 = 0; i_36 < 2; ++i_36) {
    m_clear_3[i_36] = -CUDART_INF_F;
    #pragma unroll
    for (int rv_6 = 0; rv_6 < 8; ++rv_6) {
      m_clear_3[i_36] = max(m_clear_3[i_36], acc_s[((((rv_6 & 3) * 4) + (i_36 * 2)) + (rv_6 >> 2))]);
    }
    m_clear_3[i_36] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(m_clear_3[i_36]);
    m[i_36] = max(m[i_36], m_clear_3[i_36]);
  }
  float broadcast_var_21 = 0x1.172feb597d322p-3f/*1.363219e-01*/;
  float broadcast_var_22 = 0x1.172feb597d322p-3f/*1.363219e-01*/;
  float2 __31;
  float2 __32;
    float2 __33;
      float2 v__39 = *(float2*)(m_prev + 0);
      float2 v__40 = make_float2(broadcast_var_21, broadcast_var_21);
      *(float2*)(&(__33.x)) = tl::mul2(*(float2*)(&(v__39.x)), *(float2*)(&(v__40.x)));
    float2 __34;
      float2 v__41 = *(float2*)(m + 0);
      float2 v__42 = make_float2(broadcast_var_22, broadcast_var_22);
      *(float2*)(&(__34.x)) = tl::mul2(*(float2*)(&(v__41.x)), *(float2*)(&(v__42.x)));
    *(float2*)(&(__32.x)) = tl::sub2(*(float2*)(&(__33.x)), *(float2*)(&(__34.x)));
  __31.x = exp2f(__32.x);
  __31.y = exp2f(__32.y);
  *(float2*)(sc + 0) = __31;
  #pragma unroll
  for (int i_37 = 0; i_37 < 28; ++i_37) {
    float2 __35;
      float2 v__43 = *(float2*)(acc_o + (i_37 * 2));
      float2 v__44 = make_float2(sc[(i_37 & 1)], sc[(i_37 & 1)]);
      *(float2*)(&(__35.x)) = tl::mul2(*(float2*)(&(v__43.x)), *(float2*)(&(v__44.x)));
    *(float2*)(acc_o + (i_37 * 2)) = __35;
  }
  #pragma unroll
  for (int i_38 = 0; i_38 < 8; ++i_38) {
    float broadcast_var_23 = 0x1.172feb597d322p-3f/*1.363219e-01*/;
    float2 __36;
    float2 __37;
      float2 __38;
        float2 v__45 = *(float2*)(acc_s + (i_38 * 2));
        float2 v__46 = make_float2(broadcast_var_23, broadcast_var_23);
        *(float2*)(&(__38.x)) = tl::mul2(*(float2*)(&(v__45.x)), *(float2*)(&(v__46.x)));
      float2 v__47 = make_float2((m[(i_38 & 1)] * 0x1.172feb597d322p-3f/*1.363219e-01*/), (m[(i_38 & 1)] * 0x1.172feb597d322p-3f/*1.363219e-01*/));
      *(float2*)(&(__37.x)) = tl::sub2(*(float2*)(&(__38.x)), *(float2*)(&(v__47.x)));
    __36.x = exp2f(__37.x);
    __36.y = exp2f(__37.y);
    *(float2*)(acc_s + (i_38 * 2)) = __36;
  }
  #pragma unroll
  for (int i_39 = 0; i_39 < 4; ++i_39) {
    uint2 __39;
    float4 v__48 = *(float4*)(acc_s + (i_39 * 4));
    ((half2*)(&__39))[0] = __float22half2_rn(((float2*)(&v__48))[0]);
    ((half2*)(&__39))[1] = __float22half2_rn(((float2*)(&v__48))[1]);
    *(uint2*)(acc_s_cast + (i_39 * 4)) = __39;
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    half_t B_local_7[56];
    for (int ki_7 = 0; ki_7 < 2; ++ki_7) {
      #pragma unroll
      for (int i_40 = 0; i_40 < 7; ++i_40) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)V_shared)[(((((i_40 * 512) + (ki_7 * 256)) + ((((int)threadIdx.x) & 15) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 3584)])), (&(B_local_7[(i_40 * 8)])));
      }
      for (int j_7 = 0; j_7 < 7; ++j_7) {
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_7 * 8)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_7 * 8)), reinterpret_cast<const unsigned*>(B_local_7 + (j_7 * 8)));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_7 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_7 * 8)), reinterpret_cast<const unsigned*>(B_local_7 + ((j_7 * 8) + 4)));
      }
    }
  }
  #pragma unroll
  for (int i_41 = 0; i_41 < 2; ++i_41) {
    ssum[i_41] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_7 = 0; rv_7 < 8; ++rv_7) {
      ssum[i_41] = (ssum[i_41] + acc_s[((((rv_7 & 3) * 4) + (i_41 * 2)) + (rv_7 >> 2))]);
    }
    ssum[i_41] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(ssum[i_41]);
  }
  float2 __40;
    float2 v__49 = *(float2*)(lsum + 0);
    float2 v__50 = *(float2*)(sc + 0);
    float2 v__51 = *(float2*)(ssum + 0);
    *(float2*)(&(__40.x)) = tl::fma2(*(float2*)(&(v__49.x)), *(float2*)(&(v__50.x)), *(float2*)(&(v__51.x)));
  *(float2*)(lsum + 0) = __40;
  #pragma unroll
  for (int i_42 = 0; i_42 < 28; ++i_42) {
    float2 __41;
      float2 v__52 = *(float2*)(acc_o + (i_42 * 2));
      float2 v__53 = make_float2(lsum[(i_42 & 1)], lsum[(i_42 & 1)]);
      __41.x = (v__52.x/v__53.x);
      __41.y = (v__52.y/v__53.y);
    *(float2*)(acc_o + (i_42 * 2)) = __41;
  }
  __syncthreads();
  #pragma unroll
  for (int i_43 = 0; i_43 < 7; ++i_43) {
    tl::ptx_stmatrix_m8n8_x4((&(((half_t*)O_shared)[(((((((int)threadIdx.x) >> 5) * 1792) + ((((int)threadIdx.x) & 15) * 112)) + (i_43 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), __pack_half2(((half_t)acc_o[(i_43 * 8)]), ((half_t)acc_o[((i_43 * 8) + 1)])), __pack_half2(((half_t)acc_o[((i_43 * 8) + 2)]), ((half_t)acc_o[((i_43 * 8) + 3)])), __pack_half2(((half_t)acc_o[((i_43 * 8) + 4)]), ((half_t)acc_o[((i_43 * 8) + 5)])), __pack_half2(((half_t)acc_o[((i_43 * 8) + 6)]), ((half_t)acc_o[((i_43 * 8) + 7)])));
  }
  __syncthreads();
  #pragma unroll
  for (int i_44 = 0; i_44 < 7; ++i_44) {
    if (((((int)blockIdx.x) * 2) + (((i_44 * 64) + (((int)threadIdx.x) >> 1)) / 224)) < 65) {
      *(uint4*)(Out + ((((((int)blockIdx.x) * 71680) + ((((i_44 * 64) + (((int)threadIdx.x) >> 1)) / 7) * 1120)) + (((int)blockIdx.y) * 112)) + ((((i_44 * 128) + ((int)threadIdx.x)) % 14) * 8))) = *(uint4*)(((half_t*)O_shared) + ((i_44 * 1024) + (((int)threadIdx.x) * 8)));
    }
  }
}

