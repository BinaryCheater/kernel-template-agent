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

extern "C" __global__ void part_kernel(const half_t* __restrict__ K, half_t* __restrict__ Opart, const half_t* __restrict__ Q, const half_t* __restrict__ V, half_t* __restrict__ glse);
extern "C" __global__ void __launch_bounds__(128, 1) part_kernel(const half_t* __restrict__ K, half_t* __restrict__ Opart, const half_t* __restrict__ Q, const half_t* __restrict__ V, half_t* __restrict__ glse) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* O_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* Q_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* K_shared = ((void*)((char*)buf_dyn_shmem + 14336));
  void* V_shared = ((void*)((char*)buf_dyn_shmem + 28672));
  float acc_o[56];
  float lsum[2];
  float m[2];
  float acc_s[16];
  float m_prev[2];
  float sc[2];
  float ssum[2];
  half_t acc_s_cast[16];
  float m_clear[2];
  float m_clear_1[2];
  float m_clear_2[2];
  #pragma unroll
  for (int i = 0; i < 7; ++i) {
    half_t broadcast_var = half_t(0x0p+0f/*0.000000e+00*/);
    uint4 condval;
    if ((((((i * 64) + (((int)threadIdx.x) >> 1)) / 21) + ((int)blockIdx.y)) < 13)) {
      condval = *(uint4*)(Q + (((i * 1024) + (((int)blockIdx.y) * 336)) + (((int)threadIdx.x) * 8)));
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
    tl::cp_async_gs<8>((&(((half_t*)K_shared)[(((((((i_2 * 32) + (((int)threadIdx.x) >> 2)) % 7) * 512) + ((((i_2 * 32) + (((int)threadIdx.x) >> 2)) / 7) * 16)) + (((((((i_2 * 8) + (((int)threadIdx.x) >> 4)) % 14) / 7) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 1) * 4))])), (&(K[((((((int)blockIdx.z) * 946400) + ((((i_2 * 32) + (((int)threadIdx.x) >> 2)) / 7) * 1456)) + (((int)blockIdx.y) * 112)) + ((((i_2 * 128) + ((int)threadIdx.x)) % 28) * 4))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_3 = 0; i_3 < 7; ++i_3) {
    tl::cp_async_gs<8>((&(((half_t*)V_shared)[(((((((i_3 * 32) + (((int)threadIdx.x) >> 2)) % 7) * 512) + ((((i_3 * 32) + (((int)threadIdx.x) >> 2)) / 7) * 16)) + (((((((i_3 * 8) + (((int)threadIdx.x) >> 4)) % 14) / 7) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 1) * 4))])), (&(V[((((((int)blockIdx.z) * 946400) + ((((i_3 * 32) + (((int)threadIdx.x) >> 2)) / 7) * 1456)) + (((int)blockIdx.y) * 112)) + ((((i_3 * 128) + ((int)threadIdx.x)) % 28) * 4))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_4 = 0; i_4 < 7; ++i_4) {
    tl::cp_async_gs<8>((&(((half_t*)K_shared)[((((((((i_4 * 32) + (((int)threadIdx.x) >> 2)) % 7) * 512) + ((((i_4 * 32) + (((int)threadIdx.x) >> 2)) / 7) * 16)) + (((((((i_4 * 8) + (((int)threadIdx.x) >> 4)) % 14) / 7) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 3584)])), (&(K[(((((((int)blockIdx.z) * 946400) + ((((i_4 * 32) + (((int)threadIdx.x) >> 2)) / 7) * 1456)) + (((int)blockIdx.y) * 112)) + ((((i_4 * 128) + ((int)threadIdx.x)) % 28) * 4)) + 46592)])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_5 = 0; i_5 < 7; ++i_5) {
    tl::cp_async_gs<8>((&(((half_t*)V_shared)[((((((((i_5 * 32) + (((int)threadIdx.x) >> 2)) % 7) * 512) + ((((i_5 * 32) + (((int)threadIdx.x) >> 2)) / 7) * 16)) + (((((((i_5 * 8) + (((int)threadIdx.x) >> 4)) % 14) / 7) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 3584)])), (&(V[(((((((int)blockIdx.z) * 946400) + ((((i_5 * 32) + (((int)threadIdx.x) >> 2)) / 7) * 1456)) + (((int)blockIdx.y) * 112)) + ((((i_5 * 128) + ((int)threadIdx.x)) % 28) * 4)) + 46592)])));
  }
  tl::cp_async_commit();
  for (int k = 0; k < 19; ++k) {
    #pragma unroll
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      float broadcast_var_4 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_s + (i_6 * 4)) = make_float4(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4);
    }
    tl::cp_async_wait<3>();
    __syncthreads();
    {
      half_t A_local[8];
      half_t B_local[16];
      for (int ki = 0; ki < 7; ++ki) {
        tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[((((ki * 1024) + ((((int)threadIdx.x) >> 5) * 256)) + ((((int)threadIdx.x) & 15) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(A_local[0])));
        #pragma unroll
        for (int i_7 = 0; i_7 < 2; ++i_7) {
          tl::ptx_ldmatrix_x4((&(((half_t*)K_shared)[(((((((k & 1) * 3584) + (ki * 512)) + (i_7 * 256)) + (((((int)threadIdx.x) & 31) >> 4) * 128)) + ((((int)threadIdx.x) & 7) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(B_local[(i_7 * 8)])));
        }
        for (int j = 0; j < 2; ++j) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (j * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_8 = 0; i_8 < 7; ++i_8) {
      tl::cp_async_gs_conditional<8>((&(((half_t*)K_shared)[((((((k & 1) * 3584) + ((((i_8 * 32) + (((int)threadIdx.x) >> 2)) % 7) * 512)) + ((((i_8 * 32) + (((int)threadIdx.x) >> 2)) / 7) * 16)) + (((((((i_8 * 8) + (((int)threadIdx.x) >> 4)) % 14) / 7) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 1) * 4))])), (&(K[((((((((int)blockIdx.z) * 946400) + (k * 46592)) + ((((i_8 * 32) + (((int)threadIdx.x) >> 2)) / 7) * 1456)) + (((int)blockIdx.y) * 112)) + ((((i_8 * 128) + ((int)threadIdx.x)) % 28) * 4)) + 93184)])), (((((((int)blockIdx.z) * 325) + (k * 16)) + (((i_8 * 32) + (((int)threadIdx.x) >> 2)) / 14)) < 1268) && ((((((int)blockIdx.z) * 325) + (k * 16)) + (((i_8 * 32) + (((int)threadIdx.x) >> 2)) / 14)) < 1268)));
    }
    tl::cp_async_commit();
    *(float2*)(m_prev + 0) = *(float2*)(m + 0);
    float broadcast_var_5 = -CUDART_INF_F;
    *(float2*)(m + 0) = make_float2(broadcast_var_5, broadcast_var_5);
    #pragma unroll
    for (int i_9 = 0; i_9 < 2; ++i_9) {
      m_clear[i_9] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 8; ++rv) {
        m_clear[i_9] = max(m_clear[i_9], acc_s[((((rv & 3) * 4) + (i_9 * 2)) + (rv >> 2))]);
      }
      m_clear[i_9] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(m_clear[i_9]);
      m[i_9] = max(m[i_9], m_clear[i_9]);
    }
    float2 __1;
      float2 v_ = *(float2*)(m + 0);
      float2 v__1 = *(float2*)(m_prev + 0);
      *(float2*)(&(__1.x)) = tl::max2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)));
    *(float2*)(m + 0) = __1;
    float broadcast_var_6 = 0x1.172feb597d322p-3f/*1.363219e-01*/;
    float broadcast_var_7 = 0x1.172feb597d322p-3f/*1.363219e-01*/;
    float2 __2;
    float2 __3;
      float2 __4;
        float2 v__2 = *(float2*)(m_prev + 0);
        float2 v__3 = make_float2(broadcast_var_6, broadcast_var_6);
        *(float2*)(&(__4.x)) = tl::mul2(*(float2*)(&(v__2.x)), *(float2*)(&(v__3.x)));
      float2 __5;
        float2 v__4 = *(float2*)(m + 0);
        float2 v__5 = make_float2(broadcast_var_7, broadcast_var_7);
        *(float2*)(&(__5.x)) = tl::mul2(*(float2*)(&(v__4.x)), *(float2*)(&(v__5.x)));
      *(float2*)(&(__3.x)) = tl::sub2(*(float2*)(&(__4.x)), *(float2*)(&(__5.x)));
    __2.x = exp2f(__3.x);
    __2.y = exp2f(__3.y);
    *(float2*)(sc + 0) = __2;
    #pragma unroll
    for (int i_10 = 0; i_10 < 8; ++i_10) {
      float broadcast_var_8 = 0x1.172feb597d322p-3f/*1.363219e-01*/;
      float2 __6;
      float2 __7;
        float2 __8;
          float2 v__6 = *(float2*)(acc_s + (i_10 * 2));
          float2 v__7 = make_float2(broadcast_var_8, broadcast_var_8);
          *(float2*)(&(__8.x)) = tl::mul2(*(float2*)(&(v__6.x)), *(float2*)(&(v__7.x)));
        float2 v__8 = make_float2((m[(i_10 & 1)] * 0x1.172feb597d322p-3f/*1.363219e-01*/), (m[(i_10 & 1)] * 0x1.172feb597d322p-3f/*1.363219e-01*/));
        *(float2*)(&(__7.x)) = tl::sub2(*(float2*)(&(__8.x)), *(float2*)(&(v__8.x)));
      __6.x = exp2f(__7.x);
      __6.y = exp2f(__7.y);
      *(float2*)(acc_s + (i_10 * 2)) = __6;
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 2; ++i_11) {
      ssum[i_11] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 8; ++rv_1) {
        ssum[i_11] = (ssum[i_11] + acc_s[((((rv_1 & 3) * 4) + (i_11 * 2)) + (rv_1 >> 2))]);
      }
      ssum[i_11] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(ssum[i_11]);
    }
    float2 __9;
      float2 v__9 = *(float2*)(lsum + 0);
      float2 v__10 = *(float2*)(sc + 0);
      float2 v__11 = *(float2*)(ssum + 0);
      *(float2*)(&(__9.x)) = tl::fma2(*(float2*)(&(v__9.x)), *(float2*)(&(v__10.x)), *(float2*)(&(v__11.x)));
    *(float2*)(lsum + 0) = __9;
    #pragma unroll
    for (int i_12 = 0; i_12 < 4; ++i_12) {
      uint2 __10;
      float4 v__12 = *(float4*)(acc_s + (i_12 * 4));
      ((half2*)(&__10))[0] = __float22half2_rn(((float2*)(&v__12))[0]);
      ((half2*)(&__10))[1] = __float22half2_rn(((float2*)(&v__12))[1]);
      *(uint2*)(acc_s_cast + (i_12 * 4)) = __10;
    }
    #pragma unroll
    for (int i_13 = 0; i_13 < 28; ++i_13) {
      float2 __11;
        float2 v__13 = *(float2*)(acc_o + (i_13 * 2));
        float2 v__14 = make_float2(sc[(i_13 & 1)], sc[(i_13 & 1)]);
        *(float2*)(&(__11.x)) = tl::mul2(*(float2*)(&(v__13.x)), *(float2*)(&(v__14.x)));
      *(float2*)(acc_o + (i_13 * 2)) = __11;
    }
    tl::cp_async_wait<3>();
    __syncthreads();
    {
      half_t B_local_1[56];
      for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
        #pragma unroll
        for (int i_14 = 0; i_14 < 7; ++i_14) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)V_shared)[((((((k & 1) * 3584) + (i_14 * 512)) + (ki_1 * 256)) + ((((int)threadIdx.x) & 15) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(B_local_1[(i_14 * 8)])));
        }
        for (int j_1 = 0; j_1 < 7; ++j_1) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_1 * 8)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_1 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_1 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_15 = 0; i_15 < 7; ++i_15) {
      tl::cp_async_gs_conditional<8>((&(((half_t*)V_shared)[((((((k & 1) * 3584) + ((((i_15 * 32) + (((int)threadIdx.x) >> 2)) % 7) * 512)) + ((((i_15 * 32) + (((int)threadIdx.x) >> 2)) / 7) * 16)) + (((((((i_15 * 8) + (((int)threadIdx.x) >> 4)) % 14) / 7) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 1) * 4))])), (&(V[((((((((int)blockIdx.z) * 946400) + (k * 46592)) + ((((i_15 * 32) + (((int)threadIdx.x) >> 2)) / 7) * 1456)) + (((int)blockIdx.y) * 112)) + ((((i_15 * 128) + ((int)threadIdx.x)) % 28) * 4)) + 93184)])), (((((((int)blockIdx.z) * 325) + (k * 16)) + (((i_15 * 32) + (((int)threadIdx.x) >> 2)) / 14)) < 1268) && ((((((int)blockIdx.z) * 325) + (k * 16)) + (((i_15 * 32) + (((int)threadIdx.x) >> 2)) / 14)) < 1268)));
    }
    tl::cp_async_commit();
  }
  #pragma unroll
  for (int i_16 = 0; i_16 < 4; ++i_16) {
    float broadcast_var_9 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_s + (i_16 * 4)) = make_float4(broadcast_var_9, broadcast_var_9, broadcast_var_9, broadcast_var_9);
  }
  tl::cp_async_wait<3>();
  __syncthreads();
  {
    half_t A_local_1[8];
    half_t B_local_2[16];
    for (int ki_2 = 0; ki_2 < 7; ++ki_2) {
      tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[((((ki_2 * 1024) + ((((int)threadIdx.x) >> 5) * 256)) + ((((int)threadIdx.x) & 15) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(A_local_1[0])));
      #pragma unroll
      for (int i_17 = 0; i_17 < 2; ++i_17) {
        tl::ptx_ldmatrix_x4((&(((half_t*)K_shared)[((((((ki_2 * 512) + (i_17 * 256)) + (((((int)threadIdx.x) & 31) >> 4) * 128)) + ((((int)threadIdx.x) & 7) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 3584)])), (&(B_local_2[(i_17 * 8)])));
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
  for (int i_18 = 0; i_18 < 2; ++i_18) {
    m_clear_1[i_18] = -CUDART_INF_F;
    #pragma unroll
    for (int rv_2 = 0; rv_2 < 8; ++rv_2) {
      m_clear_1[i_18] = max(m_clear_1[i_18], acc_s[((((rv_2 & 3) * 4) + (i_18 * 2)) + (rv_2 >> 2))]);
    }
    m_clear_1[i_18] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(m_clear_1[i_18]);
    m[i_18] = max(m[i_18], m_clear_1[i_18]);
  }
  float2 __12;
    float2 v__15 = *(float2*)(m + 0);
    float2 v__16 = *(float2*)(m_prev + 0);
    *(float2*)(&(__12.x)) = tl::max2(*(float2*)(&(v__15.x)), *(float2*)(&(v__16.x)));
  *(float2*)(m + 0) = __12;
  float broadcast_var_11 = 0x1.172feb597d322p-3f/*1.363219e-01*/;
  float broadcast_var_12 = 0x1.172feb597d322p-3f/*1.363219e-01*/;
  float2 __13;
  float2 __14;
    float2 __15;
      float2 v__17 = *(float2*)(m_prev + 0);
      float2 v__18 = make_float2(broadcast_var_11, broadcast_var_11);
      *(float2*)(&(__15.x)) = tl::mul2(*(float2*)(&(v__17.x)), *(float2*)(&(v__18.x)));
    float2 __16;
      float2 v__19 = *(float2*)(m + 0);
      float2 v__20 = make_float2(broadcast_var_12, broadcast_var_12);
      *(float2*)(&(__16.x)) = tl::mul2(*(float2*)(&(v__19.x)), *(float2*)(&(v__20.x)));
    *(float2*)(&(__14.x)) = tl::sub2(*(float2*)(&(__15.x)), *(float2*)(&(__16.x)));
  __13.x = exp2f(__14.x);
  __13.y = exp2f(__14.y);
  *(float2*)(sc + 0) = __13;
  #pragma unroll
  for (int i_19 = 0; i_19 < 8; ++i_19) {
    float broadcast_var_13 = 0x1.172feb597d322p-3f/*1.363219e-01*/;
    float2 __17;
    float2 __18;
      float2 __19;
        float2 v__21 = *(float2*)(acc_s + (i_19 * 2));
        float2 v__22 = make_float2(broadcast_var_13, broadcast_var_13);
        *(float2*)(&(__19.x)) = tl::mul2(*(float2*)(&(v__21.x)), *(float2*)(&(v__22.x)));
      float2 v__23 = make_float2((m[(i_19 & 1)] * 0x1.172feb597d322p-3f/*1.363219e-01*/), (m[(i_19 & 1)] * 0x1.172feb597d322p-3f/*1.363219e-01*/));
      *(float2*)(&(__18.x)) = tl::sub2(*(float2*)(&(__19.x)), *(float2*)(&(v__23.x)));
    __17.x = exp2f(__18.x);
    __17.y = exp2f(__18.y);
    *(float2*)(acc_s + (i_19 * 2)) = __17;
  }
  #pragma unroll
  for (int i_20 = 0; i_20 < 2; ++i_20) {
    ssum[i_20] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_3 = 0; rv_3 < 8; ++rv_3) {
      ssum[i_20] = (ssum[i_20] + acc_s[((((rv_3 & 3) * 4) + (i_20 * 2)) + (rv_3 >> 2))]);
    }
    ssum[i_20] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(ssum[i_20]);
  }
  float2 __20;
    float2 v__24 = *(float2*)(lsum + 0);
    float2 v__25 = *(float2*)(sc + 0);
    float2 v__26 = *(float2*)(ssum + 0);
    *(float2*)(&(__20.x)) = tl::fma2(*(float2*)(&(v__24.x)), *(float2*)(&(v__25.x)), *(float2*)(&(v__26.x)));
  *(float2*)(lsum + 0) = __20;
  #pragma unroll
  for (int i_21 = 0; i_21 < 4; ++i_21) {
    uint2 __21;
    float4 v__27 = *(float4*)(acc_s + (i_21 * 4));
    ((half2*)(&__21))[0] = __float22half2_rn(((float2*)(&v__27))[0]);
    ((half2*)(&__21))[1] = __float22half2_rn(((float2*)(&v__27))[1]);
    *(uint2*)(acc_s_cast + (i_21 * 4)) = __21;
  }
  #pragma unroll
  for (int i_22 = 0; i_22 < 28; ++i_22) {
    float2 __22;
      float2 v__28 = *(float2*)(acc_o + (i_22 * 2));
      float2 v__29 = make_float2(sc[(i_22 & 1)], sc[(i_22 & 1)]);
      *(float2*)(&(__22.x)) = tl::mul2(*(float2*)(&(v__28.x)), *(float2*)(&(v__29.x)));
    *(float2*)(acc_o + (i_22 * 2)) = __22;
  }
  tl::cp_async_wait<2>();
  __syncthreads();
  {
    half_t B_local_3[56];
    for (int ki_3 = 0; ki_3 < 2; ++ki_3) {
      #pragma unroll
      for (int i_23 = 0; i_23 < 7; ++i_23) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)V_shared)[(((((i_23 * 512) + (ki_3 * 256)) + ((((int)threadIdx.x) & 15) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 3584)])), (&(B_local_3[(i_23 * 8)])));
      }
      for (int j_3 = 0; j_3 < 7; ++j_3) {
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_3 * 8)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_3 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + (j_3 * 8)));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_3 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_3 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + ((j_3 * 8) + 4)));
      }
    }
  }
  #pragma unroll
  for (int i_24 = 0; i_24 < 4; ++i_24) {
    float broadcast_var_14 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_s + (i_24 * 4)) = make_float4(broadcast_var_14, broadcast_var_14, broadcast_var_14, broadcast_var_14);
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  {
    half_t A_local_2[8];
    half_t B_local_4[16];
    for (int ki_4 = 0; ki_4 < 7; ++ki_4) {
      tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[((((ki_4 * 1024) + ((((int)threadIdx.x) >> 5) * 256)) + ((((int)threadIdx.x) & 15) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(A_local_2[0])));
      #pragma unroll
      for (int i_25 = 0; i_25 < 2; ++i_25) {
        tl::ptx_ldmatrix_x4((&(((half_t*)K_shared)[(((((ki_4 * 512) + (i_25 * 256)) + (((((int)threadIdx.x) & 31) >> 4) * 128)) + ((((int)threadIdx.x) & 7) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(B_local_4[(i_25 * 8)])));
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
  for (int i_26 = 0; i_26 < 2; ++i_26) {
    m_clear_2[i_26] = -CUDART_INF_F;
    #pragma unroll
    for (int rv_4 = 0; rv_4 < 8; ++rv_4) {
      m_clear_2[i_26] = max(m_clear_2[i_26], acc_s[((((rv_4 & 3) * 4) + (i_26 * 2)) + (rv_4 >> 2))]);
    }
    m_clear_2[i_26] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(m_clear_2[i_26]);
    m[i_26] = max(m[i_26], m_clear_2[i_26]);
  }
  float2 __23;
    float2 v__30 = *(float2*)(m + 0);
    float2 v__31 = *(float2*)(m_prev + 0);
    *(float2*)(&(__23.x)) = tl::max2(*(float2*)(&(v__30.x)), *(float2*)(&(v__31.x)));
  *(float2*)(m + 0) = __23;
  float broadcast_var_16 = 0x1.172feb597d322p-3f/*1.363219e-01*/;
  float broadcast_var_17 = 0x1.172feb597d322p-3f/*1.363219e-01*/;
  float2 __24;
  float2 __25;
    float2 __26;
      float2 v__32 = *(float2*)(m_prev + 0);
      float2 v__33 = make_float2(broadcast_var_16, broadcast_var_16);
      *(float2*)(&(__26.x)) = tl::mul2(*(float2*)(&(v__32.x)), *(float2*)(&(v__33.x)));
    float2 __27;
      float2 v__34 = *(float2*)(m + 0);
      float2 v__35 = make_float2(broadcast_var_17, broadcast_var_17);
      *(float2*)(&(__27.x)) = tl::mul2(*(float2*)(&(v__34.x)), *(float2*)(&(v__35.x)));
    *(float2*)(&(__25.x)) = tl::sub2(*(float2*)(&(__26.x)), *(float2*)(&(__27.x)));
  __24.x = exp2f(__25.x);
  __24.y = exp2f(__25.y);
  *(float2*)(sc + 0) = __24;
  #pragma unroll
  for (int i_27 = 0; i_27 < 8; ++i_27) {
    float broadcast_var_18 = 0x1.172feb597d322p-3f/*1.363219e-01*/;
    float2 __28;
    float2 __29;
      float2 __30;
        float2 v__36 = *(float2*)(acc_s + (i_27 * 2));
        float2 v__37 = make_float2(broadcast_var_18, broadcast_var_18);
        *(float2*)(&(__30.x)) = tl::mul2(*(float2*)(&(v__36.x)), *(float2*)(&(v__37.x)));
      float2 v__38 = make_float2((m[(i_27 & 1)] * 0x1.172feb597d322p-3f/*1.363219e-01*/), (m[(i_27 & 1)] * 0x1.172feb597d322p-3f/*1.363219e-01*/));
      *(float2*)(&(__29.x)) = tl::sub2(*(float2*)(&(__30.x)), *(float2*)(&(v__38.x)));
    __28.x = exp2f(__29.x);
    __28.y = exp2f(__29.y);
    *(float2*)(acc_s + (i_27 * 2)) = __28;
  }
  #pragma unroll
  for (int i_28 = 0; i_28 < 2; ++i_28) {
    ssum[i_28] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_5 = 0; rv_5 < 8; ++rv_5) {
      ssum[i_28] = (ssum[i_28] + acc_s[((((rv_5 & 3) * 4) + (i_28 * 2)) + (rv_5 >> 2))]);
    }
    ssum[i_28] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(ssum[i_28]);
  }
  float2 __31;
    float2 v__39 = *(float2*)(lsum + 0);
    float2 v__40 = *(float2*)(sc + 0);
    float2 v__41 = *(float2*)(ssum + 0);
    *(float2*)(&(__31.x)) = tl::fma2(*(float2*)(&(v__39.x)), *(float2*)(&(v__40.x)), *(float2*)(&(v__41.x)));
  *(float2*)(lsum + 0) = __31;
  #pragma unroll
  for (int i_29 = 0; i_29 < 4; ++i_29) {
    uint2 __32;
    float4 v__42 = *(float4*)(acc_s + (i_29 * 4));
    ((half2*)(&__32))[0] = __float22half2_rn(((float2*)(&v__42))[0]);
    ((half2*)(&__32))[1] = __float22half2_rn(((float2*)(&v__42))[1]);
    *(uint2*)(acc_s_cast + (i_29 * 4)) = __32;
  }
  #pragma unroll
  for (int i_30 = 0; i_30 < 28; ++i_30) {
    float2 __33;
      float2 v__43 = *(float2*)(acc_o + (i_30 * 2));
      float2 v__44 = make_float2(sc[(i_30 & 1)], sc[(i_30 & 1)]);
      *(float2*)(&(__33.x)) = tl::mul2(*(float2*)(&(v__43.x)), *(float2*)(&(v__44.x)));
    *(float2*)(acc_o + (i_30 * 2)) = __33;
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    half_t B_local_5[56];
    for (int ki_5 = 0; ki_5 < 2; ++ki_5) {
      #pragma unroll
      for (int i_31 = 0; i_31 < 7; ++i_31) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)V_shared)[((((i_31 * 512) + (ki_5 * 256)) + ((((int)threadIdx.x) & 15) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(B_local_5[(i_31 * 8)])));
      }
      for (int j_5 = 0; j_5 < 7; ++j_5) {
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_5 * 8)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_5 * 8)), reinterpret_cast<const unsigned*>(B_local_5 + (j_5 * 8)));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_5 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_5 * 8)), reinterpret_cast<const unsigned*>(B_local_5 + ((j_5 * 8) + 4)));
      }
    }
  }
  #pragma unroll
  for (int i_32 = 0; i_32 < 28; ++i_32) {
    float2 __34;
      float2 v__45 = *(float2*)(acc_o + (i_32 * 2));
      float2 v__46 = make_float2(lsum[(i_32 & 1)], lsum[(i_32 & 1)]);
      __34.x = (v__45.x/v__46.x);
      __34.y = (v__45.y/v__46.y);
    *(float2*)(acc_o + (i_32 * 2)) = __34;
  }
  float broadcast_var_19 = 0x1.172feb597d322p-3f/*1.363219e-01*/;
  float2 __35;
    float2 v__47 = *(float2*)(m + 0);
    float2 v__48 = make_float2(broadcast_var_19, broadcast_var_19);
    float2 __36;
    float2 v__49 = *(float2*)(lsum + 0);
    __36.x = log2f(v__49.x);
    __36.y = log2f(v__49.y);
    *(float2*)(&(__35.x)) = tl::fma2(*(float2*)(&(v__47.x)), *(float2*)(&(v__48.x)), *(float2*)(&(__36.x)));
  *(float2*)(lsum + 0) = __35;
  if ((((int)threadIdx.x) % 4) == 0) {
    #pragma unroll
    for (int i_33 = 0; i_33 < 2; ++i_33) {
      if (((((((int)threadIdx.x) >> 5) * 16) + (i_33 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < 3) {
        glse[((((((((int)threadIdx.x) >> 5) * 64) + (i_33 * 32)) + (((int)blockIdx.y) * 12)) + (((((int)threadIdx.x) & 31) >> 2) * 4)) + ((int)blockIdx.z))] = ((half_t)lsum[i_33]);
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_34 = 0; i_34 < 28; ++i_34) {
    half_t O_shared_local_cast[2];
    if (((((((int)threadIdx.x) >> 5) * 16) + ((i_34 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < 3) {
      uint1 __37;
      float2 v__50 = *(float2*)(acc_o + (i_34 * 2));
      ((half2*)(&__37))[0] = __float22half2_rn(((float2*)(&v__50))[0]);
      *(uint1*)(O_shared_local_cast + 0) = __37;
      *(uint1*)(((half_t*)O_shared) + ((((((((int)threadIdx.x) >> 5) * 1792) + ((i_34 & 1) * 896)) + (((((int)threadIdx.x) & 31) >> 2) * 112)) + ((i_34 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(O_shared_local_cast + 0);
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_35 = 0; i_35 < 3; ++i_35) {
    if (((i_35 * 8) + (((int)threadIdx.x) >> 4)) < 21) {
      Opart[((((((int)blockIdx.y) * 1344) + ((((i_35 * 8) + (((int)threadIdx.x) >> 4)) / 7) * 448)) + (((int)blockIdx.z) * 112)) + (((i_35 * 128) + ((int)threadIdx.x)) % 112))] = ((half_t*)O_shared)[((i_35 * 128) + ((int)threadIdx.x))];
    }
  }
}

