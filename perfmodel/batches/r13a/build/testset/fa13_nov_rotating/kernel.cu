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

extern "C" __global__ void fa_kernel(const half_t* __restrict__ K, half_t* __restrict__ Out, const half_t* __restrict__ Q);
extern "C" __global__ void __launch_bounds__(128, 1) fa_kernel(const half_t* __restrict__ K, half_t* __restrict__ Out, const half_t* __restrict__ Q) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* O_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* Q_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* K_shared = ((void*)((char*)buf_dyn_shmem + 10240));
  float acc_o[160];
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
  #pragma unroll
  for (int i = 0; i < 5; ++i) {
    *(uint4*)(((half_t*)Q_shared) + ((((((i * 64) + (((int)threadIdx.x) >> 1)) % 5) * 1024) + ((((i * 64) + (((int)threadIdx.x) >> 1)) / 5) * 16)) + (((((((i * 16) + (((int)threadIdx.x) >> 3)) % 10) / 5) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(Q + ((((((int)blockIdx.x) * 51200) + ((((i * 64) + (((int)threadIdx.x) >> 1)) / 5) * 800)) + (((int)blockIdx.y) * 80)) + ((((i * 128) + ((int)threadIdx.x)) % 10) * 8)));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 40; ++i_1) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_o + (i_1 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(lsum + 0) = make_float2(broadcast_var_1, broadcast_var_1);
  float broadcast_var_2 = -CUDART_INF_F;
  *(float2*)(m + 0) = make_float2(broadcast_var_2, broadcast_var_2);
  __syncthreads();
  #pragma unroll
  for (int i_2 = 0; i_2 < 5; ++i_2) {
    tl::cp_async_gs<16>((&(((half_t*)K_shared)[((((((i_2 * 64) + (((int)threadIdx.x) >> 1)) % 5) * 1024) + ((((i_2 * 64) + (((int)threadIdx.x) >> 1)) / 5) * 16)) + (((((((i_2 * 16) + (((int)threadIdx.x) >> 3)) % 10) / 5) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(K[((((((i_2 * 64) + (((int)threadIdx.x) >> 1)) / 5) * 800) + (((int)blockIdx.y) * 80)) + ((((i_2 * 128) + ((int)threadIdx.x)) % 10) * 8))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_3 = 0; i_3 < 5; ++i_3) {
    tl::cp_async_gs<16>((&(((half_t*)K_shared)[(((((((i_3 * 64) + (((int)threadIdx.x) >> 1)) % 5) * 1024) + ((((i_3 * 64) + (((int)threadIdx.x) >> 1)) / 5) * 16)) + (((((((i_3 * 16) + (((int)threadIdx.x) >> 3)) % 10) / 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 5120)])), (&(K[(((((((i_3 * 64) + (((int)threadIdx.x) >> 1)) / 5) * 800) + (((int)blockIdx.y) * 80)) + ((((i_3 * 128) + ((int)threadIdx.x)) % 10) * 8)) + 51200)])));
  }
  tl::cp_async_commit();
  for (int k = 0; k < 26; ++k) {
    #pragma unroll
    for (int i_4 = 0; i_4 < 8; ++i_4) {
      float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_s + (i_4 * 4)) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
    }
    tl::cp_async_wait<1>();
    __syncthreads();
    {
      half_t A_local[8];
      half_t B_local[32];
      for (int ki = 0; ki < 5; ++ki) {
        tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[((((ki * 1024) + ((((int)threadIdx.x) >> 5) * 256)) + ((((int)threadIdx.x) & 15) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(A_local[0])));
        #pragma unroll
        for (int i_5 = 0; i_5 < 4; ++i_5) {
          tl::ptx_ldmatrix_x4((&(((half_t*)K_shared)[(((((((k & 1) * 5120) + (ki * 1024)) + (i_5 * 256)) + (((((int)threadIdx.x) & 31) >> 4) * 128)) + ((((int)threadIdx.x) & 7) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(B_local[(i_5 * 8)])));
        }
        for (int j = 0; j < 4; ++j) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (j * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_6 = 0; i_6 < 5; ++i_6) {
      tl::cp_async_gs<16>((&(((half_t*)K_shared)[(((((k & 1) * 5120) + ((((i_6 * 64) + (((int)threadIdx.x) >> 1)) % 5) * 1024)) + ((((i_6 * 64) + (((int)threadIdx.x) >> 1)) / 5) * 16)) + (((((((i_6 * 16) + (((int)threadIdx.x) >> 3)) % 10) / 5) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(K[(((((k * 51200) + ((((i_6 * 64) + (((int)threadIdx.x) >> 1)) / 5) * 800)) + (((int)blockIdx.y) * 80)) + ((((i_6 * 128) + ((int)threadIdx.x)) % 10) * 8)) + 102400)])));
    }
    tl::cp_async_commit();
    *(float2*)(m_prev + 0) = *(float2*)(m + 0);
    float broadcast_var_4 = -CUDART_INF_F;
    *(float2*)(m + 0) = make_float2(broadcast_var_4, broadcast_var_4);
    #pragma unroll
    for (int i_7 = 0; i_7 < 2; ++i_7) {
      m_clear[i_7] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 16; ++rv) {
        m_clear[i_7] = max(m_clear[i_7], acc_s[((((rv & 7) * 4) + (i_7 * 2)) + (rv >> 3))]);
      }
      m_clear[i_7] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(m_clear[i_7]);
      m[i_7] = max(m[i_7], m_clear[i_7]);
    }
    float broadcast_var_5 = 0x1.4a56b724e1e2p-3f/*1.612982e-01*/;
    float broadcast_var_6 = 0x1.4a56b724e1e2p-3f/*1.612982e-01*/;
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
    for (int i_8 = 0; i_8 < 40; ++i_8) {
      float4 __5;
        float4 v__4 = *(float4*)(acc_o + (i_8 * 4));
        float4 v__5 = make_float4(sc[(i_8 / 20)], sc[(i_8 / 20)], sc[(i_8 / 20)], sc[(i_8 / 20)]);
        *(float2*)(&(__5.x)) = tl::mul2(*(float2*)(&(v__4.x)), *(float2*)(&(v__5.x)));
        *(float2*)(&(__5.z)) = tl::mul2(*(float2*)(&(v__4.z)), *(float2*)(&(v__5.z)));
      *(float4*)(acc_o + (i_8 * 4)) = __5;
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 16; ++i_9) {
      float broadcast_var_7 = 0x1.4a56b724e1e2p-3f/*1.612982e-01*/;
      float2 __6;
      float2 __7;
        float2 __8;
          float2 v__6 = *(float2*)(acc_s + (i_9 * 2));
          float2 v__7 = make_float2(broadcast_var_7, broadcast_var_7);
          *(float2*)(&(__8.x)) = tl::mul2(*(float2*)(&(v__6.x)), *(float2*)(&(v__7.x)));
        float2 v__8 = make_float2((m[(i_9 & 1)] * 0x1.4a56b724e1e2p-3f/*1.612982e-01*/), (m[(i_9 & 1)] * 0x1.4a56b724e1e2p-3f/*1.612982e-01*/));
        *(float2*)(&(__7.x)) = tl::sub2(*(float2*)(&(__8.x)), *(float2*)(&(v__8.x)));
      __6.x = exp2f(__7.x);
      __6.y = exp2f(__7.y);
      *(float2*)(acc_s + (i_9 * 2)) = __6;
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 8; ++i_10) {
      uint2 __9;
      float4 v__9 = *(float4*)(acc_s + (i_10 * 4));
      ((half2*)(&__9))[0] = __float22half2_rn(((float2*)(&v__9))[0]);
      ((half2*)(&__9))[1] = __float22half2_rn(((float2*)(&v__9))[1]);
      *(uint2*)(acc_s_cast + (i_10 * 4)) = __9;
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
  for (int i_12 = 0; i_12 < 8; ++i_12) {
    float broadcast_var_8 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_s + (i_12 * 4)) = make_float4(broadcast_var_8, broadcast_var_8, broadcast_var_8, broadcast_var_8);
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  {
    half_t A_local_1[8];
    half_t B_local_1[32];
    for (int ki_1 = 0; ki_1 < 5; ++ki_1) {
      tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[((((ki_1 * 1024) + ((((int)threadIdx.x) >> 5) * 256)) + ((((int)threadIdx.x) & 15) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(A_local_1[0])));
      #pragma unroll
      for (int i_13 = 0; i_13 < 4; ++i_13) {
        tl::ptx_ldmatrix_x4((&(((half_t*)K_shared)[(((((ki_1 * 1024) + (i_13 * 256)) + (((((int)threadIdx.x) & 31) >> 4) * 128)) + ((((int)threadIdx.x) & 7) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(B_local_1[(i_13 * 8)])));
      }
      for (int j_1 = 0; j_1 < 4; ++j_1) {
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (j_1 * 8)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
      }
    }
  }
  *(float2*)(m_prev + 0) = *(float2*)(m + 0);
  float broadcast_var_9 = -CUDART_INF_F;
  *(float2*)(m + 0) = make_float2(broadcast_var_9, broadcast_var_9);
  #pragma unroll
  for (int i_14 = 0; i_14 < 2; ++i_14) {
    m_clear_1[i_14] = -CUDART_INF_F;
    #pragma unroll
    for (int rv_2 = 0; rv_2 < 16; ++rv_2) {
      m_clear_1[i_14] = max(m_clear_1[i_14], acc_s[((((rv_2 & 7) * 4) + (i_14 * 2)) + (rv_2 >> 3))]);
    }
    m_clear_1[i_14] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(m_clear_1[i_14]);
    m[i_14] = max(m[i_14], m_clear_1[i_14]);
  }
  float broadcast_var_10 = 0x1.4a56b724e1e2p-3f/*1.612982e-01*/;
  float broadcast_var_11 = 0x1.4a56b724e1e2p-3f/*1.612982e-01*/;
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
  for (int i_15 = 0; i_15 < 40; ++i_15) {
    float4 __15;
      float4 v__17 = *(float4*)(acc_o + (i_15 * 4));
      float4 v__18 = make_float4(sc[(i_15 / 20)], sc[(i_15 / 20)], sc[(i_15 / 20)], sc[(i_15 / 20)]);
      *(float2*)(&(__15.x)) = tl::mul2(*(float2*)(&(v__17.x)), *(float2*)(&(v__18.x)));
      *(float2*)(&(__15.z)) = tl::mul2(*(float2*)(&(v__17.z)), *(float2*)(&(v__18.z)));
    *(float4*)(acc_o + (i_15 * 4)) = __15;
  }
  #pragma unroll
  for (int i_16 = 0; i_16 < 16; ++i_16) {
    float broadcast_var_12 = 0x1.4a56b724e1e2p-3f/*1.612982e-01*/;
    float2 __16;
    float2 __17;
      float2 __18;
        float2 v__19 = *(float2*)(acc_s + (i_16 * 2));
        float2 v__20 = make_float2(broadcast_var_12, broadcast_var_12);
        *(float2*)(&(__18.x)) = tl::mul2(*(float2*)(&(v__19.x)), *(float2*)(&(v__20.x)));
      float2 v__21 = make_float2((m[(i_16 & 1)] * 0x1.4a56b724e1e2p-3f/*1.612982e-01*/), (m[(i_16 & 1)] * 0x1.4a56b724e1e2p-3f/*1.612982e-01*/));
      *(float2*)(&(__17.x)) = tl::sub2(*(float2*)(&(__18.x)), *(float2*)(&(v__21.x)));
    __16.x = exp2f(__17.x);
    __16.y = exp2f(__17.y);
    *(float2*)(acc_s + (i_16 * 2)) = __16;
  }
  #pragma unroll
  for (int i_17 = 0; i_17 < 8; ++i_17) {
    uint2 __19;
    float4 v__22 = *(float4*)(acc_s + (i_17 * 4));
    ((half2*)(&__19))[0] = __float22half2_rn(((float2*)(&v__22))[0]);
    ((half2*)(&__19))[1] = __float22half2_rn(((float2*)(&v__22))[1]);
    *(uint2*)(acc_s_cast + (i_17 * 4)) = __19;
  }
  #pragma unroll
  for (int i_18 = 0; i_18 < 2; ++i_18) {
    ssum[i_18] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_3 = 0; rv_3 < 16; ++rv_3) {
      ssum[i_18] = (ssum[i_18] + acc_s[((((rv_3 & 7) * 4) + (i_18 * 2)) + (rv_3 >> 3))]);
    }
    ssum[i_18] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(ssum[i_18]);
  }
  float2 __20;
    float2 v__23 = *(float2*)(lsum + 0);
    float2 v__24 = *(float2*)(sc + 0);
    float2 v__25 = *(float2*)(ssum + 0);
    *(float2*)(&(__20.x)) = tl::fma2(*(float2*)(&(v__23.x)), *(float2*)(&(v__24.x)), *(float2*)(&(v__25.x)));
  *(float2*)(lsum + 0) = __20;
  #pragma unroll
  for (int i_19 = 0; i_19 < 8; ++i_19) {
    float broadcast_var_13 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_s + (i_19 * 4)) = make_float4(broadcast_var_13, broadcast_var_13, broadcast_var_13, broadcast_var_13);
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    half_t A_local_2[8];
    half_t B_local_2[32];
    for (int ki_2 = 0; ki_2 < 5; ++ki_2) {
      tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[((((ki_2 * 1024) + ((((int)threadIdx.x) >> 5) * 256)) + ((((int)threadIdx.x) & 15) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(A_local_2[0])));
      #pragma unroll
      for (int i_20 = 0; i_20 < 4; ++i_20) {
        tl::ptx_ldmatrix_x4((&(((half_t*)K_shared)[((((((ki_2 * 1024) + (i_20 * 256)) + (((((int)threadIdx.x) & 31) >> 4) * 128)) + ((((int)threadIdx.x) & 7) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 5120)])), (&(B_local_2[(i_20 * 8)])));
      }
      for (int j_2 = 0; j_2 < 4; ++j_2) {
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (j_2 * 8)), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_2 + (j_2 * 8)));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((j_2 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_2 + ((j_2 * 8) + 4)));
      }
    }
  }
  *(float2*)(m_prev + 0) = *(float2*)(m + 0);
  float broadcast_var_14 = -CUDART_INF_F;
  *(float2*)(m + 0) = make_float2(broadcast_var_14, broadcast_var_14);
  #pragma unroll
  for (int i_21 = 0; i_21 < 2; ++i_21) {
    m_clear_2[i_21] = -CUDART_INF_F;
    #pragma unroll
    for (int rv_4 = 0; rv_4 < 16; ++rv_4) {
      m_clear_2[i_21] = max(m_clear_2[i_21], acc_s[((((rv_4 & 7) * 4) + (i_21 * 2)) + (rv_4 >> 3))]);
    }
    m_clear_2[i_21] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(m_clear_2[i_21]);
    m[i_21] = max(m[i_21], m_clear_2[i_21]);
  }
  float broadcast_var_15 = 0x1.4a56b724e1e2p-3f/*1.612982e-01*/;
  float broadcast_var_16 = 0x1.4a56b724e1e2p-3f/*1.612982e-01*/;
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
  for (int i_22 = 0; i_22 < 40; ++i_22) {
    float4 __25;
      float4 v__30 = *(float4*)(acc_o + (i_22 * 4));
      float4 v__31 = make_float4(sc[(i_22 / 20)], sc[(i_22 / 20)], sc[(i_22 / 20)], sc[(i_22 / 20)]);
      *(float2*)(&(__25.x)) = tl::mul2(*(float2*)(&(v__30.x)), *(float2*)(&(v__31.x)));
      *(float2*)(&(__25.z)) = tl::mul2(*(float2*)(&(v__30.z)), *(float2*)(&(v__31.z)));
    *(float4*)(acc_o + (i_22 * 4)) = __25;
  }
  #pragma unroll
  for (int i_23 = 0; i_23 < 16; ++i_23) {
    float broadcast_var_17 = 0x1.4a56b724e1e2p-3f/*1.612982e-01*/;
    float2 __26;
    float2 __27;
      float2 __28;
        float2 v__32 = *(float2*)(acc_s + (i_23 * 2));
        float2 v__33 = make_float2(broadcast_var_17, broadcast_var_17);
        *(float2*)(&(__28.x)) = tl::mul2(*(float2*)(&(v__32.x)), *(float2*)(&(v__33.x)));
      float2 v__34 = make_float2((m[(i_23 & 1)] * 0x1.4a56b724e1e2p-3f/*1.612982e-01*/), (m[(i_23 & 1)] * 0x1.4a56b724e1e2p-3f/*1.612982e-01*/));
      *(float2*)(&(__27.x)) = tl::sub2(*(float2*)(&(__28.x)), *(float2*)(&(v__34.x)));
    __26.x = exp2f(__27.x);
    __26.y = exp2f(__27.y);
    *(float2*)(acc_s + (i_23 * 2)) = __26;
  }
  #pragma unroll
  for (int i_24 = 0; i_24 < 8; ++i_24) {
    uint2 __29;
    float4 v__35 = *(float4*)(acc_s + (i_24 * 4));
    ((half2*)(&__29))[0] = __float22half2_rn(((float2*)(&v__35))[0]);
    ((half2*)(&__29))[1] = __float22half2_rn(((float2*)(&v__35))[1]);
    *(uint2*)(acc_s_cast + (i_24 * 4)) = __29;
  }
  #pragma unroll
  for (int i_25 = 0; i_25 < 2; ++i_25) {
    ssum[i_25] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_5 = 0; rv_5 < 16; ++rv_5) {
      ssum[i_25] = (ssum[i_25] + acc_s[((((rv_5 & 7) * 4) + (i_25 * 2)) + (rv_5 >> 3))]);
    }
    ssum[i_25] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(ssum[i_25]);
  }
  float2 __30;
    float2 v__36 = *(float2*)(lsum + 0);
    float2 v__37 = *(float2*)(sc + 0);
    float2 v__38 = *(float2*)(ssum + 0);
    *(float2*)(&(__30.x)) = tl::fma2(*(float2*)(&(v__36.x)), *(float2*)(&(v__37.x)), *(float2*)(&(v__38.x)));
  *(float2*)(lsum + 0) = __30;
  #pragma unroll
  for (int i_26 = 0; i_26 < 40; ++i_26) {
    float4 __31;
      float4 v__39 = *(float4*)(acc_o + (i_26 * 4));
      float4 v__40 = make_float4(lsum[(i_26 / 20)], lsum[(i_26 / 20)], lsum[(i_26 / 20)], lsum[(i_26 / 20)]);
      __31.x = (v__39.x/v__40.x);
      __31.y = (v__39.y/v__40.y);
      __31.z = (v__39.z/v__40.z);
      __31.w = (v__39.w/v__40.w);
    *(float4*)(acc_o + (i_26 * 4)) = __31;
  }
  __syncthreads();
  if ((((int)threadIdx.x) % 4) == 0) {
    #pragma unroll
    for (int i_27 = 0; i_27 < 20; ++i_27) {
      half_t O_shared_local_cast[8];
      for (int vec = 0; vec < 2; ++vec) {
        uint2 __32;
        float4 v__41 = *(float4*)(acc_o + ((i_27 * 8) + (vec * 4)));
        ((half2*)(&__32))[0] = __float22half2_rn(((float2*)(&v__41))[0]);
        ((half2*)(&__32))[1] = __float22half2_rn(((float2*)(&v__41))[1]);
        *(uint2*)(O_shared_local_cast + (vec * 4)) = __32;
      }
      *(uint4*)(((half_t*)O_shared) + (((((((int)threadIdx.x) >> 5) * 1280) + ((i_27 / 10) * 640)) + (((((int)threadIdx.x) & 31) >> 2) * 80)) + ((i_27 % 10) * 8))) = *(uint4*)(O_shared_local_cast + 0);
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_28 = 0; i_28 < 5; ++i_28) {
    *(uint4*)(Out + ((((((int)blockIdx.x) * 51200) + ((((i_28 * 64) + (((int)threadIdx.x) >> 1)) / 5) * 800)) + (((int)blockIdx.y) * 80)) + ((((i_28 * 128) + ((int)threadIdx.x)) % 10) * 8))) = *(uint4*)(((half_t*)O_shared) + ((i_28 * 1024) + (((int)threadIdx.x) * 8)));
  }
}

