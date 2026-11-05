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
  void* K_shared = ((void*)((char*)buf_dyn_shmem + 14336));
  float acc_o[224];
  float lsum[2];
  float m[2];
  float acc_s[32];
  float m_prev[2];
  float sc[2];
  half_t acc_s_cast[32];
  float ssum[2];
  float m_clear[2];
  float m_clear_1[2];
  #pragma unroll
  for (int i = 0; i < 7; ++i) {
    *(uint4*)(((half_t*)Q_shared) + ((((((i * 64) + (((int)threadIdx.x) >> 1)) % 7) * 1024) + ((((i * 64) + (((int)threadIdx.x) >> 1)) / 7) * 16)) + (((((((i * 16) + (((int)threadIdx.x) >> 3)) % 14) / 7) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(Q + ((((((int)blockIdx.x) * 50176) + ((((i * 64) + (((int)threadIdx.x) >> 1)) / 7) * 784)) + (((int)blockIdx.y) * 112)) + ((((i * 128) + ((int)threadIdx.x)) % 14) * 8)));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 56; ++i_1) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_o + (i_1 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(lsum + 0) = make_float2(broadcast_var_1, broadcast_var_1);
  float broadcast_var_2 = -CUDART_INF_F;
  *(float2*)(m + 0) = make_float2(broadcast_var_2, broadcast_var_2);
  __syncthreads();
  #pragma unroll
  for (int i_2 = 0; i_2 < 7; ++i_2) {
    tl::cp_async_gs<16>((&(((half_t*)K_shared)[((((((i_2 * 64) + (((int)threadIdx.x) >> 1)) % 7) * 1024) + ((((i_2 * 64) + (((int)threadIdx.x) >> 1)) / 7) * 16)) + (((((((i_2 * 16) + (((int)threadIdx.x) >> 3)) % 14) / 7) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(K[((((((i_2 * 64) + (((int)threadIdx.x) >> 1)) / 7) * 784) + (((int)blockIdx.y) * 112)) + ((((i_2 * 128) + ((int)threadIdx.x)) % 14) * 8))])));
  }
  tl::cp_async_commit();
  for (int k = 0; k < 34; ++k) {
    #pragma unroll
    for (int i_3 = 0; i_3 < 8; ++i_3) {
      float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_s + (i_3 * 4)) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
    }
    tl::cp_async_wait<0>();
    __syncthreads();
    {
      half_t A_local[8];
      half_t B_local[32];
      for (int ki = 0; ki < 7; ++ki) {
        tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[((((ki * 1024) + ((((int)threadIdx.x) >> 5) * 256)) + ((((int)threadIdx.x) & 15) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(A_local[0])));
        #pragma unroll
        for (int i_4 = 0; i_4 < 4; ++i_4) {
          tl::ptx_ldmatrix_x4((&(((half_t*)K_shared)[(((((ki * 1024) + (i_4 * 256)) + (((((int)threadIdx.x) & 31) >> 4) * 128)) + ((((int)threadIdx.x) & 7) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(B_local[(i_4 * 8)])));
        }
        for (int j = 0; j < 4; ++j) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (j * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_5 = 0; i_5 < 7; ++i_5) {
      tl::cp_async_gs<16>((&(((half_t*)K_shared)[((((((i_5 * 64) + (((int)threadIdx.x) >> 1)) % 7) * 1024) + ((((i_5 * 64) + (((int)threadIdx.x) >> 1)) / 7) * 16)) + (((((((i_5 * 16) + (((int)threadIdx.x) >> 3)) % 14) / 7) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(K[(((((k * 50176) + ((((i_5 * 64) + (((int)threadIdx.x) >> 1)) / 7) * 784)) + (((int)blockIdx.y) * 112)) + ((((i_5 * 128) + ((int)threadIdx.x)) % 14) * 8)) + 50176)])));
    }
    tl::cp_async_commit();
    *(float2*)(m_prev + 0) = *(float2*)(m + 0);
    float broadcast_var_4 = -CUDART_INF_F;
    *(float2*)(m + 0) = make_float2(broadcast_var_4, broadcast_var_4);
    #pragma unroll
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      m_clear[i_6] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 16; ++rv) {
        m_clear[i_6] = max(m_clear[i_6], acc_s[((((rv & 7) * 4) + (i_6 * 2)) + (rv >> 3))]);
      }
      m_clear[i_6] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(m_clear[i_6]);
      m[i_6] = max(m[i_6], m_clear[i_6]);
    }
    float broadcast_var_5 = 0x1.172feb597d322p-3f/*1.363219e-01*/;
    float broadcast_var_6 = 0x1.172feb597d322p-3f/*1.363219e-01*/;
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
    for (int i_7 = 0; i_7 < 56; ++i_7) {
      float4 __5;
        float4 v__4 = *(float4*)(acc_o + (i_7 * 4));
        float4 v__5 = make_float4(sc[(i_7 / 28)], sc[(i_7 / 28)], sc[(i_7 / 28)], sc[(i_7 / 28)]);
        *(float2*)(&(__5.x)) = tl::mul2(*(float2*)(&(v__4.x)), *(float2*)(&(v__5.x)));
        *(float2*)(&(__5.z)) = tl::mul2(*(float2*)(&(v__4.z)), *(float2*)(&(v__5.z)));
      *(float4*)(acc_o + (i_7 * 4)) = __5;
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 16; ++i_8) {
      float broadcast_var_7 = 0x1.172feb597d322p-3f/*1.363219e-01*/;
      float2 __6;
      float2 __7;
        float2 __8;
          float2 v__6 = *(float2*)(acc_s + (i_8 * 2));
          float2 v__7 = make_float2(broadcast_var_7, broadcast_var_7);
          *(float2*)(&(__8.x)) = tl::mul2(*(float2*)(&(v__6.x)), *(float2*)(&(v__7.x)));
        float2 v__8 = make_float2((m[(i_8 & 1)] * 0x1.172feb597d322p-3f/*1.363219e-01*/), (m[(i_8 & 1)] * 0x1.172feb597d322p-3f/*1.363219e-01*/));
        *(float2*)(&(__7.x)) = tl::sub2(*(float2*)(&(__8.x)), *(float2*)(&(v__8.x)));
      __6.x = exp2f(__7.x);
      __6.y = exp2f(__7.y);
      *(float2*)(acc_s + (i_8 * 2)) = __6;
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 8; ++i_9) {
      uint2 __9;
      float4 v__9 = *(float4*)(acc_s + (i_9 * 4));
      ((half2*)(&__9))[0] = __float22half2_rn(((float2*)(&v__9))[0]);
      ((half2*)(&__9))[1] = __float22half2_rn(((float2*)(&v__9))[1]);
      *(uint2*)(acc_s_cast + (i_9 * 4)) = __9;
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 2; ++i_10) {
      ssum[i_10] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
        ssum[i_10] = (ssum[i_10] + acc_s[((((rv_1 & 7) * 4) + (i_10 * 2)) + (rv_1 >> 3))]);
      }
      ssum[i_10] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(ssum[i_10]);
    }
    float2 __10;
      float2 v__10 = *(float2*)(lsum + 0);
      float2 v__11 = *(float2*)(sc + 0);
      float2 v__12 = *(float2*)(ssum + 0);
      *(float2*)(&(__10.x)) = tl::fma2(*(float2*)(&(v__10.x)), *(float2*)(&(v__11.x)), *(float2*)(&(v__12.x)));
    *(float2*)(lsum + 0) = __10;
  }
  #pragma unroll
  for (int i_11 = 0; i_11 < 8; ++i_11) {
    float broadcast_var_8 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_s + (i_11 * 4)) = make_float4(broadcast_var_8, broadcast_var_8, broadcast_var_8, broadcast_var_8);
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    half_t A_local_1[8];
    half_t B_local_1[32];
    for (int ki_1 = 0; ki_1 < 7; ++ki_1) {
      tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[((((ki_1 * 1024) + ((((int)threadIdx.x) >> 5) * 256)) + ((((int)threadIdx.x) & 15) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(A_local_1[0])));
      #pragma unroll
      for (int i_12 = 0; i_12 < 4; ++i_12) {
        tl::ptx_ldmatrix_x4((&(((half_t*)K_shared)[(((((ki_1 * 1024) + (i_12 * 256)) + (((((int)threadIdx.x) & 31) >> 4) * 128)) + ((((int)threadIdx.x) & 7) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(B_local_1[(i_12 * 8)])));
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
  for (int i_13 = 0; i_13 < 2; ++i_13) {
    m_clear_1[i_13] = -CUDART_INF_F;
    #pragma unroll
    for (int rv_2 = 0; rv_2 < 16; ++rv_2) {
      m_clear_1[i_13] = max(m_clear_1[i_13], acc_s[((((rv_2 & 7) * 4) + (i_13 * 2)) + (rv_2 >> 3))]);
    }
    m_clear_1[i_13] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(m_clear_1[i_13]);
    m[i_13] = max(m[i_13], m_clear_1[i_13]);
  }
  float broadcast_var_10 = 0x1.172feb597d322p-3f/*1.363219e-01*/;
  float broadcast_var_11 = 0x1.172feb597d322p-3f/*1.363219e-01*/;
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
  for (int i_14 = 0; i_14 < 56; ++i_14) {
    float4 __15;
      float4 v__17 = *(float4*)(acc_o + (i_14 * 4));
      float4 v__18 = make_float4(sc[(i_14 / 28)], sc[(i_14 / 28)], sc[(i_14 / 28)], sc[(i_14 / 28)]);
      *(float2*)(&(__15.x)) = tl::mul2(*(float2*)(&(v__17.x)), *(float2*)(&(v__18.x)));
      *(float2*)(&(__15.z)) = tl::mul2(*(float2*)(&(v__17.z)), *(float2*)(&(v__18.z)));
    *(float4*)(acc_o + (i_14 * 4)) = __15;
  }
  #pragma unroll
  for (int i_15 = 0; i_15 < 16; ++i_15) {
    float broadcast_var_12 = 0x1.172feb597d322p-3f/*1.363219e-01*/;
    float2 __16;
    float2 __17;
      float2 __18;
        float2 v__19 = *(float2*)(acc_s + (i_15 * 2));
        float2 v__20 = make_float2(broadcast_var_12, broadcast_var_12);
        *(float2*)(&(__18.x)) = tl::mul2(*(float2*)(&(v__19.x)), *(float2*)(&(v__20.x)));
      float2 v__21 = make_float2((m[(i_15 & 1)] * 0x1.172feb597d322p-3f/*1.363219e-01*/), (m[(i_15 & 1)] * 0x1.172feb597d322p-3f/*1.363219e-01*/));
      *(float2*)(&(__17.x)) = tl::sub2(*(float2*)(&(__18.x)), *(float2*)(&(v__21.x)));
    __16.x = exp2f(__17.x);
    __16.y = exp2f(__17.y);
    *(float2*)(acc_s + (i_15 * 2)) = __16;
  }
  #pragma unroll
  for (int i_16 = 0; i_16 < 8; ++i_16) {
    uint2 __19;
    float4 v__22 = *(float4*)(acc_s + (i_16 * 4));
    ((half2*)(&__19))[0] = __float22half2_rn(((float2*)(&v__22))[0]);
    ((half2*)(&__19))[1] = __float22half2_rn(((float2*)(&v__22))[1]);
    *(uint2*)(acc_s_cast + (i_16 * 4)) = __19;
  }
  #pragma unroll
  for (int i_17 = 0; i_17 < 2; ++i_17) {
    ssum[i_17] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_3 = 0; rv_3 < 16; ++rv_3) {
      ssum[i_17] = (ssum[i_17] + acc_s[((((rv_3 & 7) * 4) + (i_17 * 2)) + (rv_3 >> 3))]);
    }
    ssum[i_17] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(ssum[i_17]);
  }
  float2 __20;
    float2 v__23 = *(float2*)(lsum + 0);
    float2 v__24 = *(float2*)(sc + 0);
    float2 v__25 = *(float2*)(ssum + 0);
    *(float2*)(&(__20.x)) = tl::fma2(*(float2*)(&(v__23.x)), *(float2*)(&(v__24.x)), *(float2*)(&(v__25.x)));
  *(float2*)(lsum + 0) = __20;
  #pragma unroll
  for (int i_18 = 0; i_18 < 56; ++i_18) {
    float4 __21;
      float4 v__26 = *(float4*)(acc_o + (i_18 * 4));
      float4 v__27 = make_float4(lsum[(i_18 / 28)], lsum[(i_18 / 28)], lsum[(i_18 / 28)], lsum[(i_18 / 28)]);
      __21.x = (v__26.x/v__27.x);
      __21.y = (v__26.y/v__27.y);
      __21.z = (v__26.z/v__27.z);
      __21.w = (v__26.w/v__27.w);
    *(float4*)(acc_o + (i_18 * 4)) = __21;
  }
  __syncthreads();
  if ((((int)threadIdx.x) % 4) == 0) {
    #pragma unroll
    for (int i_19 = 0; i_19 < 28; ++i_19) {
      half_t O_shared_local_cast[8];
      for (int vec = 0; vec < 2; ++vec) {
        uint2 __22;
        float4 v__28 = *(float4*)(acc_o + ((i_19 * 8) + (vec * 4)));
        ((half2*)(&__22))[0] = __float22half2_rn(((float2*)(&v__28))[0]);
        ((half2*)(&__22))[1] = __float22half2_rn(((float2*)(&v__28))[1]);
        *(uint2*)(O_shared_local_cast + (vec * 4)) = __22;
      }
      *(uint4*)(((half_t*)O_shared) + (((((((int)threadIdx.x) >> 5) * 1792) + ((i_19 / 14) * 896)) + (((((int)threadIdx.x) & 31) >> 2) * 112)) + ((i_19 % 14) * 8))) = *(uint4*)(O_shared_local_cast + 0);
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_20 = 0; i_20 < 7; ++i_20) {
    *(uint4*)(Out + ((((((int)blockIdx.x) * 50176) + ((((i_20 * 64) + (((int)threadIdx.x) >> 1)) / 7) * 784)) + (((int)blockIdx.y) * 112)) + ((((i_20 * 128) + ((int)threadIdx.x)) % 14) * 8))) = *(uint4*)(((half_t*)O_shared) + ((i_20 * 1024) + (((int)threadIdx.x) * 8)));
  }
}

