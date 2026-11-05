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
  void* K_shared = ((void*)((char*)buf_dyn_shmem + 18432));
  void* V_shared = ((void*)((char*)buf_dyn_shmem + 92160));
  float acc_o[72];
  float lsum[2];
  float m[2];
  float acc_s[64];
  float m_prev[2];
  float sc[2];
  half_t acc_s_cast[64];
  float ssum[2];
  float m_clear[2];
  float m_clear_1[2];
  float m_clear_2[2];
  #pragma unroll
  for (int i = 0; i < 9; ++i) {
    half_t broadcast_var = half_t(0x0p+0f/*0.000000e+00*/);
    uint4 condval;
    if ((((((int)blockIdx.x) * 2) + (((i * 64) + (((int)threadIdx.x) >> 1)) / 288)) < 63)) {
      condval = *(uint4*)(Q + ((((((int)blockIdx.x) * 82944) + ((((i * 64) + (((int)threadIdx.x) >> 1)) / 9) * 1296)) + (((int)blockIdx.y) * 144)) + ((((i * 128) + ((int)threadIdx.x)) % 18) * 8)));
    } else {
      condval = make_uint4(__pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var));
    }
    *(uint4*)(((half_t*)Q_shared) + ((((((i * 64) + (((int)threadIdx.x) >> 1)) % 9) * 1024) + ((((i * 64) + (((int)threadIdx.x) >> 1)) / 9) * 16)) + (((((((i * 16) + (((int)threadIdx.x) >> 3)) % 18) / 9) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval;
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 18; ++i_1) {
    float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_o + (i_1 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
  }
  float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(lsum + 0) = make_float2(broadcast_var_2, broadcast_var_2);
  float broadcast_var_3 = -CUDART_INF_F;
  *(float2*)(m + 0) = make_float2(broadcast_var_3, broadcast_var_3);
  __syncthreads();
  #pragma unroll
  for (int i_2 = 0; i_2 < 18; ++i_2) {
    tl::cp_async_gs<16>((&(((half_t*)K_shared)[((((((i_2 * 64) + (((int)threadIdx.x) >> 1)) % 9) * 2048) + ((((i_2 * 64) + (((int)threadIdx.x) >> 1)) / 9) * 16)) + (((((((i_2 * 16) + (((int)threadIdx.x) >> 3)) % 18) / 9) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(K[((((((i_2 * 64) + (((int)threadIdx.x) >> 1)) / 9) * 1296) + (((int)blockIdx.y) * 144)) + ((((i_2 * 128) + ((int)threadIdx.x)) % 18) * 8))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_3 = 0; i_3 < 18; ++i_3) {
    tl::cp_async_gs<16>((&(((half_t*)V_shared)[((((((i_3 * 64) + (((int)threadIdx.x) >> 1)) % 9) * 2048) + ((((i_3 * 64) + (((int)threadIdx.x) >> 1)) / 9) * 16)) + (((((((i_3 * 16) + (((int)threadIdx.x) >> 3)) % 18) / 9) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(V[((((((i_3 * 64) + (((int)threadIdx.x) >> 1)) / 9) * 1296) + (((int)blockIdx.y) * 144)) + ((((i_3 * 128) + ((int)threadIdx.x)) % 18) * 8))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_4 = 0; i_4 < 18; ++i_4) {
    tl::cp_async_gs<16>((&(((half_t*)K_shared)[(((((((i_4 * 64) + (((int)threadIdx.x) >> 1)) % 9) * 2048) + ((((i_4 * 64) + (((int)threadIdx.x) >> 1)) / 9) * 16)) + (((((((i_4 * 16) + (((int)threadIdx.x) >> 3)) % 18) / 9) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 18432)])), (&(K[(((((((i_4 * 64) + (((int)threadIdx.x) >> 1)) / 9) * 1296) + (((int)blockIdx.y) * 144)) + ((((i_4 * 128) + ((int)threadIdx.x)) % 18) * 8)) + 165888)])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_5 = 0; i_5 < 18; ++i_5) {
    tl::cp_async_gs<16>((&(((half_t*)V_shared)[(((((((i_5 * 64) + (((int)threadIdx.x) >> 1)) % 9) * 2048) + ((((i_5 * 64) + (((int)threadIdx.x) >> 1)) / 9) * 16)) + (((((((i_5 * 16) + (((int)threadIdx.x) >> 3)) % 18) / 9) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 18432)])), (&(V[(((((((i_5 * 64) + (((int)threadIdx.x) >> 1)) / 9) * 1296) + (((int)blockIdx.y) * 144)) + ((((i_5 * 128) + ((int)threadIdx.x)) % 18) * 8)) + 165888)])));
  }
  tl::cp_async_commit();
  for (int k = 0; k < 14; ++k) {
    #pragma unroll
    for (int i_6 = 0; i_6 < 16; ++i_6) {
      float broadcast_var_4 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_s + (i_6 * 4)) = make_float4(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4);
    }
    tl::cp_async_wait<3>();
    __syncthreads();
    {
      half_t A_local[8];
      half_t B_local[64];
      for (int ki = 0; ki < 9; ++ki) {
        tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[((((ki * 1024) + ((((int)threadIdx.x) >> 5) * 256)) + ((((int)threadIdx.x) & 15) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(A_local[0])));
        #pragma unroll
        for (int i_7 = 0; i_7 < 8; ++i_7) {
          tl::ptx_ldmatrix_x4((&(((half_t*)K_shared)[(((((((k & 1) * 18432) + (ki * 2048)) + (i_7 * 256)) + (((((int)threadIdx.x) & 31) >> 4) * 128)) + ((((int)threadIdx.x) & 7) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(B_local[(i_7 * 8)])));
        }
        for (int j = 0; j < 8; ++j) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (j * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_8 = 0; i_8 < 18; ++i_8) {
      tl::cp_async_gs_conditional<16>((&(((half_t*)K_shared)[(((((k & 1) * 18432) + ((((i_8 * 64) + (((int)threadIdx.x) >> 1)) % 9) * 2048)) + ((((i_8 * 64) + (((int)threadIdx.x) >> 1)) / 9) * 16)) + (((((((i_8 * 16) + (((int)threadIdx.x) >> 3)) % 18) / 9) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(K[(((((k * 165888) + ((((i_8 * 64) + (((int)threadIdx.x) >> 1)) / 9) * 1296)) + (((int)blockIdx.y) * 144)) + ((((i_8 * 128) + ((int)threadIdx.x)) % 18) * 8)) + 331776)])), ((((k * 4) + (((i_8 * 64) + (((int)threadIdx.x) >> 1)) / 288)) < 55) && (((k * 4) + (((i_8 * 64) + (((int)threadIdx.x) >> 1)) / 288)) < 55)));
    }
    tl::cp_async_commit();
    *(float2*)(m_prev + 0) = *(float2*)(m + 0);
    float broadcast_var_5 = -CUDART_INF_F;
    *(float2*)(m + 0) = make_float2(broadcast_var_5, broadcast_var_5);
    #pragma unroll
    for (int i_9 = 0; i_9 < 2; ++i_9) {
      m_clear[i_9] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 32; ++rv) {
        m_clear[i_9] = max(m_clear[i_9], acc_s[((((rv & 15) * 4) + (i_9 * 2)) + (rv >> 4))]);
      }
      m_clear[i_9] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(m_clear[i_9]);
      m[i_9] = max(m[i_9], m_clear[i_9]);
    }
    float broadcast_var_6 = 0x1.ec709dbe8903ep-4f/*1.202246e-01*/;
    float broadcast_var_7 = 0x1.ec709dbe8903ep-4f/*1.202246e-01*/;
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
    for (int i_10 = 0; i_10 < 36; ++i_10) {
      float2 __5;
        float2 v__4 = *(float2*)(acc_o + (i_10 * 2));
        float2 v__5 = make_float2(sc[(i_10 & 1)], sc[(i_10 & 1)]);
        *(float2*)(&(__5.x)) = tl::mul2(*(float2*)(&(v__4.x)), *(float2*)(&(v__5.x)));
      *(float2*)(acc_o + (i_10 * 2)) = __5;
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 32; ++i_11) {
      float broadcast_var_8 = 0x1.ec709dbe8903ep-4f/*1.202246e-01*/;
      float2 __6;
      float2 __7;
        float2 __8;
          float2 v__6 = *(float2*)(acc_s + (i_11 * 2));
          float2 v__7 = make_float2(broadcast_var_8, broadcast_var_8);
          *(float2*)(&(__8.x)) = tl::mul2(*(float2*)(&(v__6.x)), *(float2*)(&(v__7.x)));
        float2 v__8 = make_float2((m[(i_11 & 1)] * 0x1.ec709dbe8903ep-4f/*1.202246e-01*/), (m[(i_11 & 1)] * 0x1.ec709dbe8903ep-4f/*1.202246e-01*/));
        *(float2*)(&(__7.x)) = tl::sub2(*(float2*)(&(__8.x)), *(float2*)(&(v__8.x)));
      __6.x = exp2f(__7.x);
      __6.y = exp2f(__7.y);
      *(float2*)(acc_s + (i_11 * 2)) = __6;
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 16; ++i_12) {
      uint2 __9;
      float4 v__9 = *(float4*)(acc_s + (i_12 * 4));
      ((half2*)(&__9))[0] = __float22half2_rn(((float2*)(&v__9))[0]);
      ((half2*)(&__9))[1] = __float22half2_rn(((float2*)(&v__9))[1]);
      *(uint2*)(acc_s_cast + (i_12 * 4)) = __9;
    }
    tl::cp_async_wait<3>();
    __syncthreads();
    {
      half_t B_local_1[72];
      for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
        #pragma unroll
        for (int i_13 = 0; i_13 < 9; ++i_13) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)V_shared)[((((((k & 1) * 18432) + (i_13 * 2048)) + (ki_1 * 256)) + ((((int)threadIdx.x) & 15) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(B_local_1[(i_13 * 8)])));
        }
        for (int j_1 = 0; j_1 < 9; ++j_1) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_1 * 8)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_1 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_1 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_14 = 0; i_14 < 18; ++i_14) {
      tl::cp_async_gs_conditional<16>((&(((half_t*)V_shared)[(((((k & 1) * 18432) + ((((i_14 * 64) + (((int)threadIdx.x) >> 1)) % 9) * 2048)) + ((((i_14 * 64) + (((int)threadIdx.x) >> 1)) / 9) * 16)) + (((((((i_14 * 16) + (((int)threadIdx.x) >> 3)) % 18) / 9) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(V[(((((k * 165888) + ((((i_14 * 64) + (((int)threadIdx.x) >> 1)) / 9) * 1296)) + (((int)blockIdx.y) * 144)) + ((((i_14 * 128) + ((int)threadIdx.x)) % 18) * 8)) + 331776)])), ((((k * 4) + (((i_14 * 64) + (((int)threadIdx.x) >> 1)) / 288)) < 55) && (((k * 4) + (((i_14 * 64) + (((int)threadIdx.x) >> 1)) / 288)) < 55)));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_15 = 0; i_15 < 2; ++i_15) {
      ssum[i_15] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 32; ++rv_1) {
        ssum[i_15] = (ssum[i_15] + acc_s[((((rv_1 & 15) * 4) + (i_15 * 2)) + (rv_1 >> 4))]);
      }
      ssum[i_15] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(ssum[i_15]);
    }
    float2 __10;
      float2 v__10 = *(float2*)(lsum + 0);
      float2 v__11 = *(float2*)(sc + 0);
      float2 v__12 = *(float2*)(ssum + 0);
      *(float2*)(&(__10.x)) = tl::fma2(*(float2*)(&(v__10.x)), *(float2*)(&(v__11.x)), *(float2*)(&(v__12.x)));
    *(float2*)(lsum + 0) = __10;
  }
  #pragma unroll
  for (int i_16 = 0; i_16 < 16; ++i_16) {
    float broadcast_var_9 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_s + (i_16 * 4)) = make_float4(broadcast_var_9, broadcast_var_9, broadcast_var_9, broadcast_var_9);
  }
  tl::cp_async_wait<3>();
  __syncthreads();
  {
    half_t A_local_1[8];
    half_t B_local_2[64];
    for (int ki_2 = 0; ki_2 < 9; ++ki_2) {
      tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[((((ki_2 * 1024) + ((((int)threadIdx.x) >> 5) * 256)) + ((((int)threadIdx.x) & 15) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(A_local_1[0])));
      #pragma unroll
      for (int i_17 = 0; i_17 < 8; ++i_17) {
        tl::ptx_ldmatrix_x4((&(((half_t*)K_shared)[(((((ki_2 * 2048) + (i_17 * 256)) + (((((int)threadIdx.x) & 31) >> 4) * 128)) + ((((int)threadIdx.x) & 7) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(B_local_2[(i_17 * 8)])));
      }
      for (int j_2 = 0; j_2 < 8; ++j_2) {
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
    for (int rv_2 = 0; rv_2 < 32; ++rv_2) {
      m_clear_1[i_18] = max(m_clear_1[i_18], acc_s[((((rv_2 & 15) * 4) + (i_18 * 2)) + (rv_2 >> 4))]);
    }
    m_clear_1[i_18] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(m_clear_1[i_18]);
    m[i_18] = max(m[i_18], m_clear_1[i_18]);
  }
  float broadcast_var_11 = 0x1.ec709dbe8903ep-4f/*1.202246e-01*/;
  float broadcast_var_12 = 0x1.ec709dbe8903ep-4f/*1.202246e-01*/;
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
  for (int i_19 = 0; i_19 < 36; ++i_19) {
    float2 __15;
      float2 v__17 = *(float2*)(acc_o + (i_19 * 2));
      float2 v__18 = make_float2(sc[(i_19 & 1)], sc[(i_19 & 1)]);
      *(float2*)(&(__15.x)) = tl::mul2(*(float2*)(&(v__17.x)), *(float2*)(&(v__18.x)));
    *(float2*)(acc_o + (i_19 * 2)) = __15;
  }
  #pragma unroll
  for (int i_20 = 0; i_20 < 32; ++i_20) {
    float broadcast_var_13 = 0x1.ec709dbe8903ep-4f/*1.202246e-01*/;
    float2 __16;
    float2 __17;
      float2 __18;
        float2 v__19 = *(float2*)(acc_s + (i_20 * 2));
        float2 v__20 = make_float2(broadcast_var_13, broadcast_var_13);
        *(float2*)(&(__18.x)) = tl::mul2(*(float2*)(&(v__19.x)), *(float2*)(&(v__20.x)));
      float2 v__21 = make_float2((m[(i_20 & 1)] * 0x1.ec709dbe8903ep-4f/*1.202246e-01*/), (m[(i_20 & 1)] * 0x1.ec709dbe8903ep-4f/*1.202246e-01*/));
      *(float2*)(&(__17.x)) = tl::sub2(*(float2*)(&(__18.x)), *(float2*)(&(v__21.x)));
    __16.x = exp2f(__17.x);
    __16.y = exp2f(__17.y);
    *(float2*)(acc_s + (i_20 * 2)) = __16;
  }
  #pragma unroll
  for (int i_21 = 0; i_21 < 16; ++i_21) {
    uint2 __19;
    float4 v__22 = *(float4*)(acc_s + (i_21 * 4));
    ((half2*)(&__19))[0] = __float22half2_rn(((float2*)(&v__22))[0]);
    ((half2*)(&__19))[1] = __float22half2_rn(((float2*)(&v__22))[1]);
    *(uint2*)(acc_s_cast + (i_21 * 4)) = __19;
  }
  tl::cp_async_wait<2>();
  __syncthreads();
  {
    half_t B_local_3[72];
    for (int ki_3 = 0; ki_3 < 8; ++ki_3) {
      #pragma unroll
      for (int i_22 = 0; i_22 < 9; ++i_22) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)V_shared)[((((i_22 * 2048) + (ki_3 * 256)) + ((((int)threadIdx.x) & 15) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(B_local_3[(i_22 * 8)])));
      }
      for (int j_3 = 0; j_3 < 9; ++j_3) {
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_3 * 8)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_3 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + (j_3 * 8)));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_3 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_3 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + ((j_3 * 8) + 4)));
      }
    }
  }
  #pragma unroll
  for (int i_23 = 0; i_23 < 2; ++i_23) {
    ssum[i_23] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_3 = 0; rv_3 < 32; ++rv_3) {
      ssum[i_23] = (ssum[i_23] + acc_s[((((rv_3 & 15) * 4) + (i_23 * 2)) + (rv_3 >> 4))]);
    }
    ssum[i_23] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(ssum[i_23]);
  }
  float2 __20;
    float2 v__23 = *(float2*)(lsum + 0);
    float2 v__24 = *(float2*)(sc + 0);
    float2 v__25 = *(float2*)(ssum + 0);
    *(float2*)(&(__20.x)) = tl::fma2(*(float2*)(&(v__23.x)), *(float2*)(&(v__24.x)), *(float2*)(&(v__25.x)));
  *(float2*)(lsum + 0) = __20;
  #pragma unroll
  for (int i_24 = 0; i_24 < 16; ++i_24) {
    float broadcast_var_14 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_s + (i_24 * 4)) = make_float4(broadcast_var_14, broadcast_var_14, broadcast_var_14, broadcast_var_14);
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  {
    half_t A_local_2[8];
    half_t B_local_4[64];
    for (int ki_4 = 0; ki_4 < 9; ++ki_4) {
      tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[((((ki_4 * 1024) + ((((int)threadIdx.x) >> 5) * 256)) + ((((int)threadIdx.x) & 15) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8))])), (&(A_local_2[0])));
      #pragma unroll
      for (int i_25 = 0; i_25 < 8; ++i_25) {
        tl::ptx_ldmatrix_x4((&(((half_t*)K_shared)[((((((ki_4 * 2048) + (i_25 * 256)) + (((((int)threadIdx.x) & 31) >> 4) * 128)) + ((((int)threadIdx.x) & 7) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 18432)])), (&(B_local_4[(i_25 * 8)])));
      }
      for (int j_4 = 0; j_4 < 8; ++j_4) {
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
    for (int rv_4 = 0; rv_4 < 32; ++rv_4) {
      m_clear_2[i_26] = max(m_clear_2[i_26], acc_s[((((rv_4 & 15) * 4) + (i_26 * 2)) + (rv_4 >> 4))]);
    }
    m_clear_2[i_26] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(m_clear_2[i_26]);
    m[i_26] = max(m[i_26], m_clear_2[i_26]);
  }
  float broadcast_var_16 = 0x1.ec709dbe8903ep-4f/*1.202246e-01*/;
  float broadcast_var_17 = 0x1.ec709dbe8903ep-4f/*1.202246e-01*/;
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
  for (int i_27 = 0; i_27 < 36; ++i_27) {
    float2 __25;
      float2 v__30 = *(float2*)(acc_o + (i_27 * 2));
      float2 v__31 = make_float2(sc[(i_27 & 1)], sc[(i_27 & 1)]);
      *(float2*)(&(__25.x)) = tl::mul2(*(float2*)(&(v__30.x)), *(float2*)(&(v__31.x)));
    *(float2*)(acc_o + (i_27 * 2)) = __25;
  }
  #pragma unroll
  for (int i_28 = 0; i_28 < 32; ++i_28) {
    float broadcast_var_18 = 0x1.ec709dbe8903ep-4f/*1.202246e-01*/;
    float2 __26;
    float2 __27;
      float2 __28;
        float2 v__32 = *(float2*)(acc_s + (i_28 * 2));
        float2 v__33 = make_float2(broadcast_var_18, broadcast_var_18);
        *(float2*)(&(__28.x)) = tl::mul2(*(float2*)(&(v__32.x)), *(float2*)(&(v__33.x)));
      float2 v__34 = make_float2((m[(i_28 & 1)] * 0x1.ec709dbe8903ep-4f/*1.202246e-01*/), (m[(i_28 & 1)] * 0x1.ec709dbe8903ep-4f/*1.202246e-01*/));
      *(float2*)(&(__27.x)) = tl::sub2(*(float2*)(&(__28.x)), *(float2*)(&(v__34.x)));
    __26.x = exp2f(__27.x);
    __26.y = exp2f(__27.y);
    *(float2*)(acc_s + (i_28 * 2)) = __26;
  }
  #pragma unroll
  for (int i_29 = 0; i_29 < 16; ++i_29) {
    uint2 __29;
    float4 v__35 = *(float4*)(acc_s + (i_29 * 4));
    ((half2*)(&__29))[0] = __float22half2_rn(((float2*)(&v__35))[0]);
    ((half2*)(&__29))[1] = __float22half2_rn(((float2*)(&v__35))[1]);
    *(uint2*)(acc_s_cast + (i_29 * 4)) = __29;
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    half_t B_local_5[72];
    for (int ki_5 = 0; ki_5 < 8; ++ki_5) {
      #pragma unroll
      for (int i_30 = 0; i_30 < 9; ++i_30) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)V_shared)[(((((i_30 * 2048) + (ki_5 * 256)) + ((((int)threadIdx.x) & 15) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + 18432)])), (&(B_local_5[(i_30 * 8)])));
      }
      for (int j_5 = 0; j_5 < 9; ++j_5) {
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_5 * 8)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_5 * 8)), reinterpret_cast<const unsigned*>(B_local_5 + (j_5 * 8)));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_5 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_5 * 8)), reinterpret_cast<const unsigned*>(B_local_5 + ((j_5 * 8) + 4)));
      }
    }
  }
  #pragma unroll
  for (int i_31 = 0; i_31 < 2; ++i_31) {
    ssum[i_31] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_5 = 0; rv_5 < 32; ++rv_5) {
      ssum[i_31] = (ssum[i_31] + acc_s[((((rv_5 & 15) * 4) + (i_31 * 2)) + (rv_5 >> 4))]);
    }
    ssum[i_31] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(ssum[i_31]);
  }
  float2 __30;
    float2 v__36 = *(float2*)(lsum + 0);
    float2 v__37 = *(float2*)(sc + 0);
    float2 v__38 = *(float2*)(ssum + 0);
    *(float2*)(&(__30.x)) = tl::fma2(*(float2*)(&(v__36.x)), *(float2*)(&(v__37.x)), *(float2*)(&(v__38.x)));
  *(float2*)(lsum + 0) = __30;
  #pragma unroll
  for (int i_32 = 0; i_32 < 36; ++i_32) {
    float2 __31;
      float2 v__39 = *(float2*)(acc_o + (i_32 * 2));
      float2 v__40 = make_float2(lsum[(i_32 & 1)], lsum[(i_32 & 1)]);
      __31.x = (v__39.x/v__40.x);
      __31.y = (v__39.y/v__40.y);
    *(float2*)(acc_o + (i_32 * 2)) = __31;
  }
  __syncthreads();
  #pragma unroll
  for (int i_33 = 0; i_33 < 9; ++i_33) {
    tl::ptx_stmatrix_m8n8_x4((&(((half_t*)O_shared)[(((((((int)threadIdx.x) >> 5) * 2304) + ((((int)threadIdx.x) & 15) * 144)) + (i_33 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), __pack_half2(((half_t)acc_o[(i_33 * 8)]), ((half_t)acc_o[((i_33 * 8) + 1)])), __pack_half2(((half_t)acc_o[((i_33 * 8) + 2)]), ((half_t)acc_o[((i_33 * 8) + 3)])), __pack_half2(((half_t)acc_o[((i_33 * 8) + 4)]), ((half_t)acc_o[((i_33 * 8) + 5)])), __pack_half2(((half_t)acc_o[((i_33 * 8) + 6)]), ((half_t)acc_o[((i_33 * 8) + 7)])));
  }
  __syncthreads();
  #pragma unroll
  for (int i_34 = 0; i_34 < 9; ++i_34) {
    if (((((int)blockIdx.x) * 2) + (((i_34 * 64) + (((int)threadIdx.x) >> 1)) / 288)) < 63) {
      *(uint4*)(Out + ((((((int)blockIdx.x) * 82944) + ((((i_34 * 64) + (((int)threadIdx.x) >> 1)) / 9) * 1296)) + (((int)blockIdx.y) * 144)) + ((((i_34 * 128) + ((int)threadIdx.x)) % 18) * 8))) = *(uint4*)(((half_t*)O_shared) + ((i_34 * 1024) + (((int)threadIdx.x) * 8)));
    }
  }
}

