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
  void* K_shared = ((void*)((char*)buf_dyn_shmem + 16384));
  void* V_shared = ((void*)((char*)buf_dyn_shmem + 32768));
  float acc_o[64];
  float lsum[2];
  float m[2];
  float acc_s[32];
  float m_prev[2];
  float sc[2];
  float ssum[2];
  half_t acc_s_cast[32];
  float m_clear[2];
  float m_clear_1[2];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    half_t broadcast_var = half_t(0x0p+0f/*0.000000e+00*/);
    uint4 condval;
    if ((((((((int)threadIdx.x) >> 6) + ((int)blockIdx.y)) >> 1) + i) < 4)) {
      condval = *(uint4*)(Q + (((i * 1024) + (((int)blockIdx.y) * 512)) + (((int)threadIdx.x) * 8)));
    } else {
      condval = make_uint4(__pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var));
    }
    *(uint4*)(((half_t*)Q_shared) + ((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval;
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 16; ++i_1) {
    float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_o + (i_1 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
  }
  float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
  *(float2*)(lsum + 0) = make_float2(broadcast_var_2, broadcast_var_2);
  float broadcast_var_3 = -CUDART_INF_F;
  *(float2*)(m + 0) = make_float2(broadcast_var_3, broadcast_var_3);
  __syncthreads();
  #pragma unroll
  for (int i_2 = 0; i_2 < 8; ++i_2) {
    tl::cp_async_gs<16>((&(((half_t*)K_shared)[((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_2 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(K[(((((((int)blockIdx.z) * 524288) + (i_2 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_3 = 0; i_3 < 8; ++i_3) {
    tl::cp_async_gs<16>((&(((half_t*)V_shared)[((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_3 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(V[(((((((int)blockIdx.z) * 524288) + (i_3 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8))])));
  }
  tl::cp_async_commit();
  for (int k = 0; k < 7; ++k) {
    #pragma unroll
    for (int i_4 = 0; i_4 < 8; ++i_4) {
      float broadcast_var_4 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_s + (i_4 * 4)) = make_float4(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4);
    }
    tl::cp_async_wait<1>();
    __syncthreads();
    {
      half_t A_local[8];
      half_t B_local[32];
      for (int ki = 0; ki < 8; ++ki) {
        tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[(((((((ki >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local[0])));
        #pragma unroll
        for (int i_5 = 0; i_5 < 4; ++i_5) {
          tl::ptx_ldmatrix_x4((&(((half_t*)K_shared)[((((((((ki >> 2) * 4096) + (i_5 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[(i_5 * 8)])));
        }
        for (int j = 0; j < 4; ++j) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (j * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_6 = 0; i_6 < 8; ++i_6) {
      tl::cp_async_gs<16>((&(((half_t*)K_shared)[((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_6 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(K[(((((((((int)blockIdx.z) * 524288) + (k * 65536)) + (i_6 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 65536)])));
    }
    tl::cp_async_commit();
    *(float2*)(m_prev + 0) = *(float2*)(m + 0);
    float broadcast_var_5 = -CUDART_INF_F;
    *(float2*)(m + 0) = make_float2(broadcast_var_5, broadcast_var_5);
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
    float2 __1;
      float2 v_ = *(float2*)(m + 0);
      float2 v__1 = *(float2*)(m_prev + 0);
      *(float2*)(&(__1.x)) = tl::max2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)));
    *(float2*)(m + 0) = __1;
    float broadcast_var_6 = 0x1.0527dbd5cafffp-3f/*1.275174e-01*/;
    float broadcast_var_7 = 0x1.0527dbd5cafffp-3f/*1.275174e-01*/;
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
    for (int i_8 = 0; i_8 < 16; ++i_8) {
      float broadcast_var_8 = 0x1.0527dbd5cafffp-3f/*1.275174e-01*/;
      float2 __6;
      float2 __7;
        float2 __8;
          float2 v__6 = *(float2*)(acc_s + (i_8 * 2));
          float2 v__7 = make_float2(broadcast_var_8, broadcast_var_8);
          *(float2*)(&(__8.x)) = tl::mul2(*(float2*)(&(v__6.x)), *(float2*)(&(v__7.x)));
        float2 v__8 = make_float2((m[(i_8 & 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/), (m[(i_8 & 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/));
        *(float2*)(&(__7.x)) = tl::sub2(*(float2*)(&(__8.x)), *(float2*)(&(v__8.x)));
      __6.x = exp2f(__7.x);
      __6.y = exp2f(__7.y);
      *(float2*)(acc_s + (i_8 * 2)) = __6;
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 2; ++i_9) {
      ssum[i_9] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
        ssum[i_9] = (ssum[i_9] + acc_s[((((rv_1 & 7) * 4) + (i_9 * 2)) + (rv_1 >> 3))]);
      }
      ssum[i_9] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(ssum[i_9]);
    }
    float2 __9;
      float2 v__9 = *(float2*)(lsum + 0);
      float2 v__10 = *(float2*)(sc + 0);
      float2 v__11 = *(float2*)(ssum + 0);
      *(float2*)(&(__9.x)) = tl::fma2(*(float2*)(&(v__9.x)), *(float2*)(&(v__10.x)), *(float2*)(&(v__11.x)));
    *(float2*)(lsum + 0) = __9;
    #pragma unroll
    for (int i_10 = 0; i_10 < 8; ++i_10) {
      uint2 __10;
      float4 v__12 = *(float4*)(acc_s + (i_10 * 4));
      ((half2*)(&__10))[0] = __float22half2_rn(((float2*)(&v__12))[0]);
      ((half2*)(&__10))[1] = __float22half2_rn(((float2*)(&v__12))[1]);
      *(uint2*)(acc_s_cast + (i_10 * 4)) = __10;
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 32; ++i_11) {
      float2 __11;
        float2 v__13 = *(float2*)(acc_o + (i_11 * 2));
        float2 v__14 = make_float2(sc[(i_11 & 1)], sc[(i_11 & 1)]);
        *(float2*)(&(__11.x)) = tl::mul2(*(float2*)(&(v__13.x)), *(float2*)(&(v__14.x)));
      *(float2*)(acc_o + (i_11 * 2)) = __11;
    }
    tl::cp_async_wait<1>();
    __syncthreads();
    {
      half_t B_local_1[64];
      for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
        #pragma unroll
        for (int i_12 = 0; i_12 < 8; ++i_12) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)V_shared)[(((((((i_12 >> 2) * 4096) + (ki_1 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_12 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_12 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_1[(i_12 * 8)])));
        }
        for (int j_1 = 0; j_1 < 8; ++j_1) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_1 * 8)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_1 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_1 * 8)), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 8) + 4)));
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_13 = 0; i_13 < 8; ++i_13) {
      tl::cp_async_gs<16>((&(((half_t*)V_shared)[((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_13 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(V[(((((((((int)blockIdx.z) * 524288) + (k * 65536)) + (i_13 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 65536)])));
    }
    tl::cp_async_commit();
  }
  #pragma unroll
  for (int i_14 = 0; i_14 < 8; ++i_14) {
    float broadcast_var_9 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_s + (i_14 * 4)) = make_float4(broadcast_var_9, broadcast_var_9, broadcast_var_9, broadcast_var_9);
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  {
    half_t A_local_1[8];
    half_t B_local_2[32];
    for (int ki_2 = 0; ki_2 < 8; ++ki_2) {
      tl::ptx_ldmatrix_x4((&(((half_t*)Q_shared)[(((((((ki_2 >> 2) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local_1[0])));
      #pragma unroll
      for (int i_15 = 0; i_15 < 4; ++i_15) {
        tl::ptx_ldmatrix_x4((&(((half_t*)K_shared)[((((((((ki_2 >> 2) * 4096) + (i_15 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_2[(i_15 * 8)])));
      }
      for (int j_2 = 0; j_2 < 4; ++j_2) {
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + (j_2 * 8)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_2 + (j_2 * 8)));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s + ((j_2 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_2 + ((j_2 * 8) + 4)));
      }
    }
  }
  *(float2*)(m_prev + 0) = *(float2*)(m + 0);
  float broadcast_var_10 = -CUDART_INF_F;
  *(float2*)(m + 0) = make_float2(broadcast_var_10, broadcast_var_10);
  #pragma unroll
  for (int i_16 = 0; i_16 < 2; ++i_16) {
    m_clear_1[i_16] = -CUDART_INF_F;
    #pragma unroll
    for (int rv_2 = 0; rv_2 < 16; ++rv_2) {
      m_clear_1[i_16] = max(m_clear_1[i_16], acc_s[((((rv_2 & 7) * 4) + (i_16 * 2)) + (rv_2 >> 3))]);
    }
    m_clear_1[i_16] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(m_clear_1[i_16]);
    m[i_16] = max(m[i_16], m_clear_1[i_16]);
  }
  float2 __12;
    float2 v__15 = *(float2*)(m + 0);
    float2 v__16 = *(float2*)(m_prev + 0);
    *(float2*)(&(__12.x)) = tl::max2(*(float2*)(&(v__15.x)), *(float2*)(&(v__16.x)));
  *(float2*)(m + 0) = __12;
  float broadcast_var_11 = 0x1.0527dbd5cafffp-3f/*1.275174e-01*/;
  float broadcast_var_12 = 0x1.0527dbd5cafffp-3f/*1.275174e-01*/;
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
  for (int i_17 = 0; i_17 < 16; ++i_17) {
    float broadcast_var_13 = 0x1.0527dbd5cafffp-3f/*1.275174e-01*/;
    float2 __17;
    float2 __18;
      float2 __19;
        float2 v__21 = *(float2*)(acc_s + (i_17 * 2));
        float2 v__22 = make_float2(broadcast_var_13, broadcast_var_13);
        *(float2*)(&(__19.x)) = tl::mul2(*(float2*)(&(v__21.x)), *(float2*)(&(v__22.x)));
      float2 v__23 = make_float2((m[(i_17 & 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/), (m[(i_17 & 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/));
      *(float2*)(&(__18.x)) = tl::sub2(*(float2*)(&(__19.x)), *(float2*)(&(v__23.x)));
    __17.x = exp2f(__18.x);
    __17.y = exp2f(__18.y);
    *(float2*)(acc_s + (i_17 * 2)) = __17;
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
    float2 v__24 = *(float2*)(lsum + 0);
    float2 v__25 = *(float2*)(sc + 0);
    float2 v__26 = *(float2*)(ssum + 0);
    *(float2*)(&(__20.x)) = tl::fma2(*(float2*)(&(v__24.x)), *(float2*)(&(v__25.x)), *(float2*)(&(v__26.x)));
  *(float2*)(lsum + 0) = __20;
  #pragma unroll
  for (int i_19 = 0; i_19 < 8; ++i_19) {
    uint2 __21;
    float4 v__27 = *(float4*)(acc_s + (i_19 * 4));
    ((half2*)(&__21))[0] = __float22half2_rn(((float2*)(&v__27))[0]);
    ((half2*)(&__21))[1] = __float22half2_rn(((float2*)(&v__27))[1]);
    *(uint2*)(acc_s_cast + (i_19 * 4)) = __21;
  }
  #pragma unroll
  for (int i_20 = 0; i_20 < 32; ++i_20) {
    float2 __22;
      float2 v__28 = *(float2*)(acc_o + (i_20 * 2));
      float2 v__29 = make_float2(sc[(i_20 & 1)], sc[(i_20 & 1)]);
      *(float2*)(&(__22.x)) = tl::mul2(*(float2*)(&(v__28.x)), *(float2*)(&(v__29.x)));
    *(float2*)(acc_o + (i_20 * 2)) = __22;
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    half_t B_local_3[64];
    for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
      #pragma unroll
      for (int i_21 = 0; i_21 < 8; ++i_21) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)V_shared)[(((((((i_21 >> 2) * 4096) + (ki_3 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_21 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_21 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local_3[(i_21 * 8)])));
      }
      for (int j_3 = 0; j_3 < 8; ++j_3) {
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + (j_3 * 8)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_3 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + (j_3 * 8)));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o + ((j_3 * 8) + 4)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_3 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + ((j_3 * 8) + 4)));
      }
    }
  }
  #pragma unroll
  for (int i_22 = 0; i_22 < 32; ++i_22) {
    float2 __23;
      float2 v__30 = *(float2*)(acc_o + (i_22 * 2));
      float2 v__31 = make_float2(lsum[(i_22 & 1)], lsum[(i_22 & 1)]);
      __23.x = (v__30.x/v__31.x);
      __23.y = (v__30.y/v__31.y);
    *(float2*)(acc_o + (i_22 * 2)) = __23;
  }
  float broadcast_var_14 = 0x1.0527dbd5cafffp-3f/*1.275174e-01*/;
  float2 __24;
    float2 v__32 = *(float2*)(m + 0);
    float2 v__33 = make_float2(broadcast_var_14, broadcast_var_14);
    float2 __25;
    float2 v__34 = *(float2*)(lsum + 0);
    __25.x = log2f(v__34.x);
    __25.y = log2f(v__34.y);
    *(float2*)(&(__24.x)) = tl::fma2(*(float2*)(&(v__32.x)), *(float2*)(&(v__33.x)), *(float2*)(&(__25.x)));
  *(float2*)(lsum + 0) = __24;
  if ((((int)threadIdx.x) % 4) == 0) {
    #pragma unroll
    for (int i_23 = 0; i_23 < 2; ++i_23) {
      if (((((((int)threadIdx.x) >> 5) * 4) + (i_23 * 2)) + ((((int)threadIdx.x) & 31) >> 4)) < 1) {
        glse[((((((((int)threadIdx.x) >> 5) * 128) + (i_23 * 64)) + (((int)blockIdx.y) * 32)) + (((((int)threadIdx.x) & 31) >> 2) * 8)) + ((int)blockIdx.z))] = ((half_t)lsum[i_23]);
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_24 = 0; i_24 < 32; ++i_24) {
    half_t O_shared_local_cast[2];
    if (((((((int)threadIdx.x) >> 5) * 4) + ((i_24 & 1) * 2)) + ((((int)threadIdx.x) & 31) >> 4)) < 1) {
      uint1 __26;
      float2 v__35 = *(float2*)(acc_o + (i_24 * 2));
      ((half2*)(&__26))[0] = __float22half2_rn(((float2*)(&v__35))[0]);
      *(uint1*)(O_shared_local_cast + 0) = __26;
      *(uint1*)(((half_t*)O_shared) + ((((((((int)threadIdx.x) >> 5) * 2048) + ((i_24 & 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_24 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(uint1*)(O_shared_local_cast + 0);
    }
  }
  __syncthreads();
  *(uint2*)(Opart + ((((((int)blockIdx.y) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (((int)blockIdx.z) * 128)) + ((((int)threadIdx.x) & 31) * 4))) = *(uint2*)(((half_t*)O_shared) + (((int)threadIdx.x) * 4));
}

