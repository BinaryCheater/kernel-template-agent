#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/copy_sm90.h>
#include <tl_templates/cuda/copy_sm100.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void gemm_kernel(const half_t* __restrict__ A, const signed char* __restrict__ Bq, half_t* __restrict__ C, const half_t* __restrict__ Sc);
extern "C" __global__ void __launch_bounds__(128, 1) gemm_kernel(const half_t* __restrict__ A, const signed char* __restrict__ Bq, half_t* __restrict__ C, const half_t* __restrict__ Sc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* Bq_shared = ((void*)((char*)buf_dyn_shmem + 0));
  void* Sc_shared = ((void*)((char*)buf_dyn_shmem + 6144));
  void* A_shared = ((void*)((char*)buf_dyn_shmem + 6912));
  void* B_shared = ((void*)((char*)buf_dyn_shmem + 31488));
  float C_local[16384];
  #pragma unroll
  for (int i = 0; i < 4096; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(C_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  tl::cp_async_gs<16>((&(((signed char*)Bq_shared)[(((int)threadIdx.x) * 16)])), (&(Bq[((((int)blockIdx.x) * 131072) + (((int)threadIdx.x) * 1024))])));
  __syncthreads();
  ((half_t*)Sc_shared)[((int)threadIdx.x)] = Sc[((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 64))];
  tl::cp_async_commit();
  __syncthreads();
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    tl::cp_async_gs<16>((&(((half_t*)A_shared)[((i_1 * 1024) + (((int)threadIdx.x) * 8))])), (&(A[((((((int)blockIdx.y) * 262144) + (i_1 * 65536)) + ((((int)threadIdx.x) >> 2) * 2048)) + ((((int)threadIdx.x) & 3) * 8))])));
  }
  tl::cp_async_commit();
  tl::cp_async_gs<16>((&(((signed char*)Bq_shared)[((((int)threadIdx.x) * 16) + 2048)])), (&(Bq[(((((int)blockIdx.x) * 131072) + (((int)threadIdx.x) * 1024)) + 16)])));
  __syncthreads();
  ((half_t*)Sc_shared)[(((int)threadIdx.x) + 128)] = Sc[(((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 64)) + 1)];
  tl::cp_async_commit();
  __syncthreads();
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    tl::cp_async_gs<16>((&(((half_t*)A_shared)[(((i_2 * 1024) + (((int)threadIdx.x) * 8)) + 4096)])), (&(A[(((((((int)blockIdx.y) * 262144) + (i_2 * 65536)) + ((((int)threadIdx.x) >> 2) * 2048)) + ((((int)threadIdx.x) & 3) * 8)) + 32)])));
  }
  tl::cp_async_commit();
  tl::cp_async_gs<16>((&(((signed char*)Bq_shared)[((((int)threadIdx.x) * 16) + 4096)])), (&(Bq[(((((int)blockIdx.x) * 131072) + (((int)threadIdx.x) * 1024)) + 32)])));
  __syncthreads();
  ((half_t*)Sc_shared)[(((int)threadIdx.x) + 256)] = Sc[(((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 64)) + 2)];
  tl::cp_async_commit();
  __syncthreads();
  #pragma unroll
  for (int i_3 = 0; i_3 < 4; ++i_3) {
    tl::cp_async_gs<16>((&(((half_t*)A_shared)[(((i_3 * 1024) + (((int)threadIdx.x) * 8)) + 8192)])), (&(A[(((((((int)blockIdx.y) * 262144) + (i_3 * 65536)) + ((((int)threadIdx.x) >> 2) * 2048)) + ((((int)threadIdx.x) & 3) * 8)) + 64)])));
  }
  tl::cp_async_commit();
  for (int ko = 0; ko < 61; ++ko) {
    tl::cp_async_wait<5>();
    __syncthreads();
    #pragma unroll
    for (int i_4 = 0; i_4 < 16; ++i_4) {
      signed char Bq_shared_local_cast_1[2];
      half_t Sc_shared_local_cast_2[2];
      half_t B_shared_local_cast[2];
      *(char2*)(Bq_shared_local_cast_1 + 0) = make_char2(((signed char*)Bq_shared)[((((ko % 3) * 2048) + (i_4 * 128)) + ((int)threadIdx.x))], ((signed char*)Bq_shared)[((((ko % 3) * 2048) + (i_4 * 128)) + ((int)threadIdx.x))]);
      *(uint1*)(Sc_shared_local_cast_2 + 0) = make_uint1(__pack_half2(((half_t*)Sc_shared)[((((ko % 3) * 128) + (i_4 * 8)) + (((int)threadIdx.x) >> 4))], ((half_t*)Sc_shared)[((((ko % 3) * 128) + (i_4 * 8)) + (((int)threadIdx.x) >> 4))]));
      int broadcast_var_1 = 15;
      half_t broadcast_var_2 = half_t(0x1p+3f/*8.000000e+00*/);
      uint1 __1;
        uint1 __2;
          uint1 __3;
          int2 __4;
            int2 __5;
              int2 __6;
              char2 v_ = *(char2*)(Bq_shared_local_cast_1 + 0);
              __6.x = (int)(v_.x);
              __6.y = (int)(v_.y);
              int2 v__1 = make_int2((0)+(4*0), (0)+(4*1));
              __5.x = (__6.x >> v__1.x);
              __5.y = (__6.y >> v__1.y);
            int2 v__2 = make_int2(broadcast_var_1, broadcast_var_1);
            __4.x = (__5.x & v__2.x);
            __4.y = (__5.y & v__2.y);
          ((half2*)(&(__3.x)))->x = (half_t)(__4.x);
          ((half2*)(&(__3.x)))->y = (half_t)(__4.y);
          uint1 v__3 = make_uint1(__pack_half2(broadcast_var_2, broadcast_var_2));
          *(uint1*)(&(__2.x)) = tl::to_uint1(tl::sub2(tl::from_uint1<__half2>(*(uint1*)(&(__3.x))), tl::from_uint1<__half2>(*(uint1*)(&(v__3.x)))));
        uint1 v__4 = *(uint1*)(Sc_shared_local_cast_2 + 0);
        *(uint1*)(&(__1.x)) = tl::to_uint1(tl::mul2(tl::from_uint1<__half2>(*(uint1*)(&(__2.x))), tl::from_uint1<__half2>(*(uint1*)(&(v__4.x)))));
      *(uint1*)(B_shared_local_cast + 0) = __1;
      *(uint1*)(((half_t*)B_shared) + ((i_4 * 256) + (((int)threadIdx.x) * 2))) = *(uint1*)(B_shared_local_cast + 0);
    }
    __syncthreads();
    tl::cp_async_gs<16>((&(((signed char*)Bq_shared)[(((ko % 3) * 2048) + (((int)threadIdx.x) * 16))])), (&(Bq[((((((int)blockIdx.x) * 131072) + (((int)threadIdx.x) * 1024)) + (ko * 16)) + 48)])));
    __syncthreads();
    ((half_t*)Sc_shared)[(((ko % 3) * 128) + ((int)threadIdx.x))] = Sc[((((((int)blockIdx.x) * 8192) + (((int)threadIdx.x) * 64)) + ko) + 3)];
    tl::cp_async_commit();
    tl::cp_async_wait<5>();
    __syncthreads();
    C_local[0] = (((float)((half_t*)A_shared)[((ko % 3) * 4096)]) + ((float)((half_t*)B_shared)[0]));
    __syncthreads();
    #pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      tl::cp_async_gs<16>((&(((half_t*)A_shared)[((((ko % 3) * 4096) + (i_5 * 1024)) + (((int)threadIdx.x) * 8))])), (&(A[((((((((int)blockIdx.y) * 262144) + (i_5 * 65536)) + ((((int)threadIdx.x) >> 2) * 2048)) + (ko * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 96)])));
    }
    tl::cp_async_commit();
  }
  tl::cp_async_wait<5>();
  __syncthreads();
  #pragma unroll
  for (int i_6 = 0; i_6 < 16; ++i_6) {
    signed char Bq_shared_local_cast_4[2];
    half_t Sc_shared_local_cast_5[2];
    half_t B_shared_local_cast_3[2];
    *(char2*)(Bq_shared_local_cast_4 + 0) = make_char2(((signed char*)Bq_shared)[(((i_6 * 128) + ((int)threadIdx.x)) + 2048)], ((signed char*)Bq_shared)[(((i_6 * 128) + ((int)threadIdx.x)) + 2048)]);
    *(uint1*)(Sc_shared_local_cast_5 + 0) = make_uint1(__pack_half2(((half_t*)Sc_shared)[(((i_6 * 8) + (((int)threadIdx.x) >> 4)) + 128)], ((half_t*)Sc_shared)[(((i_6 * 8) + (((int)threadIdx.x) >> 4)) + 128)]));
    int broadcast_var_3 = 15;
    half_t broadcast_var_4 = half_t(0x1p+3f/*8.000000e+00*/);
    uint1 __7;
      uint1 __8;
        uint1 __9;
        int2 __10;
          int2 __11;
            int2 __12;
            char2 v__5 = *(char2*)(Bq_shared_local_cast_4 + 0);
            __12.x = (int)(v__5.x);
            __12.y = (int)(v__5.y);
            int2 v__6 = make_int2((0)+(4*0), (0)+(4*1));
            __11.x = (__12.x >> v__6.x);
            __11.y = (__12.y >> v__6.y);
          int2 v__7 = make_int2(broadcast_var_3, broadcast_var_3);
          __10.x = (__11.x & v__7.x);
          __10.y = (__11.y & v__7.y);
        ((half2*)(&(__9.x)))->x = (half_t)(__10.x);
        ((half2*)(&(__9.x)))->y = (half_t)(__10.y);
        uint1 v__8 = make_uint1(__pack_half2(broadcast_var_4, broadcast_var_4));
        *(uint1*)(&(__8.x)) = tl::to_uint1(tl::sub2(tl::from_uint1<__half2>(*(uint1*)(&(__9.x))), tl::from_uint1<__half2>(*(uint1*)(&(v__8.x)))));
      uint1 v__9 = *(uint1*)(Sc_shared_local_cast_5 + 0);
      *(uint1*)(&(__7.x)) = tl::to_uint1(tl::mul2(tl::from_uint1<__half2>(*(uint1*)(&(__8.x))), tl::from_uint1<__half2>(*(uint1*)(&(v__9.x)))));
    *(uint1*)(B_shared_local_cast_3 + 0) = __7;
    *(uint1*)(((half_t*)B_shared) + ((i_6 * 256) + (((int)threadIdx.x) * 2))) = *(uint1*)(B_shared_local_cast_3 + 0);
  }
  tl::cp_async_wait<4>();
  __syncthreads();
  C_local[0] = (((float)((half_t*)A_shared)[4096]) + ((float)((half_t*)B_shared)[0]));
  tl::cp_async_wait<3>();
  __syncthreads();
  #pragma unroll
  for (int i_7 = 0; i_7 < 16; ++i_7) {
    signed char Bq_shared_local_cast_7[2];
    half_t Sc_shared_local_cast_8[2];
    half_t B_shared_local_cast_6[2];
    *(char2*)(Bq_shared_local_cast_7 + 0) = make_char2(((signed char*)Bq_shared)[(((i_7 * 128) + ((int)threadIdx.x)) + 4096)], ((signed char*)Bq_shared)[(((i_7 * 128) + ((int)threadIdx.x)) + 4096)]);
    *(uint1*)(Sc_shared_local_cast_8 + 0) = make_uint1(__pack_half2(((half_t*)Sc_shared)[(((i_7 * 8) + (((int)threadIdx.x) >> 4)) + 256)], ((half_t*)Sc_shared)[(((i_7 * 8) + (((int)threadIdx.x) >> 4)) + 256)]));
    int broadcast_var_5 = 15;
    half_t broadcast_var_6 = half_t(0x1p+3f/*8.000000e+00*/);
    uint1 __13;
      uint1 __14;
        uint1 __15;
        int2 __16;
          int2 __17;
            int2 __18;
            char2 v__10 = *(char2*)(Bq_shared_local_cast_7 + 0);
            __18.x = (int)(v__10.x);
            __18.y = (int)(v__10.y);
            int2 v__11 = make_int2((0)+(4*0), (0)+(4*1));
            __17.x = (__18.x >> v__11.x);
            __17.y = (__18.y >> v__11.y);
          int2 v__12 = make_int2(broadcast_var_5, broadcast_var_5);
          __16.x = (__17.x & v__12.x);
          __16.y = (__17.y & v__12.y);
        ((half2*)(&(__15.x)))->x = (half_t)(__16.x);
        ((half2*)(&(__15.x)))->y = (half_t)(__16.y);
        uint1 v__13 = make_uint1(__pack_half2(broadcast_var_6, broadcast_var_6));
        *(uint1*)(&(__14.x)) = tl::to_uint1(tl::sub2(tl::from_uint1<__half2>(*(uint1*)(&(__15.x))), tl::from_uint1<__half2>(*(uint1*)(&(v__13.x)))));
      uint1 v__14 = *(uint1*)(Sc_shared_local_cast_8 + 0);
      *(uint1*)(&(__13.x)) = tl::to_uint1(tl::mul2(tl::from_uint1<__half2>(*(uint1*)(&(__14.x))), tl::from_uint1<__half2>(*(uint1*)(&(v__14.x)))));
    *(uint1*)(B_shared_local_cast_6 + 0) = __13;
    *(uint1*)(((half_t*)B_shared) + ((i_7 * 256) + (((int)threadIdx.x) * 2))) = *(uint1*)(B_shared_local_cast_6 + 0);
  }
  tl::cp_async_wait<2>();
  __syncthreads();
  C_local[0] = (((float)((half_t*)A_shared)[8192]) + ((float)((half_t*)B_shared)[0]));
  tl::cp_async_wait<1>();
  __syncthreads();
  #pragma unroll
  for (int i_8 = 0; i_8 < 16; ++i_8) {
    signed char Bq_shared_local_cast_10[2];
    half_t Sc_shared_local_cast_11[2];
    half_t B_shared_local_cast_9[2];
    *(char2*)(Bq_shared_local_cast_10 + 0) = make_char2(((signed char*)Bq_shared)[((i_8 * 128) + ((int)threadIdx.x))], ((signed char*)Bq_shared)[((i_8 * 128) + ((int)threadIdx.x))]);
    *(uint1*)(Sc_shared_local_cast_11 + 0) = make_uint1(__pack_half2(((half_t*)Sc_shared)[((i_8 * 8) + (((int)threadIdx.x) >> 4))], ((half_t*)Sc_shared)[((i_8 * 8) + (((int)threadIdx.x) >> 4))]));
    int broadcast_var_7 = 15;
    half_t broadcast_var_8 = half_t(0x1p+3f/*8.000000e+00*/);
    uint1 __19;
      uint1 __20;
        uint1 __21;
        int2 __22;
          int2 __23;
            int2 __24;
            char2 v__15 = *(char2*)(Bq_shared_local_cast_10 + 0);
            __24.x = (int)(v__15.x);
            __24.y = (int)(v__15.y);
            int2 v__16 = make_int2((0)+(4*0), (0)+(4*1));
            __23.x = (__24.x >> v__16.x);
            __23.y = (__24.y >> v__16.y);
          int2 v__17 = make_int2(broadcast_var_7, broadcast_var_7);
          __22.x = (__23.x & v__17.x);
          __22.y = (__23.y & v__17.y);
        ((half2*)(&(__21.x)))->x = (half_t)(__22.x);
        ((half2*)(&(__21.x)))->y = (half_t)(__22.y);
        uint1 v__18 = make_uint1(__pack_half2(broadcast_var_8, broadcast_var_8));
        *(uint1*)(&(__20.x)) = tl::to_uint1(tl::sub2(tl::from_uint1<__half2>(*(uint1*)(&(__21.x))), tl::from_uint1<__half2>(*(uint1*)(&(v__18.x)))));
      uint1 v__19 = *(uint1*)(Sc_shared_local_cast_11 + 0);
      *(uint1*)(&(__19.x)) = tl::to_uint1(tl::mul2(tl::from_uint1<__half2>(*(uint1*)(&(__20.x))), tl::from_uint1<__half2>(*(uint1*)(&(v__19.x)))));
    *(uint1*)(B_shared_local_cast_9 + 0) = __19;
    *(uint1*)(((half_t*)B_shared) + ((i_8 * 256) + (((int)threadIdx.x) * 2))) = *(uint1*)(B_shared_local_cast_9 + 0);
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  C_local[0] = (((float)((half_t*)A_shared)[0]) + ((float)((half_t*)B_shared)[0]));
  if (((int)threadIdx.x) == 0) {
    #pragma unroll
    for (int i_9 = 0; i_9 < 1024; ++i_9) {
      half_t C_local_cast_12[16];
      for (int vec = 0; vec < 4; ++vec) {
        uint2 __25;
        float4 v__20 = *(float4*)(C_local + ((i_9 * 16) + (vec * 4)));
        ((half2*)(&__25))[0] = __float22half2_rn(((float2*)(&v__20))[0]);
        ((half2*)(&__25))[1] = __float22half2_rn(((float2*)(&v__20))[1]);
        *(uint2*)(C_local_cast_12 + (vec * 4)) = __25;
      }
      tl::store_global_256(&(*(ulonglong4*)(C + ((((((int)blockIdx.y) * 262144) + ((i_9 >> 3) * 2048)) + (((int)blockIdx.x) * 128)) + ((i_9 & 7) * 16)))), *(ulonglong4*)(C_local_cast_12 + 0));
    }
  }
}

