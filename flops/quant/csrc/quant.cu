#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>
#include <cmath>
#include <ATen/cuda/CUDAContext.h>

// #include <stdio.h>
// #include <stdlib.h>
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128(value) (reinterpret_cast<float4 *>(&(value))[0])

// using FP8_TYPE = c10::Float8_e4m3fnuz;
// using FP8_TYPE = __nv_fp8_e4m3;

// template <const int NUM_THREADS = 256>

template<typename T>
__inline__ __device__ T warpReduceMax(T val)
{
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val = max(val, __shfl_xor_sync(0xffffffff, val, mask, 32));
    return val;
}

template<typename T>
__inline__ __device__ T blockReduceMax(T val)
{
    static __shared__ T shared[32];
    int                 lane = threadIdx.x & 0x1f; 
    int                 wid  = threadIdx.x >> 5;   

    val = warpReduceMax(val); 

    if (lane == 0) 
        shared[wid] = val;

    __syncthreads();

    val = (lane < (blockDim.x / 32.f)) ? shared[lane] : -1e20f;
    val = warpReduceMax(val);

    return val;
}

__global__ void row_quant_bf16_kernel(__nv_bfloat16* __restrict__ x, 
                                      __nv_fp8_storage_t* __restrict__ y,
                                      float* __restrict__ s,
                                      const int64_t M, const int64_t K) {

                                                  
  int token_idx = blockIdx.x;
  int tid = threadIdx.x;
  int block_dim = blockDim.x;

  int input_idx = token_idx * K;
  int output_idx = token_idx * K;

  float max_value = 0.0f;

  uint32_t vec_size = 8;
  int32_t vec_num = K / vec_size;
  
  for (int32_t i = tid; i < vec_num; i += block_dim) {

      __nv_bfloat16 pack_x[8]; 
      LDST128(pack_x[0]) = LDST128(x[input_idx + i * vec_size]);

#pragma unroll
      for (uint32_t j = 0; j < vec_size; ++j) {
        // float val = static_cast<float>(pack_x[j]);
        float val = __bfloat162float(pack_x[j]);
        // printf("j %d, val %f\n", j, val);
        max_value = fmaxf(max_value, fabsf(val));
      }

  } 

  // if (tid == 0 && token_idx == 0){
  //   printf("max value %f\n", max_value);
  // }

  max_value = blockReduceMax(max_value);

  // if (tid == 32 && token_idx == 0){
  //   printf("max value %f\n", max_value);
  // }

  __shared__ float blockmax;
  if (tid == 0){
    blockmax = max_value / 448.0f;
    // blockmax = max_value;
    s[token_idx] = blockmax;
  }

  __syncthreads();

  const float scale = 448.0f / max_value;

  for (int32_t i = tid; i < vec_num; i += block_dim) {
    
    __nv_bfloat16 pack_x[8]; 
    LDST128(pack_x[0]) = LDST128(x[input_idx + i * vec_size]);
    // if (tid == 32 && token_idx == 0 ){
    //   printf("out value i: %d index: %d\n", i, input_idx + i * vec_size);
    // }
    
    // FP8_TYPE reg_out[8];
    __nv_fp8_storage_t reg_out[8];

#pragma unroll
    for (uint32_t j = 0; j < vec_size; ++j) {
      // float val = static_cast<float>(pack_x[j]);
      float val = fmaxf(fminf(__bfloat162float(pack_x[j]) * scale, 448.0),-448.0);
      // reg_out[j] = FP8_TYPE(val);
      reg_out[j] =  __nv_cvt_float_to_fp8(val, __NV_SATFINITE, __NV_E4M3);

      // if (tid == 0 && token_idx == 0 && i == 0 && j == 1){
      // if (tid == 32 && token_idx == 0 ){
      //     printf("pack value i: %d j: %d pack_value: %f\n", i, j, __bfloat162float(pack_x[j]));
      //     printf("float value i: %d j: %d val: %f\n", i, j, val);
      //     printf("scale value : %f\n", scale);
      //     printf("fp8 i: %d, j: %d, %f \n", i, j,  __half2float(__nv_cvt_fp8_to_halfraw(reg_out[j], __NV_E4M3)));
      // }
      // reg_out[j] = __nv_cvt_float_to_fp8(val, __NV_SATFINITE, __NV_E4M3);
      // reg_out[j] = __nv_cvt_float_to_fp8( -250.046509f, __NV_SATFINITE, __NV_E4M3);
      // printf("j %d, val %f\n", j, val);
      // y[output_idx + i * vec_size + j] = FP8_TYPE(val / blockmax);
    }

// #pragma unroll
//     for (uint32_t j = 0; j < vec_size; ++j) {
//       // y[output_idx + i * vec_size + j] = __nv_fp8_e4m3(reg_out[j]);
//       y[output_idx + i * vec_size + j] = reg_out[j];
//       // y[output_idx + i * vec_size + j] =  static_cast<FP8_TYPE>(-250.046509);
//     }
    
    LDST128(y[output_idx + i * vec_size]) = LDST128(reg_out[0]);
    
  }  
  
  
}

void row_quant_bf16(torch::Tensor x, torch::Tensor y, torch::Tensor s) {
  // CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
  // CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  // CHECK_TORCH_TENSOR_SHAPE(x, y)
  const int num_tokens = x.size(0);
  const int hidden_dim = x.size(1);

  dim3 grid(num_tokens);
  dim3 block(256);
  // dim3 block(1024);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  row_quant_bf16_kernel<<<grid, block, 0, stream>>>(
        static_cast<__nv_bfloat16*>(x.data_ptr()),
        static_cast<__nv_fp8_storage_t*>(y.data_ptr()),
        static_cast<float*>(s.data_ptr()),
        num_tokens,
        hidden_dim
  );
  // DISPATCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(N, K)
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(row_quant_bf16)
}