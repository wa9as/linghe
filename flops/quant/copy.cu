#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdio.h>
#include <iostream>

#define dtype __nv_bfloat16

__global__ void init_data_kernel(dtype *x, int N) {
    // int global_idx = blockIdx.x * 1024 + threadIdx.x;
    int global_idx = threadIdx.x;
    for (int i = global_idx; i < 1024; i += blockDim.x) {
        x[i] = __float2bfloat16(float(i));
        // x[i] = float(i);
        // if (threadIdx.x == 1 && blockIdx.x == 0) {
        // if (blockIdx.x == 0) {
        //       printf("bfx %f, x %f\n", __bfloat162float(x[i]), float(i));
        // }
    }
}

__global__ void cp_kernel(dtype *x, int N) {

  const int BYTES = 16;
  const int threads = 128;
  const int num_per_thread = BYTES / sizeof(dtype);
  __shared__ dtype smem[threads * num_per_thread];
  __shared__ __nv_fp8_e4m3 smem_fp8[threads * num_per_thread];

  int index = threadIdx.x * num_per_thread;
  uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + index));
  uint32_t smem_fp8_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_fp8 + index));
  dtype* glob_ptr = x + index;

  float A_frag[4];
  float C_frag[2];

  asm volatile(
    "{\n"
    // " cp.async.cg.shared.global [%0], [%1], %2, 8;\n" 
    " cp.async.cg.shared.global [%0], [%1], %2, 16;\n"
    // "cp.async.ca.shared.global.L2::128B [%0], [%1], 16;\n"
    "cp.async.commit_group;\n"
    "cp.async.wait_group 0\n;"
    "}\n" :: "r"(smem_ptr), "l"(glob_ptr), "n"(BYTES)
  );

  // if (threadIdx.x == 1 && blockIdx.x == 0) {
  //   for (int i = 0; i < threads * num_per_thread; i++) {
  //       printf("smem %f \n", __bfloat162float(smem[i]));
  //   }
  // }
  
  asm volatile (
    "ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n"
    : "=f"(A_frag[0]), "=f"(A_frag[1]), "=f"(A_frag[2]), "=f"(A_frag[3])
    : "r"(smem_ptr)
  );

  // __nv_bfloat162 a = __float2bfloat162_rn(A_frag[0]);
  __nv_bfloat162 a = reinterpret_cast<__nv_bfloat162&>(A_frag[0]);
  __nv_bfloat162 b = reinterpret_cast<__nv_bfloat162&>(A_frag[1]);
  // __nv_bfloat162 c = reinterpret_cast<__nv_bfloat162&>(A_frag[2]);
  // __nv_bfloat162 d = reinterpret_cast<__nv_bfloat162&>(A_frag[3]);

  
  // if (threadIdx.x == 1 && blockIdx.x == 0) {
  //     // __nv_bfloat16 low_bf16 = __low2bfloat16(a);
  //     // __nv_bfloat16 high_bf16 = __high2bfloat16(a);
  //     // // __nv_bfloat16 first_bf16 = a.x;
  //     // printf("reg a.x %f \n", __bfloat162float(low_bf16));
  //     // printf("reg a.y %f \n", __bfloat162float(high_bf16));
  //     printf("reg a.x %f \n", __bfloat162float(a.x));
  //     printf("reg a.y %f \n", __bfloat162float(a.y));
  //     printf("reg b.x %f \n", __bfloat162float(b.x));
  //     printf("reg b.y %f \n", __bfloat162float(b.y));
  //     // printf("reg c.x %f \n", __bfloat162float(c.x));
  //     // printf("reg c.y %f \n", __bfloat162float(c.y));
  // }
  float scale = 0.5f;
  __nv_bfloat16*  A_frag_bf   = reinterpret_cast<__nv_bfloat16*>(A_frag);
  __nv_fp8_e4m3*  C_frag_fp8   = reinterpret_cast<__nv_fp8_e4m3*>(C_frag);

  for (int i = 0; i < num_per_thread; i++) {
      C_frag_fp8[i] =  __nv_fp8_e4m3(__bfloat162float(A_frag_bf[i]) * scale);
      // if (threadIdx.x == 1 && blockIdx.x == 0) {
      //   printf("cfrag %f \n", __bfloat162float(A_frag_bf[i]) * scale);
      // }
  }

  // asm volatile("st.shared.v2.u32 [%0], {%1, %2};\n"
  //                     : :"r"(smem_fp8_ptr), "r"(3254828548U), "r"(3254828548U));

  asm volatile("st.shared.v2.f32 [%0], {%1, %2};\n"
                    : :"r"(smem_fp8_ptr), "r"(255.0f), "r"(255.0f));

  // asm volatile (
  //   "st.shared.v4.f32 [%0], {%1, %2, %3, %4};\n"
  //   : : "r"(addr), "f"(reg0), "f"(reg1), "f"(reg2), "f"(reg3)
  // );

}

int main() {
    const int N_DATA = 1024 * 1024;
    dtype *x;
    cudaMalloc(&x, N_DATA * sizeof(dtype));
    int block = 256;
    int grid = 1024;
    init_data_kernel<<<grid, block>>>(x, N_DATA);

    cp_kernel<<<grid, 128>>>(x, N_DATA);  
    cudaDeviceSynchronize();
    std::cout <<  cudaGetErrorString( cudaGetLastError() ) << std::endl;
    cudaFree(x);
    return 0;
}