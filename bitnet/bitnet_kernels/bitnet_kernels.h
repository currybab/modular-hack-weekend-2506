#include <cuda_runtime.h>
#include <math_constants.h>
#include <math.h>
#include <mma.h>
#include <iostream>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>


#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 800
#define TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST 1
#else
#define TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST 0
#endif

template <typename T1, typename T2>
__device__ void decode_i2s_to_i8s(T1 *_i2s, T2 *_i8s, const int N = 16)
{
  // convert 8 int2b_t to 8 int8b_t -> 2 int32
  uint *i8s = reinterpret_cast<uint *>(_i8s);

  // i2s = {e0, e4, e8, e12, e1, e5, e9, e13, e2, e6, e10, e14, e3, e7, e11, e15}
  uint const i2s = *_i2s;

  static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;     // 0b11101010
  static constexpr uint BOTTOM_MASK = 0x03030303;          // 0xf -> 0b11 select 0,3
  static constexpr uint I4s_TO_I8s_MAGIC_NUM = 0x00000000; 

#pragma unroll
  for (int i = 0; i < (N / 4); i++)
  {
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(i8s[i])
                 : "r"(i2s >> (2 * i)), "n"(BOTTOM_MASK), "n"(I4s_TO_I8s_MAGIC_NUM), "n"(immLut));
    i8s[i] = __vsubss4(i8s[i], 0x02020202);
  }
}

template <int M, int N, int K, int ws_num, int K_block_size, int N_block_size>
__global__ void __launch_bounds__(128) ladder_int8xint2_kernel(int8_t* __restrict__ A, int8_t* __restrict__ B, __nv_bfloat16* __restrict__ dtype_transform, __nv_bfloat16* __restrict__ s, __nv_bfloat16* __restrict__ ws) {
  // ==================== KERNEL CONFIGURATION ====================
  constexpr int K_per_loop = 16;    // Process 16 elements per loop iteration
  constexpr int wmma_K = 32;        // Warp matrix tile size in K dimension
  constexpr int wmma_N = 16;        // Warp matrix tile size in N dimension
  
  // ==================== LOCAL MEMORY ALLOCATION ====================
  int in_thread_C_local[1];         // Thread-local accumulator for dot products
  signed char A_local[K_per_loop];  // Local buffer for A matrix elements (int8)
  int B_reshape_local[1];           // Buffer for packed int2 B matrix data (32-bit)
  signed char B_decode_local[K_per_loop]; // Buffer for decoded int8 B matrix elements
  int red_buf0[1];                  // Buffer for warp reduction result
  
  // Initialize accumulator
  in_thread_C_local[0] = 0;
  
  // ==================== MAIN COMPUTATION LOOP ====================
  #pragma unroll
  for (int k_0 = 0; k_0 < K/(K_per_loop * K_block_size); ++k_0) {
    
    // === LOAD A MATRIX (INT8) ===
    // Vectorized load: Read 16 bytes (int4 = 128 bits) at once from A matrix
    // Each thread loads its own chunk based on thread ID and loop iteration
    *(int4*)(A_local + 0) = *(int4*)(A + ((k_0 * K_per_loop * K_block_size) + (((int)threadIdx.x) * K_per_loop)));
    
    // === LOAD B MATRIX (PACKED INT2) ===
    // Complex indexing to handle packed int2 format and tiled memory layout
    // B matrix is packed: 4 int2 values = 1 byte, so divide indices by 4
    B_reshape_local[0] = *(int*)(B + 
      (((int)blockIdx.x) * N_block_size * K / 4) +           // Block offset
      (k_0 * K_block_size * K_per_loop * wmma_N / 4) +       // Loop iteration offset
      ((((int)threadIdx.x) >> 1) * wmma_K * wmma_N / 4) +    // Thread X offset (div by 2)
      ((((int)threadIdx.y) >> 3) * (wmma_K * wmma_N / 2) / 4) +  // Thread Y offset (div by 8)
      ((((int)threadIdx.x) & 1) * (wmma_K * wmma_N / 4) / 4) +   // Thread X remainder
      ((((int)threadIdx.y) & 7) * (wmma_K / 2) / 4)              // Thread Y remainder
      );
    
    // === DECODE INT2 TO INT8 ===
    // Convert packed int2 values to separate int8 values for computation
    decode_i2s_to_i8s(B_reshape_local, B_decode_local, 16);
    
    // === DOT PRODUCT COMPUTATION ===
    #pragma unroll
    for (int k_2_0 = 0; k_2_0 < 4; ++k_2_0) {
      // __dp4a: CUDA intrinsic for 4-way dot product of int8 values
      // Computes: A[0]*B[0] + A[1]*B[1] + A[2]*B[2] + A[3]*B[3] + accumulator
      in_thread_C_local[0] = __dp4a(*(int *)&A_local[((k_2_0 * 4))],*(int *)&B_decode_local[((k_2_0 * 4))], in_thread_C_local[0]);
    }
  }
  
  // ==================== WARP-LEVEL REDUCTION ====================
  // Move thread-local result to reduction buffer
  red_buf0[0] = in_thread_C_local[0];
  
  // Perform warp-level tree reduction using shuffle operations
  // Reduces K_block_size partial results into a single value
  #pragma unroll
  for (int offset = K_block_size/2; offset > 0; offset /= 2) {
    // __shfl_down_sync: Get value from thread at (current_thread + offset)
    // Sum with current thread's value for tree reduction
    red_buf0[0] += __shfl_down_sync(__activemask(), red_buf0[0], offset, K_block_size);
  }
  
  // ==================== OUTPUT WRITING ====================
  // Calculate output position and scaling factor index
  int out_idx = ((((int)blockIdx.x) * N_block_size) + ((int)threadIdx.y));
  int ws_idx = out_idx / (N / ws_num);  // Weight scaling index
  
  // Only thread 0 in each warp writes the final result
  if (threadIdx.x == 0)
    // Apply scaling: result / s[0] * ws[ws_idx], convert to bfloat16
    dtype_transform[out_idx] = (__nv_bfloat16)(((float)red_buf0[0])/(float)s[0]*(float)ws[ws_idx]);
}