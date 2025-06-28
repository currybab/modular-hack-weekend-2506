#include <iostream>
#include <cstdint>

template <typename T1, typename T2>
__device__ void decode_i2s_to_i8s(T1 *_i2s, T2 *_i8s, const int N = 16)
{
  uint *i8s = reinterpret_cast<uint *>(_i8s);
  uint const i2s = *_i2s;

  static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
  static constexpr uint BOTTOM_MASK = 0x03030303;
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

__global__ void test_kernel(uint32_t input, int8_t *output) {
    decode_i2s_to_i8s(&input, output, 16);
}

int main() {
    // 간단한 케이스들만 테스트
    uint32_t test_cases[] = {
        0x01234567,  // 복잡한 케이스
        0x03020100   // 순차적 케이스
    };
    
    for (int t = 0; t < 2; t++) {
        uint32_t input = test_cases[t];
        printf("\n=== 0x%08X ===\n", input);
        
        // 바이트별 분석
        for (int byte_idx = 0; byte_idx < 4; byte_idx++) {
            uint8_t byte_val = (input >> (byte_idx * 8)) & 0xFF;
            printf("바이트%d(0x%02X): ", byte_idx, byte_val);
            
            for (int bit_pair = 0; bit_pair < 4; bit_pair++) {
                int two_bits = (byte_val >> (bit_pair * 2)) & 0x3;
                int decoded = (two_bits == 0) ? -2 : (two_bits == 1) ? -1 : (two_bits == 2) ? 0 : 1;
                printf("[%d→%d] ", two_bits, decoded);
            }
            printf("\n");
        }
        
        // CUDA 실행
        int8_t *d_output;
        cudaMalloc(&d_output, 16 * sizeof(int8_t));
        test_kernel<<<1, 1>>>(input, d_output);
        
        int8_t h_output[16];
        cudaMemcpy(h_output, d_output, 16 * sizeof(int8_t), cudaMemcpyDeviceToHost);
        
        printf("CUDA: ");
        for (int i = 0; i < 16; i++) {
            printf("%d ", h_output[i]);
        }
        printf("\n");
        
        cudaFree(d_output);
    }
    
    return 0;
}
