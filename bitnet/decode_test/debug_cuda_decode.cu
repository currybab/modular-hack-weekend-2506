#include <iostream>
#include <cstdint>

// CUDA decode_i2s_to_i8s 함수 (실제 구현)
template <typename T1, typename T2>
__device__ void decode_i2s_to_i8s(T1 *_i2s, T2 *_i8s, const int N = 16)
{
  uint *i8s = reinterpret_cast<uint *>(_i8s);
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

__global__ void debug_decode_kernel(uint32_t input, int8_t *output) {
    decode_i2s_to_i8s(&input, output, 16);
}

void analyze_single_case(uint32_t input) {
    printf("\n=== 분석: 0x%08X ===\n", input);
    
    // CPU에서 바이트별로 분석
    printf("바이트별 분석:\n");
    for (int byte_idx = 0; byte_idx < 4; byte_idx++) {
        uint8_t byte_val = (input >> (byte_idx * 8)) & 0xFF;
        printf("바이트 %d: 0x%02X = ", byte_idx, byte_val);
        
        // 각 바이트의 2비트 쌍들을 추출
        for (int bit_pair = 0; bit_pair < 4; bit_pair++) {
            int bit_pos = bit_pair * 2;
            int two_bits = (byte_val >> bit_pos) & 0x3;
            int decoded;
            if (two_bits == 0) decoded = -2;
            else if (two_bits == 1) decoded = -1;
            else if (two_bits == 2) decoded = 0;
            else decoded = 1;
            
            printf("[%d→%d] ", two_bits, decoded);
        }
        printf("\n");
    }
    
    // GPU에서 CUDA 함수 실행
    int8_t *d_output;
    cudaMalloc(&d_output, 16 * sizeof(int8_t));
    
    debug_decode_kernel<<<1, 1>>>(input, d_output);
    
    int8_t h_output[16];
    cudaMemcpy(h_output, d_output, 16 * sizeof(int8_t), cudaMemcpyDeviceToHost);
    
    printf("CUDA 결과: ");
    for (int i = 0; i < 16; i++) {
        printf("%d ", h_output[i]);
    }
    printf("\n");
    
    cudaFree(d_output);
}

int main() {
    // 몇 가지 테스트 케이스 분석
    analyze_single_case(0x22222222);  // 테스트 16
    analyze_single_case(0x01234567);  // 테스트 18  
    analyze_single_case(0x1B1B1B1B);  // 테스트 5 (우리가 맞다고 생각한 케이스)
    
    return 0;
}
