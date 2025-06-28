#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cstdint>

// decode_i2s_to_i8s 함수 정의
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

__global__ void test_decode_kernel(uint32_t *input, int8_t *output, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        decode_i2s_to_i8s(&input[idx], &output[idx * 16], 16);
    }
}

void print_binary(uint32_t val) {
    for (int i = 31; i >= 0; i--) {
        printf("%d", (val >> i) & 1);
        if (i % 4 == 0) printf(" ");
    }
    printf("\n");
}

int main() {
    printf("=== decode_i2s_to_i8s 테스트 ===\n\n");
    
    // 테스트 케이스들
    std::vector<uint32_t> test_cases = {
        0x00000000,  // 모든 값이 00 (모두 -2)
        0xFFFFFFFF,  // 모든 값이 11 (모두 1)
        0x55555555,  // 모든 값이 01 (모두 -1)
        0xAAAAAAAA,  // 모든 값이 10 (모두 0)
        0x1B1B1B1B,  // 0110 1101 1011 (혼합 패턴)
        
        // 추가 무작위 테스트 케이스들
        0x12345678,  // 다양한 패턴 1
        0x9ABCDEF0,  // 다양한 패턴 2
        0xCAFEBABE,  // 다양한 패턴 3
        0xDEADBEEF,  // 다양한 패턴 4
        0x13579BDF,  // 홀수 패턴
        0x2468ACE0,  // 짝수 패턴
        0x0F0F0F0F,  // 교대 패턴 1
        0xF0F0F0F0,  // 교대 패턴 2
        0x87654321,  // 역순 패턴
        0x11111111,  // 반복 패턴 1
        0x22222222,  // 반복 패턴 2
        0x33333333,  // 반복 패턴 3
        0x01234567,  // 순차 패턴
        0xFEDCBA98,  // 역순차 패턴
        0x5A5A5A5A,  // 체크보드 패턴
    };
    
    int num_tests = test_cases.size();
    
    // GPU 메모리 할당
    uint32_t *d_input;
    int8_t *d_output;
    
    cudaMalloc(&d_input, num_tests * sizeof(uint32_t));
    cudaMalloc(&d_output, num_tests * 16 * sizeof(int8_t));
    
    // 호스트 출력 버퍼
    std::vector<int8_t> h_output(num_tests * 16);
    
    // 입력 데이터 복사
    cudaMemcpy(d_input, test_cases.data(), num_tests * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    // 커널 실행
    dim3 block(32);
    dim3 grid((num_tests + block.x - 1) / block.x);
    
    test_decode_kernel<<<grid, block>>>(d_input, d_output, num_tests);
    
    // 결과 복사
    cudaMemcpy(h_output.data(), d_output, num_tests * 16 * sizeof(int8_t), cudaMemcpyDeviceToHost);
    
    // 동기화 및 에러 체크
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA 에러: %s\n", cudaGetErrorString(error));
        return -1;
    }
    
    // 결과 출력
    for (int i = 0; i < num_tests; i++) {
        printf("테스트 케이스 %d:\n", i + 1);
        printf("입력 (32비트): 0x%08X\n", test_cases[i]);
        printf("이진 표현: ");
        print_binary(test_cases[i]);
        
        printf("출력 (16개 int8): ");
        for (int j = 0; j < 16; j++) {
            printf("%3d ", h_output[i * 16 + j]);
        }
        printf("\n");
        
        // 예상 결과 확인
        printf("예상 결과: ");
        for (int j = 0; j < 16; j++) {
            int bit_pos = j * 2;
            int val = (test_cases[i] >> bit_pos) & 0x3;
            int expected = (val == 0) ? -2 : (val == 1) ? -1 : (val == 2) ? 0 : 1;
            printf("%3d ", expected);
        }
        printf("\n");
        
        // 검증
        bool correct = true;
        for (int j = 0; j < 16; j++) {
            int bit_pos = j * 2;
            int val = (test_cases[i] >> bit_pos) & 0x3;
            int expected = (val == 0) ? -2 : (val == 1) ? -1 : (val == 2) ? 0 : 1;
            if (h_output[i * 16 + j] != expected) {
                correct = false;
                break;
            }
        }
        
        printf("결과: %s\n", correct ? "✓ 통과" : "✗ 실패");
        printf("----------------------------------------\n");
    }
    
    // 메모리 해제
    cudaFree(d_input);
    cudaFree(d_output);
    
    printf("\n테스트 완료!\n");
    return 0;
}
