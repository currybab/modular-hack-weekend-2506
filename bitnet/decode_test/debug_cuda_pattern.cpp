#include <iostream>
#include <cstdint>

int main() {
    uint32_t test_val = 0x1B1B1B1B;
    printf("분석: 0x%08X\n", test_val);
    
    // 각 바이트별로 분석
    for (int byte_idx = 0; byte_idx < 4; byte_idx++) {
        uint8_t byte_val = (test_val >> (byte_idx * 8)) & 0xFF;
        printf("바이트 %d: 0x%02X = ", byte_idx, byte_val);
        
        // 이진수로 출력
        for (int bit = 7; bit >= 0; bit--) {
            printf("%d", (byte_val >> bit) & 1);
        }
        printf(" = ");
        
        // 2비트씩 분석
        for (int pair = 0; pair < 4; pair++) {
            int two_bits = (byte_val >> (pair * 2)) & 0x3;
            int decoded = (two_bits == 0) ? -2 : (two_bits == 1) ? -1 : (two_bits == 2) ? 0 : 1;
            printf("%d ", decoded);
        }
        printf("\n");
    }
    
    printf("\nCUDA 결과: [1,1,1,1,0,0,0,0,-1,-1,-1,-1,-2,-2,-2,-2]\n");
    printf("분석 결과를 보면 각 바이트가 4번씩 반복되는 패턴이 보입니다.\n");
    
    return 0;
}
