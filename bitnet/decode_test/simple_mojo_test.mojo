from testing import assert_equal

@always_inline
fn decode_i2s_to_i8s_cuda_compatible(i2s: UInt32, i8s: UnsafePointer[Int8], n: Int = 16):
    """CUDA compatible decode_i2s_to_i8s implementation."""
    
    var output_idx = 0
    
    # Process each bit pair position (0-3) across all 4 bytes
    for bit_pair_idx in range(4):
        for byte_idx in range(4):
            if output_idx >= n:
                break
                
            var byte_val = (i2s >> (byte_idx * 8)) & UInt32(0xFF)
            var bit_pos = bit_pair_idx * 2
            var two_bits = (byte_val >> bit_pos) & UInt32(0x3)
            
            var decoded_val: Int8
            if two_bits == 0:
                decoded_val = -2
            elif two_bits == 1:
                decoded_val = -1 
            elif two_bits == 2:
                decoded_val = 0
            else:  # two_bits == 3
                decoded_val = 1
                
            i8s[output_idx] = decoded_val
            output_idx += 1

fn main():
    # 간단한 테스트
    var test_input = UInt32(0x01234567)
    var output = UnsafePointer[Int8].alloc(16)
    
    decode_i2s_to_i8s_cuda_compatible(test_input, output, 16)
    
    print("입력: 0x01234567")
    print("Mojo 출력: ", end="")
    for i in range(16):
        print(Int(output[i]), end=" ")
    print()
    print("CUDA 예상: 1 -1 1 -1 -1 -1 -2 -2 0 -2 0 -2 -1 -1 -2 -2")
    
    # 검증
    var expected = List[Int8](1, -1, 1, -1, -1, -1, -2, -2, 0, -2, 0, -2, -1, -1, -2, -2)
    var all_match = True
    for i in range(16):
        if output[i] != expected[i]:
            all_match = False
            print("위치", i, "에서 불일치: Mojo =", Int(output[i]), ", CUDA =", Int(expected[i]))
    
    if all_match:
        print("✓ 완벽 일치!")
    else:
        print("✗ 불일치 발견")
    
    output.free()
