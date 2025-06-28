from testing import assert_equal

@always_inline
fn decode_i2s_to_i8s_cuda_compatible(i2s: Int32, i8s: UnsafePointer[Int8], n: Int = 16):
    """CUDA compatible decode_i2s_to_i8s implementation."""
    
    # CUDA groups by bit position across all bytes:
    # - All 1st bit pairs from each byte → positions 0-3
    # - All 2nd bit pairs from each byte → positions 4-7  
    # - All 3rd bit pairs from each byte → positions 8-11
    # - All 4th bit pairs from each byte → positions 12-15
    
    var output_idx = 0
    
    # Process each bit pair position (0-3) across all 4 bytes
    for bit_pair_idx in range(4):
        for byte_idx in range(4):
            if output_idx >= n:
                break
                
            var byte_val = (i2s >> (byte_idx * 8)) & Int32(0xFF)
            var bit_pos = bit_pair_idx * 2
            var two_bits = (byte_val >> bit_pos) & Int32(0x3)
            
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


fn test_decode():
    """Test decode function against CUDA reference results."""
    
    print("=== Mojo decode_i2s_to_i8s 테스트 ===")
    
    # Test cases matching CUDA test
    var test_cases = List[Int32]()
    test_cases.append(Int32(0x00000000))  # all 00 → all -2
    test_cases.append(Int32(0xFFFFFFFF))  # all 11 → all 1  
    test_cases.append(Int32(0x55555555))  # all 01 → all -1
    test_cases.append(Int32(0xAAAAAAAA)) # all 10 → all 0
    test_cases.append(Int32(0x1B1B1B1B)) # mixed pattern
    
    # Additional random test cases
    test_cases.append(Int32(0x12345678))  # various pattern 1
    test_cases.append(Int32(0x9ABCDEF0))  # various pattern 2
    test_cases.append(Int32(0xCAFEBABE))  # various pattern 3
    test_cases.append(Int32(0xDEADBEEF))  # various pattern 4
    test_cases.append(Int32(0x13579BDF))  # odd pattern
    test_cases.append(Int32(0x2468ACE0))  # even pattern
    test_cases.append(Int32(0x0F0F0F0F))  # alternating pattern 1
    test_cases.append(Int32(0xF0F0F0F0))  # alternating pattern 2
    test_cases.append(Int32(0x87654321))  # reverse pattern
    test_cases.append(Int32(0x11111111))  # repeat pattern 1
    test_cases.append(Int32(0x22222222))  # repeat pattern 2
    test_cases.append(Int32(0x33333333))  # repeat pattern 3
    test_cases.append(Int32(0x01234567))  # sequential pattern
    test_cases.append(Int32(0xFEDCBA98))  # reverse sequential pattern
    test_cases.append(Int32(0x5A5A5A5A))  # checkerboard pattern
    
    # Expected results (from CUDA reference)
    var expected_results = List[List[Int8]]()
    
    # Test case 1: 0x00000000
    expected_results.append(List[Int8](-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2))
    
    # Test case 2: 0xFFFFFFFF
    expected_results.append(List[Int8](1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1))
    
    # Test case 3: 0x55555555
    expected_results.append(List[Int8](-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1))
    
    # Test case 4: 0xAAAAAAAA
    expected_results.append(List[Int8](0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    
    # Test case 5: 0x1B1B1B1B
    expected_results.append(List[Int8](1, 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, -1, -2, -2, -2, -2))
    
    # Test case 6: 0x12345678 (from actual CUDA output)
    expected_results.append(List[Int8](-2, 0, -2, 0, 0, -1, -1, -2, 1, -1, 1, -1, -1, -1, -2, -2))
    
    # Test case 7: 0x9ABCDEF0 (from actual CUDA output)
    expected_results.append(List[Int8](-2, 0, -2, 0, -2, 1, 1, 0, 1, -1, 1, -1, 1, 1, 0, 0))
    
    # Test case 8: 0xCAFEBABE (from actual CUDA output)
    expected_results.append(List[Int8](0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, -2, 0, 0, 1, 1))
    
    # Test case 9: 0xDEADBEEF (from actual CUDA output)
    expected_results.append(List[Int8](1, 0, -1, 0, 1, 1, 1, 1, 0, 1, 0, -1, 1, 0, 0, 1))
    
    # Test case 10: 0x13579BDF (from actual CUDA output)
    expected_results.append(List[Int8](1, 1, 1, 1, 1, 0, -1, -2, -1, -1, -1, -1, 1, 0, -1, -2))
    
    # Test case 11: 0x2468ACE0 (from actual CUDA output)
    expected_results.append(List[Int8](-2, -2, -2, -2, -2, 1, 0, -1, 0, 0, 0, 0, 1, 0, -1, -2))
    
    # Test case 12: 0x0F0F0F0F (from actual CUDA output)
    expected_results.append(List[Int8](1, 1, 1, 1, 1, 1, 1, 1, -2, -2, -2, -2, -2, -2, -2, -2))
    
    # Test case 13: 0xF0F0F0F0 (from actual CUDA output)
    expected_results.append(List[Int8](-2, -2, -2, -2, -2, -2, -2, -2, 1, 1, 1, 1, 1, 1, 1, 1))
    
    # Test case 14: 0x87654321 (from actual CUDA output)
    expected_results.append(List[Int8](-1, 1, -1, 1, -2, -2, -1, -1, 0, -2, 0, -2, -2, -1, -1, 0))
    
    # Test case 15: 0x11111111 (from actual CUDA output)
    expected_results.append(List[Int8](-1, -1, -1, -1, -2, -2, -2, -2, -1, -1, -1, -1, -2, -2, -2, -2))
    
    # Test case 16: 0x22222222 (from actual CUDA output)
    expected_results.append(List[Int8](0, 0, 0, 0, -2, -2, -2, -2, 0, 0, 0, 0, -2, -2, -2, -2))
    
    # Test case 17: 0x33333333 (from actual CUDA output)
    expected_results.append(List[Int8](1, 1, 1, 1, -2, -2, -2, -2, 1, 1, 1, 1, -2, -2, -2, -2))
    
    # Test case 18: 0x01234567 (from actual CUDA output)
    expected_results.append(List[Int8](1, -1, 1, -1, -1, -1, -2, -2, 0, -2, 0, -2, -1, -1, -2, -2))
    
    # Test case 19: 0xFEDCBA98 (from actual CUDA output)
    expected_results.append(List[Int8](-2, 0, -2, 0, 0, 0, 1, 1, -1, 1, -1, 1, 0, 0, 1, 1))
    
    # Test case 20: 0x5A5A5A5A (from actual CUDA output)
    expected_results.append(List[Int8](0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1))
    
    # Run tests
    for test_idx in range(len(test_cases)):
        print("테스트 케이스", test_idx + 1, ":")
        print("입력:", hex(test_cases[test_idx]))
        
        # Allocate output buffer
        var output = UnsafePointer[Int8].alloc(16)
        
        # Run decode function
        decode_i2s_to_i8s_cuda_compatible(test_cases[test_idx], output, 16)
        
        # Print results
        print("출력: ", end="")
        for i in range(16):
            print(Int(output[i]), end=" ")
        print()
        
        # Check results against expected CUDA results
        var is_valid = True
        for i in range(16):
            if output[i] != expected_results[test_idx][i]:
                is_valid = False
                break
        
        if is_valid:
            print("결과: ✓ 통과")
        else:
            print("결과: ✗ 실패")
            print("예상:", end=" ")
            for i in range(16):
                print(expected_results[test_idx][i], end=" ")
            print()
        
        print("----------------------------------------")
        
        # Free memory
        output.free()
    
    print("테스트 완료!")


fn main():
    test_decode()
