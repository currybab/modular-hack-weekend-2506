fn dot4_manual(a0: Int8, a1: Int8, a2: Int8, a3: Int8, 
               b0: Int8, b1: Int8, b2: Int8, b3: Int8) -> Int32:
    """Manual 4-way dot product calculation."""
    return Int32(a0) * Int32(b0) + Int32(a1) * Int32(b1) + Int32(a2) * Int32(b2) + Int32(a3) * Int32(b3)

fn dot4_simd(a_vals: SIMD[DType.int8, 4], b_vals: SIMD[DType.int8, 4]) -> Int32:
    """SIMD 4-way dot product calculation (like in Mojo kernel)."""
    var a_int32 = a_vals.cast[DType.int32]()
    var b_int32 = b_vals.cast[DType.int32]()
    return (a_int32 * b_int32).reduce_add()

fn main():
    print("=== MOJO 4-way Dot Product Test (CPU) ===")
    
    # Test cases (same as CUDA)
    var test_cases = List[List[Int8]]()
    test_cases.append(List[Int8](-50, 104, -34, 71, 1, -1, -1, -1))   # a and b values
    test_cases.append(List[Int8](1, 2, 3, 4, 1, 1, 1, 1))
    test_cases.append(List[Int8](-1, -2, -3, -4, 1, -1, 1, -1))
    test_cases.append(List[Int8](127, -128, 0, 1, 1, 1, 0, -1))
    
    var expected = List[Int32](-191, 10, 2, -2)
    
    var all_correct = True
    
    for i in range(len(test_cases)):
        var test_case = test_cases[i]
        var a0 = test_case[0]
        var a1 = test_case[1] 
        var a2 = test_case[2]
        var a3 = test_case[3]
        var b0 = test_case[4]
        var b1 = test_case[5]
        var b2 = test_case[6]
        var b3 = test_case[7]
        
        # Manual calculation
        var manual_result = dot4_manual(a0, a1, a2, a3, b0, b1, b2, b3)
        
        # SIMD calculation (like in kernel)
        var a_simd = SIMD[DType.int8, 4](a0, a1, a2, a3)
        var b_simd = SIMD[DType.int8, 4](b0, b1, b2, b3)
        var simd_result = dot4_simd(a_simd, b_simd)
        
        var exp = expected[i]
        var manual_correct = (manual_result == exp)
        var simd_correct = (simd_result == exp)
        var results_match = (manual_result == simd_result)
        
        all_correct = all_correct and manual_correct and simd_correct and results_match
        
        print("Test", i, ":")
        print("  a=[", Int32(a0), ",", Int32(a1), ",", Int32(a2), ",", Int32(a3), "]")
        print("  b=[", Int32(b0), ",", Int32(b1), ",", Int32(b2), ",", Int32(b3), "]")
        print("  Manual:", manual_result, "(expected:", exp, ")", "✓" if manual_correct else "✗")
        print("  SIMD:  ", simd_result, "(expected:", exp, ")", "✓" if simd_correct else "✗") 
        print("  Match: ", "✓" if results_match else "✗")
        print("")
    
    print("MOJO dot4 all correct:", "✓" if all_correct else "✗")
    
    if not all_correct:
        print("Some tests failed!")
    else:
        print("All tests passed! MOJO SIMD matches CUDA __dp4a")
