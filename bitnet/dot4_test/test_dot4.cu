#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdio.h>

__global__ void verify_dot4_kernel(
    char* a_vals,  // 4 int8 values packed
    char* b_vals,  // 4 int8 values packed  
    int* results,  // output results
    int num_tests
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_tests) {
        // Load packed int8 values as int32
        int a_packed = *((int*)(a_vals + idx * 4));
        int b_packed = *((int*)(b_vals + idx * 4));
        
        // Use __dp4a with accumulator = 0
        int result = __dp4a(a_packed, b_packed, 0);
        results[idx] = result;
        
        // Debug output for first few results
        if (idx < 4) {
            char* a_ptr = (char*)&a_packed;
            char* b_ptr = (char*)&b_packed;
            printf("CUDA test %d: a=[%d,%d,%d,%d] b=[%d,%d,%d,%d] -> %d\\n", 
                   idx, 
                   (int)a_ptr[0], (int)a_ptr[1], (int)a_ptr[2], (int)a_ptr[3],
                   (int)b_ptr[0], (int)b_ptr[1], (int)b_ptr[2], (int)b_ptr[3],
                   result);
        }
    }
}

extern "C" {
    void verify_dot4_cuda(char* a_vals, char* b_vals, int* results, int num_tests) {
        char* d_a;
        char* d_b; 
        int* d_results;
        
        size_t a_size = num_tests * 4 * sizeof(char);
        size_t result_size = num_tests * sizeof(int);
        
        cudaMalloc(&d_a, a_size);
        cudaMalloc(&d_b, a_size);
        cudaMalloc(&d_results, result_size);
        
        cudaMemcpy(d_a, a_vals, a_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b_vals, a_size, cudaMemcpyHostToDevice);
        
        dim3 block(256);
        dim3 grid((num_tests + block.x - 1) / block.x);
        
        verify_dot4_kernel<<<grid, block>>>(d_a, d_b, d_results, num_tests);
        cudaDeviceSynchronize();
        
        cudaMemcpy(results, d_results, result_size, cudaMemcpyDeviceToHost);
        
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_results);
    }
}

int main() {
    std::cout << "=== CUDA 4-way Dot Product Test ===" << std::endl;
    
    // Test cases
    std::vector<std::vector<int8_t>> test_a = {
        {-50, 104, -34, 71},   // From debug output
        {1, 2, 3, 4},          // Simple positive
        {-1, -2, -3, -4},      // Negative values
        {127, -128, 0, 1}      // Edge cases
    };
    
    std::vector<std::vector<int8_t>> test_b = {
        {1, -1, -1, -1},
        {1, 1, 1, 1},
        {1, -1, 1, -1},
        {1, 1, 0, -1}
    };
    
    int num_tests = test_a.size();
    
    // Prepare input data
    std::vector<int8_t> a_vals(num_tests * 4);
    std::vector<int8_t> b_vals(num_tests * 4);
    std::vector<int32_t> results(num_tests);
    std::vector<int32_t> expected(num_tests);
    
    for (int i = 0; i < num_tests; i++) {
        for (int j = 0; j < 4; j++) {
            a_vals[i * 4 + j] = test_a[i][j];
            b_vals[i * 4 + j] = test_b[i][j];
        }
        
        // Calculate expected result
        int32_t sum = 0;
        for (int j = 0; j < 4; j++) {
            sum += (int32_t)test_a[i][j] * (int32_t)test_b[i][j];
        }
        expected[i] = sum;
    }
    
    // Call CUDA kernel
    verify_dot4_cuda((char*)a_vals.data(), (char*)b_vals.data(), results.data(), num_tests);
    
    // Check results
    bool all_correct = true;
    for (int i = 0; i < num_tests; i++) {
        bool correct = (results[i] == expected[i]);
        all_correct &= correct;
        
        std::cout << "Test " << i << ": ";
        std::cout << "[" << (int)test_a[i][0] << "," << (int)test_a[i][1] << "," << (int)test_a[i][2] << "," << (int)test_a[i][3] << "] ";
        std::cout << "· [" << (int)test_b[i][0] << "," << (int)test_b[i][1] << "," << (int)test_b[i][2] << "," << (int)test_b[i][3] << "] ";
        std::cout << "= " << results[i] << " (expected: " << expected[i] << ") ";
        std::cout << (correct ? "✓" : "✗") << std::endl;
    }
    
    std::cout << "\nCUDA __dp4a all correct: " << (all_correct ? "✓" : "✗") << std::endl;
    
    return all_correct ? 0 : 1;
}
