NVCC = nvcc
CFLAGS = -std=c++17 -O2
ARCH = -arch=sm_80  # A100/H100용, 다른 GPU는 적절히 수정

test_decode: test_decode.cu
	$(NVCC) $(CFLAGS) $(ARCH) -o test_decode.out test_decode.cu

simple_debug: simple_debug.cu
	$(NVCC) $(CFLAGS) $(ARCH) -o simple_debug.out simple_debug.cu

debug_cuda_pattern: debug_cuda_pattern.cpp
	$(NVCC) $(CFLAGS) $(ARCH) -o debug_cuda_pattern.out debug_cuda_pattern.cpp

debug_cuda_decode: debug_cuda_decode.cu
	$(NVCC) $(CFLAGS) $(ARCH) -o debug_cuda_decode.out debug_cuda_decode.cu

clean:
	rm -f test_decode.out
	rm -f simple_debug.out

.PHONY: clean
