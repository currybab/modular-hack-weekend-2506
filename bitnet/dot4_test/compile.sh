#!/bin/bash

echo "=== Compiling CUDA dot4 test ==="
nvcc -std=c++17 -Xcudafe --diag_suppress=177 --compiler-options -fPIC -lineinfo test_dot4.cu -lcuda -gencode=arch=compute_80,code=compute_80 -o test_dot4_cuda

echo "=== Compiling Mojo dot4 test ==="
mojo build test_mojo_dot4.mojo -o test_dot4_mojo

echo "=== Compilation complete ==="
