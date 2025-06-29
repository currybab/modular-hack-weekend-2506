#!/bin/bash

echo "=== Running CUDA dot4 test ==="
./test_dot4_cuda

echo ""
echo "=== Running Mojo dot4 test ==="
./test_dot4_mojo

echo ""
echo "=== Test complete ==="
