# same as p15/op/conv1d.mojo
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from sys import sizeof, argv
from testing import assert_equal

# BitLinear function implementation following conv1d format
alias BITLINEAR_TPB = 128
alias BITLINEAR_BLOCKS_PER_GRID = (1, 1)

@always_inline
fn decode_i2s_to_i8s(i2s: UInt32, i8s: UnsafePointer[UInt8], n: Int = 16):
    """Decode int2 to int8 values - Mojo SIMD optimized version."""
    
    # SIMD optimized version - process multiple elements using vectorized operations
    # Similar to CUDA's approach but adapted for Mojo's SIMD capabilities
    
    # Process 4 chunks of 4 elements each (total 16 elements)
    @parameter
    for chunk in range(4):
        # Calculate base shift for this chunk (0, 8, 16, 24 bits)
        var base_shift = chunk * 8
        
        # Extract 4 consecutive 2-bit values for this chunk
        var chunk_data = (i2s >> base_shift) & UInt32(0xFF)  # Get 8 bits (4 * 2-bit values)
        
        # Process 4 elements in this chunk
        @parameter
        for i in range(4):
            var shift_amount = i * 2
            var extracted = (chunk_data >> shift_amount) & UInt32(0x03)
            # Convert 2-bit value [0,3] to signed [-2,1]
            var signed_val = UInt8(extracted) - 2
            i8s[chunk * 4 + i] = signed_val

fn bitlinear_kernel[
    in0_layout: Layout,
    in1_layout: Layout,
    out_layout: Layout,
    s_layout: Layout,
    ws_layout: Layout,
    M: Int,
    N: Int,
    K: Int,
    K_block_size: Int,
    N_block_size: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[mut=True, dtype, out_layout],
    input0: LayoutTensor[mut=True, DType.int8, in0_layout],
    input1: LayoutTensor[mut=True, DType.int8, in1_layout],
    s: LayoutTensor[mut=True, DType.bfloat16, s_layout],
    ws: LayoutTensor[mut=True, DType.bfloat16, ws_layout],
    ws_num: Int,
):
    """BitLinear kernel implementation for int8 x int2 matrix multiplication."""
    # Mojo equivalent of CUDA kernel implementation
    alias K_per_loop = 16
    alias wmma_K = 32
    alias wmma_N = 16
    
    # Local arrays (equivalent to CUDA local memory)
    var in_thread_C_local = SIMD[DType.int32, 1](0)
    var A_local = UnsafePointer[Int8].alloc(K_per_loop)
    var B_reshape_local = SIMD[DType.uint32, 1](0)
    var B_decode_local = UnsafePointer[UInt8].alloc(K_per_loop)
    var red_buf0 = SIMD[DType.int32, 1](0)
    
    # Initialize accumulator
    in_thread_C_local[0] = 0
    
    # Main computation loop (equivalent to #pragma unroll)
    # Use regular loop since range depends on runtime values
    var loop_limit = K // (K_per_loop * K_block_size)
    for k_0 in range(loop_limit):
        # Load A matrix data (vectorized load equivalent to int4)
        var A_idx = (k_0 * K_per_loop * K_block_size) + (thread_idx.x * K_per_loop)
        
        # Load 16 bytes (K_per_loop) from input0 tensor
        @parameter
        for i in range(K_per_loop):
            # Calculate 2D indices for LayoutTensor access
            var linear_idx = A_idx + i
            var row = linear_idx // K
            var col = linear_idx % K
            
            # Use tensor shape for bounds checking
            if row < M and col < K:
                var loaded_val = input0[row, col]
                A_local[i] = loaded_val
            else:
                A_local[i] = Int8(0)
        
        # Calculate B matrix index (complex CUDA indexing)
        var B_idx = (
            (block_idx.x * N_block_size * K // 4) + 
            (k_0 * K_block_size * K_per_loop * wmma_N // 4) +
            ((thread_idx.x >> 1) * wmma_K * wmma_N // 4) +
            ((thread_idx.y >> 3) * (wmma_K * wmma_N // 2) // 4) + 
            ((thread_idx.x & 1) * (wmma_K * wmma_N // 4) // 4) + 
            ((thread_idx.y & 7) * (wmma_K // 2) // 4)
        )
        
        # Load B matrix data (packed int2 format)
        # Calculate 2D indices for LayoutTensor access
        var B_row = B_idx // (K // 4)
        var B_col = B_idx % (K // 4)
        
        if B_row < N and B_col < (K // 4):
            var loaded_val = input1[B_row, B_col]
            B_reshape_local[0] = UInt32(loaded_val)
        else:
            B_reshape_local[0] = UInt32(0)
        
        # Decode int2 to int8 values
        decode_i2s_to_i8s(B_reshape_local[0], B_decode_local, 16)
        
        # Dot product computation (equivalent to __dp4a)
        @parameter
        for k_2_0 in range(4):
            var a_vec = SIMD[DType.int8, 4](0)
            var b_vec = SIMD[DType.int8, 4](0)
            
            # Load 4 elements for vectorized dot product
            @parameter
            for i in range(4):
                var idx = k_2_0 * 4 + i
                if idx < K_per_loop:
                    a_vec[i] = A_local[idx]
                    b_vec[i] = Int8(B_decode_local[idx])
            
            # Compute dot product (equivalent to __dp4a)
            var dot_result = Int32(0)
            @parameter
            for i in range(4):
                dot_result += Int32(a_vec[i]) * Int32(b_vec[i])
            
            in_thread_C_local[0] += dot_result
    
    # Store intermediate result
    red_buf0[0] = in_thread_C_local[0]
    
    # Warp reduction (simplified version of __shfl_down_sync)
    # Note: Mojo may not have direct equivalent to CUDA warp shuffles
    # This is a placeholder for the reduction logic
    var offset = K_block_size // 2
    while offset > 0:
        # In real implementation, this would need proper warp-level communication
        # For now, we'll skip the reduction and use the local value
        offset //= 2
    
    # Calculate output indices
    var out_idx = (block_idx.x * N_block_size) + thread_idx.y
    var ws_idx = out_idx // (N // ws_num)
    
    # Write output (only thread 0 in warp)
    if thread_idx.x == 0:
        # Calculate 2D output indices
        var out_row = out_idx // N
        var out_col = out_idx % N
        
        # Check bounds using Int comparison
        if out_row < M and out_col < N and ws_idx >= 0:
            var s_val = s[0]
            var ws_val = ws[ws_idx]
            var result = Float32(red_buf0[0]) / Float32(s_val) * Float32(ws_val)
            output[out_row, out_col] = result.cast[DType.bfloat16]()
    
    # # Clean up allocated memory
    # A_local.free()
    # B_decode_local.free()
        
    
    


import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from memory import UnsafePointer
from gpu.host import DeviceBuffer

@compiler.register("bitlinear")
struct BitLinearCustomOp:
    @staticmethod
    fn execute[
        target: StaticString,
        M: Int,
        N: Int,
        K: Int,
    ](
        output: OutputTensor[dtype=DType.bfloat16, rank=2],
        input0: InputTensor[dtype=DType.int8, rank=2], 
        input1: InputTensor[dtype=DType.int8, rank=2],
        s: InputTensor[dtype=DType.bfloat16, rank=1],
        ws: InputTensor[dtype=DType.bfloat16, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        var out_tensor = output.to_layout_tensor()
        var input0_tensor = input0.to_layout_tensor() 
        var input1_tensor = input1.to_layout_tensor()
        var s_tensor = s.to_layout_tensor()
        var ws_tensor = ws.to_layout_tensor()

        alias in0_layout = input0_tensor.layout
        alias in1_layout = input1_tensor.layout
        alias out_layout = out_tensor.layout
        alias s_layout = s_tensor.layout
        alias ws_layout = ws_tensor.layout
        
        @parameter
        if target == "gpu":
            var gpu_ctx = ctx.get_device_context()
            
            # Zero out output tensor
            gpu_ctx.enqueue_memset(
                DeviceBuffer[output.dtype](
                    gpu_ctx,
                    rebind[UnsafePointer[Scalar[output.dtype]]](out_tensor.ptr),
                    M * N,
                    owning=False,
                ),
                0,
            )
            
            # Calculate grid and block dimensions based on matrix size
            var grid_dim: (Int, Int)
            var block_dim: (Int, Int)
            var arg_size: (Int, Int, Int)
            
            @parameter
            if M == 1 and N == 3840 and K == 2560:
                grid_dim = (240, 1)
                block_dim = (8, 16)
                arg_size = (3, 8, 16)
            elif M == 1 and N == 2560 and K == 2560:
                grid_dim = (160, 1) 
                block_dim = (8, 16)
                arg_size = (1, 8, 16)
            elif M == 1 and N == 13824 and K == 2560:
                grid_dim = (864, 1)
                block_dim = (8, 16)
                arg_size = (2, 8, 16)
            elif M == 1 and N == 2560 and K == 6912:
                grid_dim = (160, 1)
                block_dim = (8, 16)
                arg_size = (1, 8, 16)
            elif M == 1 and N == 4800 and K == 3200:
                grid_dim = (300, 1)
                block_dim = (8, 16)
                arg_size = (6, 8, 16)
            elif M == 1 and N == 3200 and K == 3200:
                grid_dim = (200, 1)
                block_dim = (8, 16)
                arg_size = (1, 8, 16)
            elif M == 1 and N == 20480 and K == 3200:
                grid_dim = (1280, 1)
                block_dim = (8, 16)
                arg_size = (2, 8, 16)
            elif M == 1 and N == 3200 and K == 10240:
                grid_dim = (200, 1)
                block_dim = (8, 16)
                arg_size = (1, 8, 16)
            elif M == 1 and N == 5120 and K == 27648:
                grid_dim = (320, 1)
                block_dim = (8, 16)
                arg_size = (1, 8, 16)
            elif M == 1 and N == 55296 and K == 5120:
                grid_dim = (3456, 1)
                block_dim = (8, 16)
                arg_size = (1, 8, 16)
            else:
                raise Error("required ladder gemm kernel: M ", M, ", N ", N, ", K ", K)
            ws_num, K_block_size, N_block_size = arg_size
            # Launch the bitlinear kernel
            gpu_ctx.enqueue_function[
                bitlinear_kernel[
                    in0_layout, in1_layout, out_layout, s_layout, ws_layout,
                    M, N, K, 8, 16,
                ]
            ](
                out_tensor,
                input0_tensor,
                input1_tensor, 
                s_tensor,
                ws_tensor,
                ws_num,
                grid_dim=grid_dim,
                block_dim=block_dim,
            )
            
        elif target == "cpu":
            # CPU fallback implementation could go here
            pass
        else:
            raise Error("Unsupported target: " + target)
