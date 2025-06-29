# same as p15/op/conv1d.mojo
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from sys import sizeof, argv
from testing import assert_equal
from gpu.warp import shuffle_down

# BitLinear function implementation following conv1d format
alias BITLINEAR_TPB = 128
alias BITLINEAR_BLOCKS_PER_GRID = (1, 1)

@always_inline
fn decode_i2s_to_i8s(_i2s: UnsafePointer[Int32], i8s: UnsafePointer[Int8], n: Int = 16):
    """CUDA compatible decode_i2s_to_i8s implementation.
    
    Decodes packed 2-bit signed integers into 8-bit signed integers.
    CUDA groups by bit position across all bytes:
    - All 1st bit pairs from each byte → positions 0-3
    - All 2nd bit pairs from each byte → positions 4-7  
    - All 3rd bit pairs from each byte → positions 8-11
    - All 4th bit pairs from each byte → positions 12-15
    """
    
    var i2s = _i2s[]
    var output_idx = 0
    
    # Process each bit pair position (0-3) across all 4 bytes
    @parameter
    for bit_pair_idx in range(4):
        @parameter
        for byte_idx in range(4):
            if output_idx >= n:
                return
                
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
    TPB: Int = 128,
](
    output: LayoutTensor[mut=True, DType.bfloat16, out_layout],
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
    var in_thread_C_local = UnsafePointer[Int32].alloc(1)
    var A_local = UnsafePointer[Int8].alloc(K_per_loop)
    var B_reshape_local = UnsafePointer[Int32].alloc(1)
    var B_decode_local = UnsafePointer[Int8].alloc(K_per_loop)
    var red_buf0 = UnsafePointer[Int32].alloc(1)
    
    # Initialize accumulator
    in_thread_C_local[] = 0

    var local_x = thread_idx.x
    var local_y = thread_idx.y
    var global_x = block_idx.x
    var global_y = block_idx.y
    
    # Main computation loop (equivalent to #pragma unroll)
    # Use regular loop since range depends on runtime values
    @parameter
    for k_0 in range(K // (K_per_loop * K_block_size)):
        # Load A matrix data (vectorized load equivalent to int4)
        
        # === LOAD A MATRIX (INT8) ===
        # Vectorized load: Read 16 bytes (int4 = 128 bits) at once from A matrix
        # Each thread loads its own chunk based on thread ID and loop iteration
        var A_local_as_int32 = A_local.bitcast[Int32]()
        var input0_ptr = input0.ptr.bitcast[Int32]()
        var vec4 = input0_ptr.load[width=4]((k_0 * K_per_loop * K_block_size + local_x * K_per_loop) // 4)
        A_local_as_int32.store(vec4)
        
        # === LOAD B MATRIX (PACKED INT2) ===
        # Complex indexing to handle packed int2 format and tiled memory layout
        # B matrix is packed: 4 int2 values = 1 byte, so divide indices by 4
        var B_as_int32 = input1.ptr.bitcast[Int32]()
        
        B_reshape_local[] = B_as_int32.load((
            global_x * N_block_size * K // 4 +                   # Block offset
            (k_0 * K_block_size * K_per_loop * wmma_N // 4) +   # Loop iteration offset
            ((local_x >> 1) * wmma_K * wmma_N // 4) +           # Thread X offset (div by 2)
            ((local_y >> 3) * (wmma_K * wmma_N // 2) // 4) +     # Thread Y offset (div by 8)
            ((local_x & 1) * (wmma_K * wmma_N // 4) // 4) +      # Thread X remainder
            ((local_y & 7) * (wmma_K // 2) // 4)                 # Thread Y remainder 
        ) // 4)

        # === DECODE INT2 TO INT8 ===
        # Convert packed int2 values to separate int8 values for computation
        decode_i2s_to_i8s(B_reshape_local, B_decode_local, 16)
        
        # === DOT PRODUCT COMPUTATION ===
        # Match CUDA kernel: 4 iterations of __dp4a with accumulation
        @parameter
        for k_2_0 in range(4):
            # __dp4a equivalent: 4-way dot product of int8 values with accumulation
            # Load 4 int8 values and extend to int32 for arithmetic
            var a_int8 = A_local.offset(k_2_0 * 4).load[width=4]()
            var b_int8 = B_decode_local.offset(k_2_0 * 4).load[width=4]()
            # if local_x == 0 and local_y == 0 and global_x == 0 and global_y == 0:
            #     print("MOJO: k_2_0 =", k_2_0, "a_int8 =", a_int8, "b_int8 =", b_int8)
            
            # Convert int8 to int32 for proper signed arithmetic
            var a_int32 = a_int8.cast[DType.int32]()
            var b_int32 = b_int8.cast[DType.int32]()
            
            # Compute dot product: sum of element-wise products
            var dot4 = (a_int32 * b_int32).reduce_add()
            in_thread_C_local[0] += dot4  # ACCUMULATE like CUDA
            # if local_x == 0 and local_y == 0 and global_x == 0 and global_y == 0:
            #     print("MOJO: in_thread_C_local =", in_thread_C_local[0])

    # ==================== WARP-LEVEL REDUCTION ====================
    # Move thread-local result to reduction buffer
    red_buf0[0] = in_thread_C_local[0];
    # check red_buf0 is correct
    # if local_x == 0 and local_y == 0:
    #     print("MOJO: red_buf0[0] = " + red_buf0[0].__str__(), "local_x = ", local_x, "local_y = ", local_y, "global_x = ", global_x, "global_y = ", global_y)
    
    # Perform warp-level tree reduction using shuffle operations
    # Reduces K_block_size partial results into a single value
    var offset = K_block_size // 2
    while offset > 0:
        red_buf0[0] += shuffle_down(red_buf0[0], offset)
        offset //= 2
    
    # ==================== OUTPUT WRITING ====================
    # Calculate output position and scaling factor index  
    var out_idx = global_x * N_block_size + local_y;
    var ws_idx = out_idx // (N // ws_num);  # Weight scaling index
    
    # Only thread 0 in each warp writes the final result
    if local_x == 0:
        # Apply scaling: result / s[0] * ws[ws_idx], convert to bfloat16
        # Convert to Float32, apply scaling, then cast to bfloat16
        var result_float = Float32(red_buf0[0]) / Float32(s.ptr[0]) * Float32(ws.ptr[ws_idx])
        output.ptr[out_idx] = result_float.cast[DType.bfloat16]()

    


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
