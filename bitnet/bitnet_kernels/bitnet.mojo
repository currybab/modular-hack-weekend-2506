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
    print("hi")
    
    


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
